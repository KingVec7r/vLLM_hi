# SPDX-License-Identifier: Apache-2.0
"""TCP + shared-memory data transfer backend (default, CPU roundtrip).

KV bulk data is transferred via ``GPU->CPU(shm)->TCP->CPU(shm)->GPU``. This is
the default data plane; all performance optimizations (recv_into / large-block
sendall / madvise / NUMA binding, etc.) are preserved as-is.
"""

from __future__ import annotations

import asyncio
import pickle
import socket
import struct
from typing import Any

from vllm.logger import init_logger

from vllm_hi.migration.backends.base import KVTransferBackend, TransferContext
from vllm_hi.migration.backends.registry import register_backend
from vllm_hi.migration.runtime_config import PROBE_CONFIG

logger = init_logger("vllm.vllm_hi.migration.backends.tcp_shm")

# Message header: 4-byte magic + 8-byte length
_MAGIC = b"VHI1"
_MAGIC_BLOBS = b"VHI2"
_HEADER_SIZE = 12
_MAX_CHUNK = 64 * 1024 * 1024  # 64 MiB


def extract_kv_blobs(
    kv_cache: dict[int, dict[int, bytes]]
) -> list[bytes]:
    """Extract the bytes of ``{rank: {layer: bytes}}`` into ordered blobs, setting each to None in place.

    The returned order matches the fill order of :func:`restore_kv_blobs` (rank
    ascending, layer ascending).
    """
    blobs: list[bytes] = []
    for rank in sorted(kv_cache.keys()):
        layer_map = kv_cache[rank]
        for layer in sorted(layer_map.keys()):
            blobs.append(layer_map[layer])
            layer_map[layer] = None  # type: ignore[assignment]
    return blobs


def restore_kv_blobs(
    kv_cache: dict[int, dict[int, bytes]],
    blobs: list[bytes],
) -> None:
    """Fill blobs back into the ``{rank: {layer: None}}`` placeholders in order."""
    idx = 0
    for rank in sorted(kv_cache.keys()):
        layer_map = kv_cache[rank]
        for layer in sorted(layer_map.keys()):
            if idx < len(blobs):
                layer_map[layer] = blobs[idx]  # type: ignore[assignment]
                idx += 1


class TcpStateTransfer:
    """TCP wire-protocol implementation for state data (not a ``KVTransferBackend``).

    Note: this is a "direct TCP wire protocol", not a data-plane backend
    abstraction -- ``TcpShmTransferBackend`` uses it as the transport carrier. The
    name drops ``Backend`` to avoid confusion with ``KVTransferBackend``.

    Protocol:
        [4B magic][8B payload_len][payload bytes]
    where payload is pickle.dumps(data).
    """

    def __init__(self, buffer_size: int = _MAX_CHUNK) -> None:
        self.buffer_size = buffer_size

    # --- Send ---

    async def send(
        self, data: Any, dst_addr: tuple[str, int]
    ) -> None:
        """Send data to dst_addr."""
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None, pickle.dumps, data, pickle.HIGHEST_PROTOCOL
        )
        header = _MAGIC + struct.pack("<Q", len(payload))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            await loop.sock_connect(sock, dst_addr)
            await loop.sock_sendall(sock, header)
            await loop.sock_sendall(sock, payload)
        finally:
            sock.close()

    async def send_state(
        self,
        meta: Any,
        blobs: list[bytes],
        dst_addr: tuple[str, int],
    ) -> None:
        """Send "metadata + large-object segments" to dst_addr (protocol VHI2)."""
        loop = asyncio.get_running_loop()
        meta_bytes = await loop.run_in_executor(
            None, pickle.dumps, meta, pickle.HIGHEST_PROTOCOL
        )
        header = _MAGIC_BLOBS + struct.pack("<Q", len(meta_bytes))

        if PROBE_CONFIG.skip_transfer:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _MAX_CHUNK)
        except OSError:  # noqa: BLE001
            pass
        try:
            await loop.sock_connect(sock, dst_addr)
            await loop.sock_sendall(sock, header)
            await loop.sock_sendall(sock, meta_bytes)
            await loop.sock_sendall(sock, struct.pack("<Q", len(blobs)))
            for blob in blobs:
                await loop.sock_sendall(sock, struct.pack("<Q", len(blob)))
                await loop.run_in_executor(None, sock.sendall, blob)
            # Blocking-read the B-side ACK so we only return after B has fully
            # processed this message. This prevents a race between the final
            # incremental update and switch_service: with fire-and-forget, the
            # final incremental's data could be applied by B's data plane after
            # activate, leaving the restored request state inconsistent and stuck.
            ack = await loop.run_in_executor(None, self._recv_ack, sock)
            if isinstance(ack, dict):
                result = ack.get("result", {})
                if isinstance(result, dict) and not result.get("applied", False):
                    raise RuntimeError(
                        f"B-side data-plane restore failed: {result.get('error', 'unknown error')}"
                    )
        finally:
            sock.close()

    def _recv_ack(self, sock: socket.socket) -> Any:
        """Blocking-read the B-side ACK (called in a thread pool), returning the ACK dict."""
        sock.setblocking(True)

        def _recv_exact(n: int) -> bytearray:
            buf = bytearray(n)
            mv = memoryview(buf)
            got = 0
            while got < n:
                c = sock.recv_into(mv[got:], n - got)
                if c == 0:
                    raise ConnectionError("connection closed (no ACK)")
                got += c
            return buf

        magic = bytes(_recv_exact(4))
        (payload_len,) = struct.unpack("<Q", _recv_exact(8))
        payload = bytes(_recv_exact(payload_len))
        return pickle.loads(payload)

    def recv_state_blocking(self, sock: socket.socket) -> tuple[Any, list[bytearray]]:
        """Blocking-read "metadata + large-object segments" (called in a thread pool), returning (meta, blobs)."""
        sock.setblocking(True)

        def _recv_exact(n: int) -> bytearray:
            buf = bytearray(n)
            mv = memoryview(buf)
            got = 0
            while got < n:
                chunk = sock.recv_into(mv[got:], min(n - got, self.buffer_size))
                if chunk == 0:
                    raise ConnectionError("connection closed")
                got += chunk
            return buf

        magic = bytes(_recv_exact(4))
        if magic == _MAGIC:
            (payload_len,) = struct.unpack("<Q", _recv_exact(8))
            payload = _recv_exact(payload_len)
            return pickle.loads(bytes(payload)), []
        if magic != _MAGIC_BLOBS:
            raise ValueError(f"message header magic mismatch: {magic!r}")

        (meta_len,) = struct.unpack("<Q", _recv_exact(8))
        meta_bytes = bytes(_recv_exact(meta_len))
        meta = pickle.loads(meta_bytes)
        (blob_count,) = struct.unpack("<Q", _recv_exact(8))
        blobs: list[bytearray] = []
        for _ in range(blob_count):
            (blob_len,) = struct.unpack("<Q", _recv_exact(8))
            blobs.append(_recv_exact(blob_len))
        return meta, blobs


@register_backend
class TcpShmTransferBackend(KVTransferBackend):
    """TCP + shm data transfer backend (default, CPU roundtrip)."""

    name = "tcp_shm"
    gpu_direct = False
    cross_node = True
    priority = 10

    def __init__(self) -> None:
        self._transfer = TcpStateTransfer()

    async def send_state(
        self,
        ctx: TransferContext,
        meta: dict,
        kv_cache: dict[int, dict[int, Any]],
        dst_addr: tuple[str, int],
    ) -> int:
        """Extract KV blobs and send them over TCP."""
        blobs = extract_kv_blobs(kv_cache)
        data_size_bytes = sum(len(b) for b in blobs)
        try:
            await self._transfer.send_state(meta, blobs, dst_addr)
        finally:
            del blobs
        return data_size_bytes

    async def _handle_data(self, ctx: TransferContext, meta: dict) -> dict:
        """Restore one complete state (full/incremental). Subclasses may override for a different restore approach."""
        data_type = meta.get("type")
        receiver = ctx.receiver
        if receiver is None:
            logger.warning("[vllm_hi] data plane: receiver unavailable, dropping data")
            return {"applied": False, "error": "receiver unavailable"}
        if data_type == "full":
            await receiver.remap_and_restore(meta["snapshot"])
            logger.info("[vllm_hi] data plane: full state restored")
            return {"applied": True}
        elif data_type == "incremental":
            await receiver.apply_incremental(meta["update"])
            logger.info("[vllm_hi] data plane: incremental state applied")
            return {"applied": True}
        else:
            return {"applied": False, "error": f"unknown data type: {data_type}"}

    async def start_data_server(
        self,
        ctx: TransferContext,
        listen_addr: tuple[str, int],
    ) -> None:
        """Run the data-plane TCP server, receiving state data from A and restoring it.

        Uses an ``asyncio.Lock`` to serialize full/incremental state rebuilds,
        ensuring an increment is only applied after full restore completes and
        avoiding interleaving on the data plane. Also maintains a ``data_idle``
        event, which the activate endpoint uses to wait for all received
        data-plane messages to finish processing before registering the output
        consumer for the restored requests.
        """
        transfer = self._transfer
        state = ctx.state
        process_lock = asyncio.Lock()
        data_idle = asyncio.Event()
        data_idle.set()
        state.vllm_hi.data_lock = process_lock
        state.vllm_hi.data_idle = data_idle
        pending = 0

        async def handler(meta: dict) -> dict:
            return await self._handle_data(ctx, meta)

        def _send_ack_blocking(conn: socket.socket, result: dict) -> None:
            payload = pickle.dumps(
                {"ack": True, "result": result}, pickle.HIGHEST_PROTOCOL
            )
            header = b"VHI1" + struct.pack("<Q", len(payload))
            conn.sendall(header + payload)

        loop = asyncio.get_running_loop()

        async def handle_conn(conn: socket.socket) -> None:
            nonlocal pending
            try:
                data, blobs = await loop.run_in_executor(
                    None, transfer.recv_state_blocking, conn
                )
                if data.get("type") == "full":
                    restore_kv_blobs(data["snapshot"].kv_cache, blobs)
                elif data.get("type") == "incremental":
                    restore_kv_blobs(data["update"].kv_cache_delta, blobs)
                async with process_lock:
                    pending += 1
                    data_idle.clear()
                    try:
                        result = await handler(data)
                    finally:
                        pending -= 1
                        if pending == 0:
                            data_idle.set()
                await loop.run_in_executor(
                    None, _send_ack_blocking, conn, result
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("[vllm_hi] data-plane processing failed")
                try:
                    await loop.run_in_executor(
                        None, _send_ack_blocking, conn,
                        {"applied": False, "error": str(exc)},
                    )
                except Exception:
                    pass
            finally:
                conn.close()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(listen_addr)
        srv.listen(5)
        srv.setblocking(False)
        logger.info(
            "[vllm_hi] B-side data-plane TCP server listening on %s:%d",
            listen_addr[0], listen_addr[1],
        )
        try:
            while True:
                conn, _addr = await loop.sock_accept(srv)
                asyncio.ensure_future(handle_conn(conn))
        finally:
            srv.close()


__all__ = [
    "TcpStateTransfer",
    "TcpShmTransferBackend",
    "extract_kv_blobs",
    "restore_kv_blobs",
]
