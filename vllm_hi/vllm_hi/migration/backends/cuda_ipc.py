# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC (GPU-direct) data-plane backend.

KV bulk data is moved directly between GPU device memories via CUDA IPC, without
a GPU->CPU->CPU->GPU roundtrip. Auto-selected by :func:`select_backend` on the
same node when P2P is available.

Data path:
- A-side workers export each layer's KV tensor IPC handle via
  ``tensor.untyped_storage()._share_cuda_()`` (only ~64B each, sent with meta over
  the control TCP channel; KV bytes are not transferred);
- B-side workers open the shared device memory via
  ``torch.UntypedStorage._new_shared_cuda(...)`` and perform a GPU->GPU
  fancy-index copy according to block_mapping + head_shard_map.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_hi.migration.backends.base import TransferContext
from vllm_hi.migration.backends.registry import register_backend
from vllm_hi.migration.backends.tcp_shm import TcpShmTransferBackend

logger = init_logger("vllm.vllm_hi.migration.backends.cuda_ipc")


def _p2p_available() -> bool:
    """Probe whether P2P is supported between local GPUs (prerequisite for CUDA IPC direct transfer)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        n = torch.cuda.device_count()
        if n < 1:
            return False
        for i in range(n):
            for j in range(n):
                if i != j and not torch.cuda.can_device_access_peer(i, j):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


_DTYPE_BYTES = {
    "torch.float16": 2, "torch.half": 2, "torch.float32": 4, "torch.float": 4,
    "torch.bfloat16": 2, "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
    "torch.int8": 1, "torch.uint8": 1, "torch.int32": 4, "torch.int64": 8,
}


def _dtype_bytes(dtype_str: str) -> int:
    return _DTYPE_BYTES.get(dtype_str, 2)


def _estimate_size(meta: dict) -> int:
    """Estimate the byte count of this KV direct transfer (for logging and incremental convergence decisions).

    ``kv_ipc_handles`` holds each layer's full-tensor handle; summing ``shape`` x
    dtype size gives the full pre-allocated KV byte count, which is then scaled by
    the proportion of ``kv_cache_blocks`` (the blocks actually moved).
    """
    snapshot = meta.get("snapshot") or meta.get("update")
    handles = getattr(snapshot, "kv_ipc_handles", None) or {}
    if not handles:
        return 0
    total_bytes = 0
    total_blocks = 0
    for info in handles.values():
        shape = info.get("shape") or []
        if not shape:
            continue
        numel = 1
        for d in shape:
            numel *= d
        total_bytes += numel * _dtype_bytes(info.get("dtype", ""))
        if shape[0] > total_blocks:
            total_blocks = shape[0]
    blocks = getattr(snapshot, "kv_cache_blocks", None)
    if not blocks:
        # No used blocks (scheduler empty); 0 bytes actually moved.
        return 0
    if total_blocks > 0:
        total_bytes = int(total_bytes * len(blocks) / total_blocks)
    return total_bytes


@register_backend
class CudaIpcTransferBackend(TcpShmTransferBackend):
    """CUDA IPC (GPU-direct) data-plane backend."""

    name = "cuda_ipc"
    gpu_direct = True
    cross_node = False
    priority = 100

    @classmethod
    def available(cls) -> bool:
        return _p2p_available()

    async def send_state(
        self,
        ctx: TransferContext,
        meta: dict,
        kv_cache: dict[int, dict[int, Any]],
        dst_addr: tuple[str, int],
    ) -> int:
        """Send state: KV goes GPU-direct; only small meta (including handles) travels over the control TCP channel."""
        data_size = _estimate_size(meta)
        await self._transfer.send_state(meta, [], dst_addr)
        return data_size

    async def _handle_data(self, ctx: TransferContext, meta: dict) -> dict:
        """IPC restore: remap + restore scheduler + GPU->GPU copy of KV."""
        receiver = ctx.receiver
        if receiver is None:
            logger.warning("[vllm_hi] data plane: receiver unavailable, dropping data")
            return {"applied": False, "error": "receiver unavailable"}
        data_type = meta.get("type")
        if data_type == "full":
            await receiver.remap_and_restore_ipc(meta["snapshot"])
            logger.info("[vllm_hi] data plane (IPC): full state restored")
            return {"applied": True}
        elif data_type == "incremental":
            await receiver.apply_incremental_ipc(meta["update"])
            logger.info("[vllm_hi] data plane (IPC): incremental state applied")
            return {"applied": True}
        else:
            return {"applied": False, "error": f"unknown data type: {data_type}"}


__all__ = ["CudaIpcTransferBackend", "_p2p_available"]
