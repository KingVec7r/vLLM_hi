# SPDX-License-Identifier: Apache-2.0
"""StateReceiver: target-side (B) state receiver and restorer.

Receives state data transferred from A, performs remapping (handling topology
changes), and restores inference state.
"""

from __future__ import annotations

from typing import Any

from vllm_hi.common.rpc import (
    RPC_APPLY_SCHEDULER_INCREMENTAL,
    RPC_SET_KV_IPC,
    RPC_SET_SCHEDULER_STATE,
)
from vllm_hi.common.utils import get_device_cpu_affinity
from vllm_hi.migration.remapper import TopologyRemapper
from vllm_hi.migration.runtime_config import RUNTIME_CONFIG
from vllm_hi.migration.state_snapshot import (
    IncrementalStateUpdate,
    StateSnapshot,
)
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.migration.receiver")


_MADV_POPULATE_WRITE = 23


def _populate_shm_write(buf, size: int) -> None:
    """Preallocate shared-memory physical pages to eliminate the page-fault storm when writing shm.

    Writing shm goes through /dev/shm (tmpfs); the first write of each 4KB page
    triggers a page fault and the kernel allocates zero pages one by one, slowing
    memcpy and putting burst pressure on the whole machine's memory subsystem.
    Use madvise(MADV_POPULATE_WRITE) to preallocate in one batch, so subsequent
    memcpy no longer faults and shm write throughput improves several-fold
    (measured 1.6GB/s -> ~10GB/s).
    """
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        libc.madvise(
            ctypes.c_void_p(addr),
            ctypes.c_size_t(size),
            ctypes.c_int(_MADV_POPULATE_WRITE),
        )
    except Exception:  # noqa: BLE001
        logger.warning("[vllm_hi Receiver] failed to preallocate shm physical pages (ignored)")


class StateReceiver:
    """Target-side (B) state receiver and restorer.

    Holds an engine_client (AsyncLLM) reference and restores engine core and
    worker state via RPC.
    """

    def __init__(self, engine_client: Any) -> None:
        self.engine_client = engine_client
        self._remapper: TopologyRemapper | None = None
        self._current_snapshot: StateSnapshot | None = None

    def setup_remap(
        self,
        src_topology: dict[str, Any],
        dst_topology: dict[str, Any],
    ) -> None:
        """Set up the topology remapper."""
        self._remapper = TopologyRemapper(src_topology, dst_topology)

    async def remap_and_restore(self, snapshot: StateSnapshot) -> None:
        """Remap and restore state.

        1. TopologyRemapper remaps the block table (if topology changed)
        2. Restore KV Cache to the target GPU (via collective_rpc)
        3. Restore request execution state (via call_utility)
        4. Restore scheduler runtime
        """
        logger.info(
            "[vllm_hi Receiver] starting state restore: snapshot=%s, requests=%d, blocks=%d",
            snapshot.snapshot_id,
            len(snapshot.request_states),
            len(snapshot.block_table),
        )

        # 1. Topology remapping
        if self._remapper and self._remapper.needs_remap:
            snapshot = self._remapper.remap_snapshot(snapshot)
            logger.info("[vllm_hi Receiver] topology remapping done")

        # 2. Restore scheduler state (CPU side), obtaining the a_block -> b_block mapping
        block_mapping = await self._restore_scheduler_state(snapshot)

        # 3. Block-level KV Cache copy per the block mapping (GPU side)
        await self._restore_kv_cache(
            snapshot.kv_cache,
            block_mapping,
            snapshot.kv_cache_blocks,
            snapshot.head_shard_map,
            snapshot.kv_tensor_metas,
        )

        self._current_snapshot = snapshot
        logger.info("[vllm_hi Receiver] state restore done")

    async def apply_incremental(self, update: IncrementalStateUpdate) -> None:
        """Apply an incremental update, discarding stale information.

        1. Remove the requests and blocks in removed
        2. Add the requests and blocks in added
        3. Update the requests and blocks in modified
        4. Apply the KV Cache increment
        """
        logger.info(
            "[vllm_hi Receiver] applying increment: update=%s, "
            "reqs(+%d ~%d -%d), blocks(+%d ~%d -%d)",
            update.update_id,
            len(update.request_states_added),
            len(update.request_states_modified),
            len(update.request_states_removed),
            len(update.block_table_added),
            len(update.block_table_modified),
            len(update.block_table_removed),
        )

        # Topology remapping (if needed)
        if self._remapper and self._remapper.needs_remap:
            update = self._remapper.remap_incremental(update)

        # Apply scheduler increment (add/modify/remove), obtaining the full block mapping
        block_mapping = await self._apply_scheduler_incremental(update)

        # Apply KV Cache increment (per-block copy per the block mapping, increment blocks only)
        await self._restore_kv_cache(
            update.kv_cache_delta,
            block_mapping,
            update.kv_cache_blocks,
            update.head_shard_map,
            update.kv_tensor_metas,
        )

        # Update the current snapshot
        self._merge_incremental_into_current(update)

    async def _merge_incremental_into_current(
        self, update: IncrementalStateUpdate
    ) -> None:
        """Merge the increment (add/modify/remove) into ``_current_snapshot`` (shared by CPU/IPC paths)."""
        if self._current_snapshot is None:
            return
        for seq_id in update.request_states_removed:
            self._current_snapshot.request_states.pop(seq_id, None)
        for seq_id in update.block_table_removed:
            self._current_snapshot.block_table.pop(seq_id, None)
        self._current_snapshot.request_states.update(
            update.request_states_added
        )
        self._current_snapshot.request_states.update(
            update.request_states_modified
        )
        self._current_snapshot.block_table.update(
            update.block_table_added
        )
        self._current_snapshot.block_table.update(
            update.block_table_modified
        )

    async def remap_and_restore_ipc(self, snapshot: StateSnapshot) -> None:
        """CUDA IPC direct restore: remap + restore scheduler + GPU→GPU KV copy."""
        if self._remapper and self._remapper.needs_remap:
            snapshot = self._remapper.remap_snapshot(snapshot)
        block_mapping = await self._restore_scheduler_state(snapshot)
        await self._restore_kv_ipc(
            snapshot.kv_ipc_handles, block_mapping, snapshot.head_shard_map
        )
        self._current_snapshot = snapshot
        logger.info("[vllm_hi Receiver] IPC state restore done")

    async def apply_incremental_ipc(self, update: IncrementalStateUpdate) -> None:
        """CUDA IPC direct incremental restore: remap + apply scheduler increment + GPU→GPU KV copy."""
        if self._remapper and self._remapper.needs_remap:
            update = self._remapper.remap_incremental(update)
        block_mapping = await self._apply_scheduler_incremental(update)
        await self._restore_kv_ipc(
            update.kv_ipc_handles, block_mapping, update.head_shard_map,
            kv_cache_blocks=update.kv_cache_blocks,
        )
        self._merge_incremental_into_current(update)
        logger.info("[vllm_hi Receiver] IPC increment applied")

    async def _restore_kv_ipc(
        self,
        kv_ipc_handles: dict,
        block_mapping: list[list[int]],
        head_shard_map: dict[int, list] | None = None,
        kv_cache_blocks: list[int] | None = None,
    ) -> None:
        """Copy A-side GPU memory to the B-side GPU directly via CUDA IPC (no CPU relay).

        When kv_cache_blocks is non-empty (increment), copy only these newly added
        blocks (A-side physical block ids), avoiding re-copying old blocks.
        """
        if not kv_ipc_handles or not block_mapping:
            return
        if kv_cache_blocks is not None:
            new_blocks = set(kv_cache_blocks)
            block_mapping = [p for p in block_mapping if p[0] in new_blocks]
        if not block_mapping:
            return
        shard_map_ser = {
            int(k): [vars(s) if hasattr(s, "__dict__") else s for s in v]
            for k, v in (head_shard_map or {}).items()
        }
        import time as _time
        _t0 = _time.perf_counter()
        await self.engine_client.engine_core.collective_rpc_async(
            method=RPC_SET_KV_IPC,
            args=(kv_ipc_handles, shard_map_ser, block_mapping),
        )
        logger.info(
            "[vllm_hi Receiver] IPC KV copy of %d block mappings took %.3fs",
            len(block_mapping), _time.perf_counter() - _t0,
        )

    async def _restore_kv_cache(
        self,
        kv_cache: dict[int, dict[int, bytes]],
        block_mapping: list[list[int]],
        kv_cache_blocks: list[int] | None = None,
        head_shard_map: dict[int, list] | None = None,
        kv_tensor_metas: dict[int, list] | None = None,
    ) -> None:
        """Copy A's KV cache block-by-block to B's target GPU per the block mapping.

        kv_cache_blocks: the A-side physical block order corresponding to dim 0 of
        each rank's tensor in kv_cache (non-empty during incremental transfer,
        meaning only these blocks were transferred; None for full, meaning dim 0 =
        all blocks).

        head_shard_map: {dst_global_rank: [HeadShard, ...]}, the head-shard mapping
        for TP/PP changes.
        kv_tensor_metas: {src_global_rank: [KVTensorMeta, ...]}, per-layer buffer
        layout metadata.

        Each source global rank writes one shm (``src_world`` copies total); the
        worker locates the shm by ``src_rank`` (source global rank) in the shard and
        "splits heads before moving to GPU", eliminating the redundancy of writing
        one copy per rank and the redundancy of fully moving then splitting heads.
        """
        if not kv_cache or not block_mapping:
            return
        src_tp = self._remapper.src_tp if self._remapper else 1
        dst_tp = self._remapper.dst_tp if self._remapper else 1
        src_pp = self._remapper.src_pp if self._remapper else 1
        dst_pp = self._remapper.dst_pp if self._remapper else 1
        from multiprocessing import shared_memory

        rank_shm: dict[int, str] = {}
        shm_names: list[str] = []
        # cloudpickle can serialize dataclasses, but worker processes do not depend on vllm_hi, so convert uniformly to dict.
        shard_map_ser = {
            int(k): [vars(s) if hasattr(s, "__dict__") else s for s in v]
            for k, v in (head_shard_map or {}).items()
        }
        metas_ser = {
            int(k): [vars(m) if hasattr(m, "__dict__") else m for m in v]
            for k, v in (kv_tensor_metas or {}).items()
        }
        import time

        try:
            # Write only src_tp shm copies: each corresponds to one source rank's full buffer.
            t0 = time.perf_counter()
            total_bytes = 0
            import torch

            # Writing shm is a relay copy from "socket receive buffer → shared
            # memory". Using torch copy_ multi-thread (OpenMP spin) to write was
            # measured at ~4.5GB/s@4 threads, but the OpenMP thread pool's
            # busy-wait/spin contends for CPU with the co-located A-side inference
            # (sampling/detokenize), causing A-side token throughput to drop to zero
            # at the tail of each transfer round.
            # Instead use numpy single-thread chunked memcpy (~10GB/s warm): it uses
            # a single core, and inter-chunk sleep yields CPU back to A, eliminating
            # CPU contention at the root.
            import os

            import numpy as np

            shm_chunk_bytes = RUNTIME_CONFIG.shm_chunk_bytes
            shm_yield_seconds = RUNTIME_CONFIG.shm_yield_seconds
            if shm_chunk_bytes <= 0:
                shm_chunk_bytes = 256 * 1024 * 1024
            if shm_yield_seconds < 0:
                shm_yield_seconds = 0.0

            # Pin shm writes to the B-side GPU's NUMA node, spatially isolating them
            # from A-side inference and eliminating the CPU/memory-bandwidth
            # contention caused by memcpy/page-fault drifting to A's node.
            old_affinity = os.sched_getaffinity(0)
            shm_cpu_affinity = get_device_cpu_affinity()
            if shm_cpu_affinity:
                try:
                    os.sched_setaffinity(0, shm_cpu_affinity)
                except OSError:
                    shm_cpu_affinity = None

            try:
                for rank in sorted(kv_cache.keys()):
                    rank_data = kv_cache[rank]
                    data = rank_data.get(0)
                    if data is None:
                        data = next(iter(rank_data.values()), None)
                    if not data:
                        continue
                    shm = shared_memory.SharedMemory(create=True, size=len(data))
                    dst_buf = None
                    src_buf = None
                    try:
                        _populate_shm_write(shm.buf, len(data))
                        dst_buf = np.frombuffer(shm.buf, dtype=np.uint8)
                        src_buf = np.frombuffer(data, dtype=np.uint8)
                        total = len(data)
                        offset = 0
                        while offset < total:
                            end = min(offset + shm_chunk_bytes, total)
                            dst_buf[offset:end] = src_buf[offset:end]
                            offset = end
                            if offset < total:
                                time.sleep(shm_yield_seconds)
                    finally:
                        if dst_buf is not None:
                            del dst_buf
                        if src_buf is not None:
                            del src_buf
                    shm.close()
                    rank_shm[rank] = shm.name
                    shm_names.append(shm.name)
                    total_bytes += len(data)
            finally:
                if shm_cpu_affinity:
                    try:
                        os.sched_setaffinity(0, old_affinity)
                    except OSError:
                        pass
            t1 = time.perf_counter()
            logger.info(
                "[vllm_hi Receiver] wrote %d shm copies (src_tp=%d, src_pp=%d): %.2fs, "
                "total data %.1f MB, chunk=%dMB, yield=%.1fms",
                len(rank_shm), src_tp, src_pp, t1 - t0,
                total_bytes / 1024 / 1024,
                shm_chunk_bytes // (1024 * 1024), shm_yield_seconds * 1000,
            )

            from vllm_hi.migration.restore_worker import set_kv_cache

            await self.engine_client.engine_core.collective_rpc_async(
                method=set_kv_cache,
                args=(
                    rank_shm, shard_map_ser, metas_ser,
                    src_tp, dst_tp, block_mapping, kv_cache_blocks,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi Receiver] failed to restore KV Cache: %s", exc)
        finally:
            # Clean up shared memory
            for name in shm_names:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except (OSError, FileNotFoundError):
                    pass

    async def _restore_scheduler_state(
        self, snapshot: StateSnapshot
    ) -> list[list[int]]:
        """Restore scheduler state (requests, block table, queues), returning the block mapping."""
        try:
            state = {
                "request_states": {
                    k: v.__dict__ for k, v in snapshot.request_states.items()
                },
                "block_table": {
                    k: v.__dict__ for k, v in snapshot.block_table.items()
                },
                "running_seq_ids": snapshot.running_seq_ids,
                "waiting_seq_ids": snapshot.waiting_seq_ids,
            }
            result = await self.engine_client.engine_core.call_utility_async(
                RPC_SET_SCHEDULER_STATE, state
            )
            return list((result or {}).get("block_mapping", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi Receiver] failed to restore scheduler state: %s", exc)
            return []

    async def _apply_scheduler_incremental(
        self, update: IncrementalStateUpdate
    ) -> list[list[int]]:
        """Apply scheduler increment, returning the current full block mapping."""
        try:
            state = {
                "request_states_added": {
                    k: v.__dict__
                    for k, v in update.request_states_added.items()
                },
                "request_states_modified": {
                    k: v.__dict__
                    for k, v in update.request_states_modified.items()
                },
                "request_states_removed": update.request_states_removed,
                "block_table_added": {
                    k: v.__dict__ for k, v in update.block_table_added.items()
                },
                "block_table_modified": {
                    k: v.__dict__ for k, v in update.block_table_modified.items()
                },
                "block_table_removed": update.block_table_removed,
                "running_seq_ids": update.running_seq_ids,
                "waiting_seq_ids": update.waiting_seq_ids,
            }
            result = await self.engine_client.engine_core.call_utility_async(
                RPC_APPLY_SCHEDULER_INCREMENTAL, state
            )
            return list((result or {}).get("block_mapping", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[vllm_hi Receiver] failed to apply scheduler increment: %s", exc
            )
            return []
