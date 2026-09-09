# SPDX-License-Identifier: Apache-2.0
"""StateCollector: source-side (A) state collector.

Collects full/incremental state data from the vLLM engine_core.
Accesses the engine core's scheduler state via call_utility_async,
and the worker's KV Cache tensors via collective_rpc_async.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from vllm_hi.common.rpc import (
    RPC_COLLECT_KV_CACHE,
    RPC_EXPORT_KV_IPC,
    RPC_GET_SCHEDULER_STATE,
    RPC_TRY_GET_RESULT,
)
from vllm_hi.common.utils import generate_id
from vllm_hi.migration.filter import MigrateAllFilter, load_request_filter
from vllm_hi.migration.runtime_config import PROBE_CONFIG, RUNTIME_CONFIG
from vllm_hi.migration.state_snapshot import (
    BlockTableInfo,
    IncrementalStateUpdate,
    KVTensorMeta,
    RequestState,
    StateSnapshot,
)
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.migration.collector")

# Chunked collection granularity: each collection processes at most this many
# physical blocks of KV.
# Trade-off: chunk too small -> the fixed per-RPC overhead is amplified into
# dozens of RPCs, making migration very slow;
#            chunk too large -> a single transfer moves a lot of data, raising peak memory.
# After moving collection to the background, the inference thread is no longer
# blocked by collection, so there is no need to rely on "small-chunk interleaving"
# to smooth token jitter. Bumped from 256 to 512 to reduce RPC round-trips and
# the cross-chunk aggregation copy overhead.
_KV_COLLECT_CHUNK_BLOCKS = RUNTIME_CONFIG.kv_collect_chunk_blocks
# Large-chunk collection granularity used by the final increment (A is paused,
# so no inference needs to stay smooth): collect in one shot to eliminate
# multiple RPC round-trips and the extra memory copy of cross-chunk aggregation.
_KV_COLLECT_CHUNK_BLOCKS_LARGE = RUNTIME_CONFIG.kv_collect_chunk_blocks_large

# Background collection polling backoff: initial interval 20ms, exponential
# backoff with poll count, capped at 200ms.
# Compared to a fixed 5ms poll, this cuts collective_rpc_async broadcast count
# by an order of magnitude and reduces interrupts injected into the worker busy
# loop. Collection tasks take seconds (GB-scale), so a 20~200ms query latency
# is negligible.
_POLL_INTERVAL_INIT = 0.020
_POLL_INTERVAL_MAX = 0.200
_POLL_INTERVAL_FACTOR = 2.0

# Empirical overall collection rate (bytes/s) used to estimate completion time:
# the serial bottleneck of rate-limited D2H + pinned→shm encode. A conservative
# value lets us sleep until near completion before the first poll, minimizing
# RPC interleaving.
_COMPLETION_ESTIMATE_RATE = 20e9


def _poll_interval(poll_count: int) -> float:
    """Compute the backoff interval from the poll count (initial 20ms, exponential backoff capped at 200ms)."""
    interval = _POLL_INTERVAL_INIT * (_POLL_INTERVAL_FACTOR ** poll_count)
    return min(interval, _POLL_INTERVAL_MAX)


class StateCollector:
    """Source-side (A) state collector.

    Holds an engine_client (AsyncLLM) reference and collects engine core and
    worker state via RPC.
    """

    def __init__(self, engine_client: Any) -> None:
        self.engine_client = engine_client
        self._last_snapshot: StateSnapshot | None = None
        self._last_request_ids: set[str] = set()
        self._pending_shm: list = []
        self._kv_tensor_metas: dict[int, list[KVTensorMeta]] = {}
        # Request filter (default migrates all; extensible via VLLM_HI_REQUEST_FILTER)
        self.filter = load_request_filter(RUNTIME_CONFIG.request_filter)

    def _filter_scheduler_state(self, scheduler_state: dict[str, Any]) -> dict[str, Any]:
        """Filter scheduler state by the request filter, keeping only requests to migrate.

        Filtered-out requests have no KV collected and are not migrated; they
        continue to be processed on the A instance.
        """
        request_states = scheduler_state["request_states"]
        block_table = scheduler_state["block_table"]
        keep_ids = {
            seq_id
            for seq_id, rs in request_states.items()
            if self.filter.should_migrate(seq_id, rs)
        }
        return {
            "request_states": {
                k: v for k, v in request_states.items() if k in keep_ids
            },
            "block_table": {
                k: v for k, v in block_table.items() if k in keep_ids
            },
            "running_seq_ids": [
                s for s in scheduler_state["running_seq_ids"] if s in keep_ids
            ],
            "waiting_seq_ids": [
                s for s in scheduler_state["waiting_seq_ids"] if s in keep_ids
            ],
        }

    def release_kv_cache_shm(self) -> None:
        """Release shared memory left over from the collection phase (called after the data has been sent).

        ``gc.collect()`` is no longer called: gc's stop-the-world holds the GIL
        and blocks the API server event loop for seconds, interrupting token
        streaming (the root cause of the throughput drop). Instead we rely on the
        explicit ``del`` in ``_collect_worker_kv_cache_chunked`` /
        ``_aggregate_kv_chunks`` to release the memoryview and torch tensor
        exported pointers; once the refcount hits zero, ``shm.close()`` no longer
        raises BufferError.
        """
        for shm in self._pending_shm:
            try:
                shm.close()
            except BufferError:
                pass
            try:
                shm.unlink()
            except (OSError, FileNotFoundError):
                pass
        self._pending_shm = []

    async def collect_full(
        self, src_topology: dict[str, Any],
        chunk_size: int = _KV_COLLECT_CHUNK_BLOCKS,
        boost_cpu: bool = False,
        kv_mode: str = "cpu",
    ) -> StateSnapshot:
        """Collect full state.

        Fetches the scheduler's request states and block table metadata via
        call_utility_async, and the worker's KV Cache tensors via
        collective_rpc_async.

        boost_cpu: set True when A is paused, allowing the worker to briefly
        raise the torch thread count to speed up GPU→CPU copy and encode and
        shorten downtime; keep False while A is serving to avoid contention.
        kv_mode: "cpu" (default, D2H collection to CPU) / "ipc" (export CUDA IPC handles).
        """
        snapshot_id = generate_id("snap")
        timestamp = time.time()

        # 1. Collect scheduler state (CPU side, in the engine core process)
        scheduler_state = await self._collect_scheduler_state()
        # Filter by request filter (default migrates all)
        scheduler_state = self._filter_scheduler_state(scheduler_state)

        # 2. Collect worker KV Cache (GPU side, in the worker process).
        # Collect only the physical blocks actually referenced in the block_table
        # rather than all preallocated blocks, drastically reducing the amount of
        # GPU->CPU copy and network transfer.
        used_blocks = sorted({
            b for bt in scheduler_state["block_table"].values()
            for b in bt.block_ids
        })

        kv_cache: dict[int, dict[int, Any]] = {}
        kv_ipc_handles: dict = {}
        if kv_mode == "ipc":
            kv_ipc_handles = await self._collect_worker_kv_ipc()
        else:
            kv_cache = await self._collect_worker_kv_cache_chunked(
                used_blocks, chunk_size, boost_cpu
            )

        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            src_topology=src_topology,
            kv_cache=kv_cache,
            kv_cache_blocks=used_blocks,
            kv_tensor_metas=self._kv_tensor_metas,
            kv_ipc_handles=kv_ipc_handles,
            block_table=scheduler_state["block_table"],
            request_states=scheduler_state["request_states"],
            running_seq_ids=scheduler_state["running_seq_ids"],
            waiting_seq_ids=scheduler_state["waiting_seq_ids"],
        )
        self._last_snapshot = snapshot
        self._last_request_ids = set(snapshot.request_states.keys())
        return snapshot

    async def collect_incremental(
        self, chunk_size: int = _KV_COLLECT_CHUNK_BLOCKS,
        boost_cpu: bool = False,
        kv_mode: str = "cpu",
    ) -> IncrementalStateUpdate:
        """Collect incremental state.

        Compares against the last snapshot and collects only what changed.

        boost_cpu: set True when A is paused, allowing the worker to briefly
        raise the torch thread count to speed up GPU→CPU copy and encode and
        shorten downtime; keep False while A is serving to avoid contention.
        kv_mode: "cpu" (default) / "ipc" (export CUDA IPC handles).
        """
        if self._last_snapshot is None:
            raise RuntimeError("full collection must be performed before incremental collection")

        update_id = generate_id("upd")
        timestamp = time.time()
        base_snapshot_id = self._last_snapshot.snapshot_id

        # Collect current scheduler state
        scheduler_state = await self._collect_scheduler_state()
        # Filter by request filter (default migrates all), keeping the same filter scope as the last snapshot
        scheduler_state = self._filter_scheduler_state(scheduler_state)
        current_requests = scheduler_state["request_states"]
        current_block_table = scheduler_state["block_table"]
        current_ids = set(current_requests.keys())
        last_ids = self._last_request_ids

        # Incremental request states
        added_ids = current_ids - last_ids
        removed_ids = last_ids - current_ids
        modified_ids = set()
        for seq_id in current_ids & last_ids:
            old = self._last_snapshot.request_states.get(seq_id)
            new = current_requests.get(seq_id)
            if old and new and (
                len(new.output_token_ids) != len(old.output_token_ids)
                or new.num_computed_tokens != old.num_computed_tokens
                or new.status != old.status
            ):
                modified_ids.add(seq_id)

        # Incremental block table
        bt_added = {
            k: v for k, v in current_block_table.items()
            if k not in self._last_snapshot.block_table
        }
        bt_modified = {}
        bt_removed = []
        for k, v in current_block_table.items():
            if k in self._last_snapshot.block_table:
                old = self._last_snapshot.block_table[k]
                if (
                    list(v.block_ids) != list(old.block_ids)
                    or v.ref_count != old.ref_count
                ):
                    bt_modified[k] = v
        for k in self._last_snapshot.block_table:
            if k not in current_block_table:
                bt_removed.append(k)

        # Incremental KV Cache: collect only the newly added physical blocks this
        # round (all blocks of new requests + newly added blocks of existing
        # requests), avoiding a full clone of the entire KV cache on every
        # increment that would block A-side inference.
        # Blocks already transferred have unchanged KV, so no re-collection.
        incr_block_ids: list[int] = []
        for bt in bt_added.values():
            incr_block_ids.extend(bt.block_ids)
        for k, v in bt_modified.items():
            old = self._last_snapshot.block_table.get(k)
            old_ids = set(old.block_ids) if old else set()
            incr_block_ids.extend(b for b in v.block_ids if b not in old_ids)
        # Deduplicate and keep a stable order
        incr_block_ids = sorted(set(incr_block_ids))

        kv_cache_delta: dict[int, dict[int, Any]] = {}
        kv_ipc_handles: dict = {}
        if kv_mode == "ipc":
            kv_ipc_handles = await self._collect_worker_kv_ipc()
        else:
            kv_cache_delta = await self._collect_worker_kv_cache_chunked(
                incr_block_ids, chunk_size, boost_cpu
            )

        update = IncrementalStateUpdate(
            update_id=update_id,
            timestamp=timestamp,
            base_snapshot_id=base_snapshot_id,
            kv_cache_delta=kv_cache_delta,
            kv_cache_blocks=incr_block_ids,
            kv_tensor_metas=self._kv_tensor_metas,
            kv_ipc_handles=kv_ipc_handles,
            block_table_added=bt_added,
            block_table_modified=bt_modified,
            block_table_removed=bt_removed,
            request_states_added={
                k: v for k, v in current_requests.items() if k in added_ids
            },
            request_states_modified={
                k: v for k, v in current_requests.items() if k in modified_ids
            },
            request_states_removed=sorted(removed_ids),
            running_seq_ids=scheduler_state["running_seq_ids"],
            waiting_seq_ids=scheduler_state["waiting_seq_ids"],
        )

        # Update last state
        self._last_snapshot.request_states = dict(current_requests)
        self._last_snapshot.block_table = dict(current_block_table)
        self._last_request_ids = current_ids
        return update

    async def _collect_scheduler_state(self) -> dict[str, Any]:
        """Collect scheduler state via call_utility_async (RPC method injected by a patch).

        On failure, raise directly and let collect_full/collect_incremental bubble
        it up to activate_endpoint, which returns ``success=False`` — this keeps
        semantics consistent with KV collection failure and avoids the correctness
        risk of "scheduler state silently lost; migration succeeds but all requests
        vanish".
        """
        engine_core = self.engine_client.engine_core
        state = await engine_core.call_utility_async(RPC_GET_SCHEDULER_STATE)
        # Deserialize into dataclasses
        request_states = {
            seq_id: RequestState(**rs)
            for seq_id, rs in state.get("request_states", {}).items()
        }
        block_table = {
            seq_id: BlockTableInfo(**bt)
            for seq_id, bt in state.get("block_table", {}).items()
        }
        return {
            "request_states": request_states,
            "block_table": block_table,
            "running_seq_ids": state.get("running_seq_ids", []),
            "waiting_seq_ids": state.get("waiting_seq_ids", []),
        }

    async def _collect_worker_kv_cache(
        self, block_ids: list[int] | None = None,
        boost_cpu: bool = False,
    ) -> tuple[dict[int, dict[int, bytes]], dict[int, list[KVTensorMeta]]]:
        """Collect a worker's KV Cache in one shot (contiguous memory buffer + metadata, no pickle).

        Dispatch + polling via a worker background thread and a separate stream
        keeps the GPU→CPU copy and CPU encode-to-shm from blocking the inference
        step. This is the sole (default) KV collection implementation.

        Returns (kv_cache, metas):
          kv_cache: {rank: {0: buffer}}, where buffer is the contiguous memory view
          of all layers for that rank.
          metas: {rank: [KVTensorMeta, ...]}, the layout of each layer tensor
          within the buffer.
        """
        if block_ids is not None:
            block_ids = list(block_ids)
        engine_core = self.engine_client.engine_core
        # 1. Lightweight dispatch (compute stream record_event + enqueue, returns task_id in microseconds).
        dispatched = await engine_core.collective_rpc_async(
            method=RPC_COLLECT_KV_CACHE,
            args=(block_ids, boost_cpu),
        )
        # All workers have no KV (pure embedding / empty kv_caches): assemble an empty result directly.
        if all(
            isinstance(r, dict) and not r.get("task_id") for r in dispatched
        ):
            return self._assemble_kv_cache(dispatched)

        # 2. Poll for completion (broadcast with no args; each worker queries its
        # own single active task result).
        # Before the first poll, sleep until near completion based on the empirical
        # completion time, compressing RPC interleaving to 1~2 times; after that,
        # exponential backoff as a fallback (from 20ms, capped at 200ms).
        if PROBE_CONFIG.skip_poll:
            results = await engine_core.collective_rpc_async(
                method=RPC_TRY_GET_RESULT,
                args=(),
            )
            return self._assemble_kv_cache(results)

        first_sleep = self._estimate_completion(block_ids)
        poll_count = 0
        while True:
            results = await engine_core.collective_rpc_async(
                method=RPC_TRY_GET_RESULT,
                args=(),
            )
            if all(isinstance(r, dict) for r in results):
                break
            if poll_count == 0:
                await asyncio.sleep(first_sleep)
            else:
                await asyncio.sleep(_poll_interval(poll_count - 1))
            poll_count += 1

        # 3. Assemble the result
        return self._assemble_kv_cache(results)

    async def _collect_worker_kv_ipc(self) -> dict:
        """Export the worker's KV Cache CUDA IPC handles (direct GPU, no CPU transfer).

        Each worker uses ``tensor.untyped_storage()._share_cuda_()`` to export the
        IPC handle (with shape/stride/dtype) of each of its KV tensor layers; the
        collector aggregates them into a flat dict keyed by
        ``(src_global_rank, global_layer_idx)`` (the ``enumerate`` index is the
        global worker rank, consistent with ``_assemble_kv_cache`` on the CPU
        collection path). It also fills ``_kv_tensor_metas`` (with shape, so the
        B-side remapper can compute TP head shards). Returns:
            {(src_global_rank, global_layer_idx): handle_info}
        """
        results = await self.engine_client.engine_core.collective_rpc_async(
            method=RPC_EXPORT_KV_IPC,
            args=(),
        )
        handles: dict = {}
        metas: dict[int, list[KVTensorMeta]] = {}
        for rank, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            if result.get("error"):
                raise RuntimeError(f"KV IPC export failed: {result['error']}")
            layer_metas: list[KVTensorMeta] = []
            for layer_idx in sorted((result.get("layers") or {}).keys()):
                info = result["layers"][layer_idx]
                handles[(rank, int(layer_idx))] = info
                layer_metas.append(KVTensorMeta(
                    layer_idx=int(layer_idx),
                    shape=list(info.get("shape") or []),
                    dtype=info.get("dtype", ""),
                ))
            metas[rank] = layer_metas
        self._kv_tensor_metas = metas
        return handles

    def _estimate_block_bytes(self) -> float | None:
        """Infer the per-block byte count from collected KV tensor metas (averaged across ranks/layers).

        Available only after at least one collection completes (``_kv_tensor_metas``
        is filled); returns ``None`` before the first full collection, and the
        caller falls back to an empirical default.
        """
        total = 0
        total_blocks = 0
        for metas in self._kv_tensor_metas.values():
            for m in metas:
                if m.nbytes > 0 and m.shape:
                    total += m.nbytes
                    total_blocks += int(m.shape[0])
        if total_blocks > 0:
            return total / total_blocks
        return None

    def _estimate_completion(self, block_ids: list[int] | None) -> float:
        """Estimate collection completion time (seconds), used to sleep until near completion before the first poll.

        Derived from the collected byte count (blocks × per-block bytes) and the
        empirical overall rate of rate-limited D2H + encode; when it cannot be
        estimated, return a conservative default and let the exponential backoff
        cover the rest.
        """
        if block_ids is None:
            return 2.0
        n_blocks = len(block_ids)
        block_bytes = self._estimate_block_bytes()
        if block_bytes is None or block_bytes <= 0:
            block_bytes = 4096.0
        data_bytes = n_blocks * block_bytes
        return max(0.05, min(2.0, data_bytes / _COMPLETION_ESTIMATE_RATE))

    def _assemble_kv_cache(
        self, results: list,
    ) -> tuple[dict[int, dict[int, bytes]], dict[int, list[KVTensorMeta]]]:
        """Assemble each worker's collection results into (kv_cache, metas).

        Opens the shm and takes a memoryview (zero-copy), deferring release until
        after send_state completes via release_kv_cache_shm; if any worker returns
        an error, raise, and the caller (activate_endpoint.collect) catches it and
        returns success=False.
        """
        kv_cache: dict[int, dict[int, bytes]] = {}
        metas: dict[int, list[KVTensorMeta]] = {}
        for rank, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            if result.get("error"):
                raise RuntimeError(
                    f"worker {rank} KV collection failed: {result['error']}"
                )
            if result.get("empty"):
                kv_cache[rank] = {}
                metas[rank] = [
                    KVTensorMeta(**m) for m in result.get("metas", [])
                ]
                continue
            if result.get("shm_name"):
                from multiprocessing import shared_memory

                shm = shared_memory.SharedMemory(name=result["shm_name"])
                self._pending_shm.append(shm)
                size = int(result["shm_size"])
                # memoryview references the shared memory, avoiding a full copy of
                # large data via bytes(); shm is unlinked only after send_state
                # completes (see release_kv_cache_shm).
                kv_cache[rank] = {0: memoryview(shm.buf)[:size]}
                metas[rank] = [
                    KVTensorMeta(**m) for m in result.get("metas", [])
                ]
        return kv_cache, metas

    async def _collect_worker_kv_cache_chunked(
        self, block_ids: list[int] | None, chunk_size: int,
        boost_cpu: bool = False,
    ) -> dict[int, dict[int, bytes]]:
        """Split block_ids into chunks, collect each chunk with lightweight RPCs interleaved between inference steps, and aggregate the results by rank/layer into a full KV buffer semantically equivalent to ``_collect_worker_kv_cache(block_ids)``. Also updates ``self._kv_tensor_metas``."""
        if block_ids is None:
            kv_cache, metas = await self._collect_worker_kv_cache(None, boost_cpu)
            self._kv_tensor_metas = metas
            return kv_cache

        block_ids = list(block_ids)
        if not block_ids:
            kv_cache, metas = await self._collect_worker_kv_cache([], boost_cpu)
            self._kv_tensor_metas = metas
            return kv_cache

        if chunk_size is None or chunk_size <= 0:
            chunk_size = _KV_COLLECT_CHUNK_BLOCKS
        chunks = [
            block_ids[i:i + chunk_size]
            for i in range(0, len(block_ids), chunk_size)
        ]
        if len(chunks) == 1:
            kv_cache, metas = await self._collect_worker_kv_cache(
                chunks[0], boost_cpu
            )
            self._kv_tensor_metas = metas
            return kv_cache

        per_chunk: list[tuple[dict, dict]] = []
        for chunk in chunks:
            kv, metas = await self._collect_worker_kv_cache(chunk, boost_cpu)
            per_chunk.append((kv, metas))

        # Aggregation (cross-chunk concatenation) is a ~30GB CPU memory copy; if
        # done synchronously on the event loop it would block the API server's
        # event loop for a long time, interrupting token streaming (rate drops to
        # 0). Run it in the thread pool: numpy's large-block copy releases the GIL,
        # so the background thread does not block the event loop.
        loop = asyncio.get_running_loop()
        kv_cache, metas = await loop.run_in_executor(
            None, self._aggregate_kv_chunks, per_chunk
        )
        # Explicitly release per_chunk (whose memoryviews reference shm.buf) so the
        # exported pointer count drops to zero and release_kv_cache_shm's
        # shm.close() can succeed, without relying on stop-the-world gc.collect()
        # (the root cause of the throughput drop).
        del per_chunk
        self._kv_tensor_metas = metas
        return kv_cache

    @staticmethod
    def _aggregate_kv_chunks(
        per_chunk: list[tuple[dict, dict]],
    ) -> tuple[dict[int, dict[int, bytes]], dict[int, list[KVTensorMeta]]]:
        """Concatenate the KV buffers of multiple chunks across chunks by rank/layer into a full buffer.

        per_chunk: list of (kv_cache, metas), kv_cache={rank:{0:buffer}},
        metas={rank:[KVTensorMeta, ...]}. Returns the fully concatenated buffer per
        rank per layer and the updated metadata (shape[0] summed, byte_offset
        re-laid-out).

        Concatenation is implemented with numpy: ``np.empty`` preallocation (no
        memset, no 30GB zeroing) + ``np.frombuffer`` zero-copy reads of the chunk +
        numpy slice assignment copies segment by segment. numpy's large-block copy
        releases the GIL, so this method is suited to running in a
        ``run_in_executor`` thread pool without blocking the API server event loop,
        eliminating the root cause of the token-stream rate dropping to 0.
        """
        import torch

        ranks: set[int] = set()
        for kv, _ in per_chunk:
            ranks.update(kv.keys())

        final_kv: dict[int, dict[int, bytes]] = {}
        final_metas: dict[int, list[KVTensorMeta]] = {}
        for rank in sorted(ranks):
            bufs: list = []
            metas_list: list[list[KVTensorMeta]] = []
            for kv, metas in per_chunk:
                if rank in kv and kv[rank]:
                    bufs.append(kv[rank][0])
                    metas_list.append(metas[rank])
            if not bufs:
                final_kv[rank] = {}
                final_metas[rank] = []
                continue

            n_layers = len(metas_list[0])
            new_metas: list[KVTensorMeta] = []
            offset = 0
            for li in range(n_layers):
                base = metas_list[0][li]
                nbytes = 0
                total_blocks = 0
                for bi in range(len(bufs)):
                    m = metas_list[bi][li]
                    nbytes += m.nbytes
                    if m.shape:
                        total_blocks += int(m.shape[0])
                new_shape = list(base.shape)
                if new_shape:
                    new_shape[0] = total_blocks
                new_metas.append(KVTensorMeta(
                    layer_idx=getattr(base, "layer_idx", li),
                    shape=new_shape,
                    dtype=base.dtype,
                    byte_offset=offset,
                    nbytes=nbytes,
                    head_dim=base.head_dim,
                ))
                offset += nbytes

            # Aggregate with torch.cat (measured ~23GB/s), replacing numpy slice
            # assignment (~1.85GB/s), eliminating ~90% of the CPU copy time in the
            # aggregation phase. torch's large-block copy releases the GIL, so it
            # can still run in the run_in_executor thread pool without blocking the
            # event loop.
            parts: list = []
            for li in range(n_layers):
                for bi, buf in enumerate(bufs):
                    m = metas_list[bi][li]
                    if m.nbytes <= 0:
                        continue
                    parts.append(torch.frombuffer(
                        buf, dtype=torch.uint8,
                        count=m.nbytes, offset=m.byte_offset,
                    ))
            total_tensor = torch.cat(parts)
            final_kv[rank] = {0: total_tensor.numpy().data}
            final_metas[rank] = new_metas
            # Explicitly release the torch tensors and memoryviews referencing
            # shm.buf for this rank, so the exported pointers are released
            # immediately, avoiding reliance on gc.collect().
            del parts
            del bufs
        return final_kv, final_metas
