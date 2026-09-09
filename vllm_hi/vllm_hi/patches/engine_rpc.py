# SPDX-License-Identifier: Apache-2.0
"""RPC method implementations for EngineCore / Worker (migration state
collection and recovery).

These module-level functions are injected as ``EngineCore._vllm_hi_*`` /
``Worker._vllm_hi_*`` (see :func:`vllm_hi.patches.patch_engine_internals`),
and invoked by name via ``call_utility_async`` (EngineCore) /
``collective_rpc_async`` (Worker) on the data plane. The ``_vllm_hi_*``
attribute names are the RPC contract and must not be changed.

Reasons for keeping these in a separate module:
1. The patches package only handles "injection"; the state-machine logic
   lives here.
2. Module-level functions can be serialized by reference with cloudpickle
   (collective_rpc passes them to the worker process for execution), which is
   better than closures (a closure captures the entire patching function scope,
   incurring larger serialization overhead and breaking more easily as the
   scope changes).
"""

from __future__ import annotations

import base64
import pickle
import time
from typing import Any

import torch

from vllm.logger import init_logger

from vllm_hi.common.parallel import layer_start_index, worker_global_rank
from vllm_hi.common.utils import deserialize_sampling_params

logger = init_logger("vllm.vllm_hi.patches.engine_rpc")


def serialize_sampling_params(sampling_params: Any) -> str | None:
    """Serialize SamplingParams to a base64 string (for cross-process reconstruction)."""
    if sampling_params is None:
        return None
    try:
        return base64.b64encode(pickle.dumps(sampling_params)).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[vllm_hi] failed to serialize SamplingParams: %s", exc)
        return None


def resolve_restore_tokens(
    num_prompt: int,
    output_token_ids: list[int],
    num_computed_tokens: int,
) -> tuple[int, int]:
    """Compute (alloc_tokens, num_computed_tokens) for a restored request.

    The restored request's KV blocks are fully copied from A to B, but the
    scheduler needs at least 1 pending token to sample the next output. In the
    A-side snapshot, num_computed_tokens for a decode request may be 0
    (unreliable); restoring it as 0 would take the prefix-cache branch and trip
    the num_computed_tokens <= num_prompt_tokens assertion. Therefore:
    - has output (decoding): allocate all num_tokens blocks, computed=num_tokens-1
    - no output and prefill complete: allocate prompt blocks, computed=prompt-1
    - prefill incomplete: keep the A-side progress
    """
    num_output = len(output_token_ids)
    num_tokens = num_prompt + num_output
    if num_output > 0:
        return num_tokens, max(num_tokens - 1, 0)
    if num_prompt > 0 and num_computed_tokens >= num_prompt:
        return num_prompt, num_prompt - 1
    return num_computed_tokens, num_computed_tokens


def restore_request(scheduler: Any, rs: dict, block_hasher: Any = None):
    """Rebuild a Request in the scheduler and allocate its KV cache blocks.

    Returns (request, newly allocated b_block_ids).
    """
    from vllm.v1.request import Request, RequestStatus

    req_id = rs["seq_id"]
    sampling_params = deserialize_sampling_params(rs.get("sampling_params"))
    prompt_token_ids = list(rs.get("prompt_token_ids") or [])
    output_token_ids = list(rs.get("output_token_ids") or [])
    num_computed_tokens = int(rs.get("num_computed_tokens") or 0)
    num_prompt = len(prompt_token_ids)

    alloc_tokens, num_computed_tokens = resolve_restore_tokens(
        num_prompt, output_token_ids, num_computed_tokens
    )

    req = Request(
        request_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
    )
    req.append_output_token_ids(output_token_ids)
    req.num_computed_tokens = num_computed_tokens

    coordinator = scheduler.kv_cache_manager.coordinator
    # Allocate KV blocks for all tokens (including the trailing unaligned block)
    # so the count matches the A-side block_ids, keeping the block mapping
    # aligned block-by-block.
    coordinator.allocate_new_blocks(
        req_id,
        num_tokens=alloc_tokens,
        num_tokens_main_model=alloc_tokens,
    )
    groups = coordinator.get_blocks(req_id)
    b_blocks = groups[0] if groups else []
    b_ids = [b.block_id for b in b_blocks]

    scheduler.requests[req_id] = req
    return req, b_ids


def engine_core_get_scheduler_state(self: Any) -> dict:
    """Return a scheduler state snapshot (request_states, block_table, queues)."""
    scheduler = self.scheduler
    request_states: dict = {}
    block_table: dict = {}
    running_seq_ids: list[str] = []
    waiting_seq_ids: list[str] = []

    for req_id, req in getattr(scheduler, "requests", {}).items():
        request_states[req_id] = {
            "seq_id": req.request_id,
            "prompt_token_ids": list(req.prompt_token_ids or []),
            "output_token_ids": list(req.output_token_ids),
            "num_computed_tokens": req.num_computed_tokens,
            "sampling_params": serialize_sampling_params(req.sampling_params),
            "status": str(req.status),
        }
        try:
            bt = scheduler.kv_cache_manager.get_block_ids(req_id)
            ids = list(bt[0]) if bt else []
        except Exception:  # noqa: BLE001
            ids = []
        block_table[req_id] = {
            "block_ids": ids,
            "ref_count": len(ids) or 1,
            "block_hash": None,
        }

    for req in getattr(scheduler, "running", []):
        running_seq_ids.append(req.request_id)
    try:
        for req in scheduler.waiting:
            waiting_seq_ids.append(req.request_id)
    except Exception:  # noqa: BLE001
        pass

    return {
        "request_states": request_states,
        "block_table": block_table,
        "running_seq_ids": running_seq_ids,
        "waiting_seq_ids": waiting_seq_ids,
    }


def engine_core_set_scheduler_state(self: Any, state: dict) -> dict:
    """Restore scheduler state: rebuild requests, allocate KV blocks, inject queues.

    Returns the block mapping ``[[a_block, b_block], ...]`` for the receiver to
    do block-level KV copy.
    """
    scheduler = self.scheduler
    request_states = state.get("request_states", {})
    block_table = state.get("block_table", {})
    running_ids = set(state.get("running_seq_ids", []))
    from vllm.v1.request import RequestStatus

    block_mapping: list[list[int]] = []
    _t0 = time.perf_counter()
    try:
        block_hasher = getattr(self, "request_block_hasher", None)
        for req_id, rs in request_states.items():
            try:
                req, b_ids = restore_request(
                    scheduler, rs, block_hasher=block_hasher
                )
                a_ids = list(
                    (block_table.get(req_id) or {}).get("block_ids", [])
                )
                for a, b in zip(a_ids, b_ids):
                    block_mapping.append([int(a), int(b)])
                # All restored requests go into waiting (status=WAITING); the
                # first schedule takes the scheduled_new_reqs path, so the
                # worker creates req_states accordingly. Putting them directly
                # into running would make the first schedule treat them as
                # cached_reqs, but the worker has no corresponding req_states,
                # causing a KeyError.
                req.status = RequestStatus.WAITING
                scheduler.waiting.add_request(req)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[vllm_hi] failed to restore request %s: %s", req_id, exc
                )
        logger.info(
            "[vllm_hi] _vllm_hi_set_scheduler_state: restored %d requests, "
            "%d block mappings, took %.3fs",
            len(scheduler.requests), len(block_mapping),
            time.perf_counter() - _t0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[vllm_hi] _vllm_hi_set_scheduler_state failed")
    return {"block_mapping": block_mapping}


def engine_core_apply_scheduler_incremental(self: Any, state: dict) -> dict:
    """Apply incremental scheduler state (add/modify/remove requests).

    Returns the current full block mapping for the receiver to redo the
    block-level KV copy.
    """
    scheduler = self.scheduler
    from vllm.v1.request import RequestStatus

    _t0 = time.perf_counter()

    added = state.get("request_states_added", {})
    modified = state.get("request_states_modified", {})
    removed = state.get("request_states_removed", [])
    bt_added = state.get("block_table_added", {})
    bt_modified = state.get("block_table_modified", {})
    running_ids = set(state.get("running_seq_ids", []))

    # 1. Remove finished requests
    for req_id in removed:
        try:
            scheduler.finish_requests(req_id, RequestStatus.FINISHED_STOPPED)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi] failed to incrementally remove request %s: %s", req_id, exc)

    # 2. Add requests
    block_hasher = getattr(self, "request_block_hasher", None)
    for req_id, rs in added.items():
        try:
            req, b_ids = restore_request(
                scheduler, rs, block_hasher=block_hasher
            )
            # Put into waiting uniformly; the first schedule takes the new_reqs
            # path to create worker req_states.
            req.status = RequestStatus.WAITING
            scheduler.waiting.add_request(req)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi] failed to incrementally add request %s: %s", req_id, exc)

    # 3. Modify requests: sync output_token_ids / num_computed_tokens and
    # allocate additional blocks
    for req_id, rs in modified.items():
        req = scheduler.requests.get(req_id)
        if req is None:
            continue
        try:
            output_token_ids = list(rs.get("output_token_ids") or [])
            num_computed_tokens = int(rs.get("num_computed_tokens") or 0)
            num_prompt = len(req.prompt_token_ids or [])
            alloc_tokens, num_computed_tokens = resolve_restore_tokens(
                num_prompt, output_token_ids, num_computed_tokens
            )
            req._output_token_ids.clear()
            req._output_token_ids.extend(output_token_ids)
            req._all_token_ids.clear()
            req._all_token_ids.extend(list(req.prompt_token_ids or []))
            req._all_token_ids.extend(output_token_ids)
            req.num_computed_tokens = num_computed_tokens
            req.update_block_hashes()
            coordinator = scheduler.kv_cache_manager.coordinator
            coordinator.allocate_new_blocks(
                req_id,
                num_tokens=alloc_tokens,
                num_tokens_main_model=alloc_tokens,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("[vllm_hi] failed to incrementally modify request %s: %s", req_id, exc)

    # 4. Aggregate the block mapping (for block-level KV copy).
    # Only iterate over added/modified requests (not all requests), avoiding the
    # O(N) wasted get_block_ids + create_kv_cache_blocks work over ~3000
    # unchanged requests.
    block_mapping: list[list[int]] = []
    for req_id in list(bt_added.keys()) + list(bt_modified.keys()):
        try:
            bt = scheduler.kv_cache_manager.get_block_ids(req_id)
            b_ids = list(bt[0]) if bt else []
        except Exception:  # noqa: BLE001
            b_ids = []
        a_bt = bt_modified.get(req_id) or bt_added.get(req_id) or {}
        a_ids = list(a_bt.get("block_ids") or [])
        for a, b in zip(a_ids, b_ids):
            block_mapping.append([int(a), int(b)])

    logger.info(
        "[vllm_hi] _vllm_hi_apply_scheduler_incremental: "
        "+%d ~%d -%d, %d requests now, %d block mappings, took %.3fs",
        len(added), len(modified), len(removed),
        len(scheduler.requests), len(block_mapping),
        time.perf_counter() - _t0,
    )
    return {"block_mapping": block_mapping}


def worker_export_kv_ipc(self: Any) -> dict:
    """Export this worker's KV Cache CUDA IPC handles (direct GPU transfer, no
    CPU round-trip).

    Each layer's KV tensor exports its IPC handle via
    ``untyped_storage()._share_cuda_()``, returning
    ``{rank, layers: {global_layer_idx: handle_info}}``, aggregated by the
    collector keyed on ``(src_global_rank, global_layer_idx)`` (``rank`` is the
    global worker rank).
    """
    model_runner = getattr(self, "model_runner", None)
    if model_runner is None:
        return {"error": "model_runner unavailable"}
    kv_caches = getattr(model_runner, "kv_caches", [])
    if not kv_caches:
        return {"rank": 0, "layers": {}}
    # Global worker rank (TP fastest dimension).
    rank = worker_global_rank()
    start_layer = layer_start_index(
        getattr(model_runner, "model_config", None),
        getattr(model_runner, "parallel_config", None),
    )
    # Synchronize the compute stream so KV is fully written before export
    # (snapshot consistency).
    torch.cuda.current_stream().synchronize()
    layers: dict = {}
    for li, t in enumerate(kv_caches):
        if not isinstance(t, torch.Tensor):
            continue
        h = t.untyped_storage()._share_cuda_()
        # h = (device, handle, size, offset, ref_counter,
        #      ref_counter_offset, event_handle, event_sync_required)
        layers[start_layer + li] = {
            "handle": h[1],
            "size": h[2],
            "offset": h[3],
            "ref_counter": h[4],
            "ref_counter_offset": h[5],
            "event_handle": h[6],
            "event_sync_required": h[7],
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": str(t.dtype),
        }
    return {"rank": rank, "layers": layers}


def worker_set_kv_ipc(
    self: Any,
    kv_ipc_handles: dict,
    head_shard_map: dict,
    block_mapping: list[list[int]],
) -> None:
    """Open the A-side shared memory and perform the GPU-to-GPU copy (CUDA IPC
    direct restore).

    kv_ipc_handles: {(src_global_rank, global_layer_idx): handle_info}
    head_shard_map: {dst_global_rank: [shard dict]} (head shards when TP/PP change)
    block_mapping: [[a_block, b_block], ...] (A physical block -> B physical block)
    """
    model_runner = getattr(self, "model_runner", None)
    if model_runner is None:
        return
    kv_caches = getattr(model_runner, "kv_caches", [])
    if not kv_caches:
        return
    # Global worker rank (TP fastest dimension), used to index head_shard_map
    # (dst global-rank key).
    rank = worker_global_rank()
    start_layer = layer_start_index(
        getattr(model_runner, "model_config", None),
        getattr(model_runner, "parallel_config", None),
    )
    shards = head_shard_map.get(rank)
    if not shards:
        shards = [{
            "src_rank": rank, "head_offset": 0,
            "head_len": 0, "dst_head_offset": 0,
        }]
    # Sort the block mapping (a->b) by a and group into ranges where both a and
    # b are contiguous. Contiguous ranges can be copied with a single memcpy
    # (dst[b0:b1]=src[a0:a1]) instead of per-block fancy-index gather,
    # approaching P2P peak bandwidth (scattered gather measured only ~1.5GB/s).
    sorted_pairs = sorted((int(a), int(b)) for a, b in block_mapping)
    ranges: list[tuple[int, int, int, int]] = []  # (a_start, a_end, b_start, b_end)
    for a, b in sorted_pairs:
        if ranges and a == ranges[-1][1] and b == ranges[-1][3]:
            ranges[-1] = (ranges[-1][0], a + 1, ranges[-1][2], b + 1)
        else:
            ranges.append((a, a + 1, b, b + 1))

    device = torch.cuda.current_device()
    torch.cuda._lazy_init()
    _t0 = time.perf_counter()
    # Cache opened shared memory across rounds: repeatedly opening/closing the
    # same CUDA allocation triggers cudaErrorInvalidResourceHandle (a
    # cudaIpcMemHandle can only be opened once per device). With caching, each
    # allocation is opened only once; later writes by A to the same physical
    # memory are automatically visible through B's cached view.
    open_storages = getattr(self, "_vllm_hi_ipc_storages", None)
    if open_storages is None:
        open_storages = {}
        self._vllm_hi_ipc_storages = open_storages

    def _open(info: dict):
        handle = info["handle"]
        if handle not in open_storages:
            open_storages[handle] = torch.UntypedStorage._new_shared_cuda(
                device,
                handle,
                info["size"],
                info["offset"],
                info["ref_counter"],
                info["ref_counter_offset"],
                info["event_handle"],
                info["event_sync_required"],
            )
        storage = open_storages[handle]
        dtype = getattr(torch, str(info["dtype"]).split(".")[-1])
        shape = info["shape"]
        stride = info["stride"]
        # Use set_ to reconstruct the shared memory as a dtype'd GPU tensor
        # (zero-copy view).
        return torch.empty(0, dtype=dtype, device=device).set_(
            storage, int(info.get("storage_offset", 0)), shape, stride
        )

    for layer_idx in range(len(kv_caches)):
        dst = kv_caches[layer_idx]
        if not isinstance(dst, torch.Tensor):
            continue
        global_layer = start_layer + layer_idx
        for shard in shards:
            src_rank = int(shard["src_rank"])
            info = kv_ipc_handles.get((src_rank, global_layer))
            if info is None:
                continue
            src = _open(info)
            head_offset = int(shard.get("head_offset", 0))
            head_len = int(shard.get("head_len", 0))
            dst_head_offset = int(shard.get("dst_head_offset", 0))
            if head_len > 0:
                head_end = head_offset + head_len
            else:
                head_end = src.shape[1]
            seg_heads = head_end - head_offset
            src_blocks = src.shape[0]
            dst_blocks = dst.shape[0]
            # Copy contiguous ranges in bulk (a and b both contiguous; one
            # memcpy instead of per-block gather).
            for a_start, a_end, b_start, b_end in ranges:
                if a_start >= src_blocks or b_start >= dst_blocks:
                    continue
                n = min(a_end, src_blocks) - a_start
                n = min(n, min(b_end, dst_blocks) - b_start)
                if n <= 0:
                    continue
                dst[
                    b_start:b_start + n,
                    dst_head_offset:dst_head_offset + seg_heads,
                ] = src[a_start:a_start + n, head_offset:head_end]
    # Shared-memory references are cached in self._vllm_hi_ipc_storages and
    # reused across rounds; A releases them after migration completes (or they
    # are released naturally when the worker exits).
    torch.cuda.synchronize()
    logger.info(
        "[vllm_hi] worker=%d IPC KV copy %d block mappings (%d contiguous ranges), "
        "took %.3fs",
        rank, len(block_mapping), len(ranges), time.perf_counter() - _t0,
    )


def worker_ensure_kv_collector(self: Any) -> bool:
    """Lazily initialize the in-process KvCollector (background thread + dedicated stream).

    Created on first dispatch; on failure (no model_runner / empty kv_caches)
    logs and returns False, so the caller returns an empty result instead of
    raising.
    """
    if getattr(self, "_vllm_hi_kv_collector", None) is not None:
        return True
    model_runner = getattr(self, "model_runner", None)
    if model_runner is None:
        return False
    try:
        from vllm_hi.migration.kv_collector import KvCollector

        self._vllm_hi_kv_collector = KvCollector(model_runner)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[vllm_hi] KvCollector initialization failed: %s", exc)
        self._vllm_hi_kv_collector = None
        return False


def worker_collect_kv_cache(self: Any, block_ids, boost_cpu) -> dict:
    """Lightweight dispatch of a KV collection task (record_event + enqueue),
    returning task_id.

    Records a snapshot event on the compute stream and enqueues the task before
    returning immediately, without any GPU copy or CPU encoding, so it returns
    in microseconds without blocking inference. When there is no KV to collect
    (pure embedding model / empty kv_caches) it returns
    ``{"empty": True, "metas": []}``.
    """
    if not worker_ensure_kv_collector(self):
        return {"empty": True, "metas": []}
    collector = self._vllm_hi_kv_collector
    snapshot_event = torch.cuda.Event()
    torch.cuda.current_stream().record_event(snapshot_event)
    task_id = collector.dispatch(block_ids, boost_cpu, snapshot_event)
    return {"task_id": task_id}


def worker_try_get_result(self: Any, task_id=None):
    """Non-blocking query of the KV collection task result; returns None if not finished."""
    collector = getattr(self, "_vllm_hi_kv_collector", None)
    if collector is None:
        return None
    return collector.try_get_result(task_id)
