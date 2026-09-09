# SPDX-License-Identifier: Apache-2.0
"""In-worker KV Cache restore function on the B side.

``set_kv_cache`` is a module-level function serialized to the worker process via
``collective_rpc_async``'s cloudpickle (the worker is passed as the first argument).
As a standalone module it can be unit-tested in isolation and easily hook
capabilities such as "KV quantized restore".
"""

from __future__ import annotations

from typing import Any

from vllm_hi.common.parallel import layer_start_index, worker_global_rank
from vllm_hi.migration.runtime_config import RUNTIME_CONFIG


def set_kv_cache(
    worker: Any,
    shm_map: dict[int, str],
    shard_map: dict[int, list[dict]],
    tensor_metas: dict[int, list[dict]],
    src_tp: int,
    dst_tp: int,
    block_mapping: list[list[int]],
    kv_cache_blocks: list[int] | None,
) -> None:
    """Restore the KV Cache within the worker per the head-shard mapping.

    All rank keys are **global worker ranks** (``global_rank = pp_rank * tp_size + tp_rank``):
    - shm_map: {src_global_rank: shm name}
    - shard_map: {dst_global_rank: [shard dict]}
    - tensor_metas: {src_global_rank: [per-layer meta dict]}

    This worker computes its dst global rank from ``dst_tp`` and its own pp/tp rank
    and takes the shards accordingly; it then locates the shm and metas by
    shard.src_rank (source global rank) and aligns PP layers via the global
    ``layer_idx`` (see ``global_layer = start_layer + layer_idx`` below).
    """
    import os
    import time

    import torch
    from multiprocessing import shared_memory

    # Global worker rank (TP fastest-varying). The B worker is instantiated per
    # dst_tp, so the current world group's TP size == dst_tp and
    # ``worker_global_rank()`` is equivalent to ``pp_rank * dst_tp + tp_rank``;
    # unify on the common helper to keep the rank basis from drifting again.
    rank = worker_global_rank()
    shards = shard_map.get(rank)
    if not shards:
        # Default one-to-one (topology unchanged or no mapping provided): full head.
        shards = [{
            "src_rank": rank, "head_offset": 0,
            "head_len": 0, "dst_head_offset": 0,
        }]
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return
    kv_caches = getattr(model_runner, "kv_caches", [])
    if not kv_caches:
        return
    restore_t0 = time.perf_counter()

    # Compute the starting global layer index owned by this worker (under PP each
    # stage holds only part of the layers; use the global layer index to align with
    # the source meta).
    start_layer = layer_start_index(
        getattr(model_runner, "model_config", None),
        getattr(model_runner, "parallel_config", None),
    )

    mapping: dict[int, int] = {}
    for a, b in block_mapping:
        mapping[int(a)] = int(b)

    open_shms: dict[int, shared_memory.SharedMemory] = {}
    metas_by_layer: dict[int, dict[int, dict]] = {}
    # vLLM steady-state worker thread count is 1; raising to 8-16 threads to speed
    # up the CPU→GPU copy means 4 workers restoring simultaneously would spawn
    # 32-64 CPU threads, contending with the A-side inference on the same NUMA node
    # for CPU/memory bandwidth and dropping A-side token throughput to zero at the
    # tail of each transfer round. Cap the threads low and yield between layers to
    # reduce restore-phase CPU contention at the root.
    kv_restore_threads = RUNTIME_CONFIG.kv_restore_threads
    if kv_restore_threads < 1:
        kv_restore_threads = 4
    kv_yield_seconds = RUNTIME_CONFIG.kv_restore_yield_seconds
    if kv_yield_seconds < 0:
        kv_yield_seconds = 0.0
    old_threads = torch.get_num_threads()
    torch.set_num_threads(kv_restore_threads)
    try:
        for shard in shards:
            src_rank = int(shard["src_rank"])
            name = shm_map.get(src_rank)
            if name and src_rank not in open_shms:
                open_shms[src_rank] = shared_memory.SharedMemory(
                    name=name
                )
            metas = tensor_metas.get(src_rank) or []
            metas_by_layer[src_rank] = {
                int(m.get("layer_idx", i)): m
                for i, m in enumerate(metas)
            }

        for layer_idx in range(len(kv_caches)):
            dst = kv_caches[layer_idx]
            if not isinstance(dst, torch.Tensor):
                continue
            dst_device = dst.device
            global_layer = start_layer + layer_idx
            for shard in shards:
                src_rank = int(shard["src_rank"])
                shm = open_shms.get(src_rank)
                meta = metas_by_layer.get(src_rank, {}).get(
                    global_layer
                )
                if shm is None or meta is None:
                    continue
                shape = meta.get("shape")
                nbytes = meta.get("nbytes")
                if not shape or not nbytes:
                    continue
                dtype = getattr(
                    torch, str(meta["dtype"]).split(".")[-1]
                )
                numel = 1
                for d in shape:
                    numel *= d
                src = torch.frombuffer(
                    shm.buf, dtype=dtype, count=numel,
                    offset=int(meta["byte_offset"]),
                ).reshape(shape)
                head_offset = int(shard.get("head_offset", 0))
                head_len = int(shard.get("head_len", 0))
                dst_head_offset = int(
                    shard.get("dst_head_offset", 0)
                )
                # Split heads first, then move to GPU
                if head_len > 0:
                    seg = src[:, head_offset:head_offset + head_len]
                else:
                    seg = src[:, head_offset:]
                # First compute the blocks to copy for this layer (js=source block
                # index, bs=target block), then move only the used blocks to the GPU.
                # Whole-layer .to(device) would occupy an extra full layer of VRAM and
                # OOM when B's KV cache is full (high util); moving only the used
                # blocks keeps the temporary GPU usage down to len(js) blocks.
                if kv_cache_blocks is not None:
                    js: list[int] = []
                    bs: list[int] = []
                    for j, a in enumerate(kv_cache_blocks):
                        b = mapping.get(int(a))
                        if (b is None or j >= seg.shape[0]
                                or b >= dst.shape[0]):
                            continue
                        js.append(j)
                        bs.append(b)
                else:
                    js = []
                    bs = []
                    for a, b in mapping.items():
                        if (a >= seg.shape[0] or b >= dst.shape[0]):
                            continue
                        js.append(a)
                        bs.append(b)
                if not js:
                    continue
                # Move only the used blocks to the GPU, chunked by blocks
                # (_KV_RESTORE_CHUNK_BLOCKS blocks per chunk) to control temporary
                # GPU usage: whole-layer .to(device) would occupy an extra full layer
                # of VRAM and OOM when B's KV cache is full (high util). After
                # chunking, each chunk's temporary GPU usage is only chunk blocks,
                # while still keeping the single-kernel benefit of batch fancy-index
                # assignment.
                seg_heads = seg.shape[1]
                _KV_RESTORE_CHUNK_BLOCKS = 512
                for start in range(0, len(js), _KV_RESTORE_CHUNK_BLOCKS):
                    cj = js[start:start + _KV_RESTORE_CHUNK_BLOCKS]
                    cb = bs[start:start + _KV_RESTORE_CHUNK_BLOCKS]
                    seg_chunk = seg[cj].contiguous().to(dst_device)
                    dst[
                        cb,
                        dst_head_offset:dst_head_offset + seg_heads,
                    ] = seg_chunk
                    del seg_chunk
            # Yield CPU/PCIe bandwidth to A-side inference between layers, smoothing restore bursts.
            if layer_idx < len(kv_caches) - 1 and kv_yield_seconds > 0:
                time.sleep(kv_yield_seconds)
    finally:
        torch.set_num_threads(old_threads)
        for shm in open_shms.values():
            shm.close()
        from vllm.logger import init_logger

        logger = init_logger("vllm.vllm_hi.migration.restore_worker")
        logger.info(
            "[vllm_hi Receiver] worker=%d KV restore: %.2fs, "
            "dst_tp=%d, head shards=%s",
            rank, time.perf_counter() - restore_t0, dst_tp,
            [
                (int(s["src_rank"]), int(s["head_offset"]),
                 int(s["head_len"]), int(s["dst_head_offset"]))
                for s in shards
            ],
        )


__all__ = ["set_kv_cache"]
