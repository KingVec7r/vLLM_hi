# SPDX-License-Identifier: Apache-2.0
"""Shared conversion helpers for parallel topology: global worker rank and PP layer ranges.

These conversions are repeated in several places (collector / restore_worker /
engine_rpc / kv_collector / remapper) and are a frequent source of bugs.
Centralizing them here means the entire pipeline uses a single convention:

``global_rank = pp_rank * tp_size + tp_rank`` (TP is the fastest-varying
dimension, consistent with ``pp_rank = rank // tp_size`` in vLLM's
``ModelConfig.get_layers_start_end_indices``).

Note: functions depending on ``vllm.distributed`` are imported lazily, to avoid
accessing singletons such as ``get_pp_group`` at import time before the worker
subprocess has finished distributed initialization.
"""

from __future__ import annotations

from typing import Any


def global_rank(pp_rank: int, tp_rank: int, tp_size: int) -> int:
    """Compute the global worker rank of ``(pp_rank, tp_rank)`` within the vLLM world group (TP is the fastest-varying dimension)."""
    return pp_rank * tp_size + tp_rank


def worker_global_rank() -> int:
    """Return the current worker's global worker rank within the vLLM world group.

    Used during KV collection/restore to index ``shm_map`` / ``head_shard_map`` /
    ``kv_ipc_handles`` and similar dicts by global rank.
    """
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return global_rank(
        get_pp_group().rank_in_group,
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )


def layer_start_end(model_config: Any, parallel_config: Any) -> tuple[int, int]:
    """Return the global layer range ``[start, end)`` owned by the current worker.

    Under PP each stage only holds a subset of layers, so the worker's local layer
    indices must be converted to global layer numbers to align with the source
    meta's ``layer_idx``. Returns ``(0, 0)`` when ``model_config``/
    ``parallel_config`` are unavailable (single-PP case, where start_layer=0).
    """
    if model_config is None or parallel_config is None:
        return 0, 0
    return model_config.get_layers_start_end_indices(parallel_config)


def layer_start_index(model_config: Any, parallel_config: Any) -> int:
    """Return the global starting layer index owned by the current worker (the ``start`` of ``layer_start_end``)."""
    return layer_start_end(model_config, parallel_config)[0]


__all__ = [
    "global_rank",
    "worker_global_rank",
    "layer_start_end",
    "layer_start_index",
]
