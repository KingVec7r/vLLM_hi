# SPDX-License-Identifier: Apache-2.0
"""Centralized migration runtime configuration.

Consolidates the magic numbers and performance-probe switches scattered across
``collector.py`` / ``migration.py`` / ``kv_collector.py`` / ``receiver.py`` into
one place, read uniformly from environment variables, for easier understanding,
maintenance, and tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() or default


@dataclass(frozen=True)
class MigrationRuntimeConfig:
    """Core live-migration runtime configuration.

    Attributes:
        max_incr_rounds: Max number of incremental rounds before the final round (including tn-1 base rounds).
        incr_stall_rounds: Number of consecutive rounds whose incremental data volume is not below the historical minimum before it is considered "stalled" and the final round starts early (default 2).
        kv_collect_chunk_blocks: Chunked collection granularity (physical blocks processed per collection).
        kv_collect_chunk_blocks_large: Large-chunk collection granularity for the final increment (A paused).
        transfer_backend: Explicitly specify the data-plane backend ("tcp_shm"/"cuda_ipc"/"rdma"), None for auto-select.
        request_filter: Fully qualified name of a custom request filter (module.Class); None for the default migrate-all.
        shm_chunk_bytes: Chunk size (bytes) for B-side shm writes.
        shm_yield_seconds: Duration (seconds) to yield CPU between B-side shm write chunks.
        kv_restore_threads: torch thread count when the worker restores KV.
        kv_restore_yield_seconds: Duration (seconds) to yield between layers during worker KV restore.
    """

    max_incr_rounds: int = 8
    incr_stall_rounds: int = 2
    kv_collect_chunk_blocks: int = 512
    kv_collect_chunk_blocks_large: int = 1 << 20
    transfer_backend: str | None = None
    request_filter: str | None = None
    shm_chunk_bytes: int = 256 * 1024 * 1024
    shm_yield_seconds: float = 0.02
    kv_restore_threads: int = 4
    kv_restore_yield_seconds: float = 0.01

    @classmethod
    def from_env(cls) -> "MigrationRuntimeConfig":
        return cls(
            max_incr_rounds=_env_int("VLLM_HI_MAX_INCR_ROUNDS", 8),
            incr_stall_rounds=_env_int("VLLM_HI_INCR_STALL_ROUNDS", 2),
            kv_collect_chunk_blocks=_env_int(
                "VLLM_HI_KV_COLLECT_CHUNK_BLOCKS", 512
            ),
            kv_collect_chunk_blocks_large=_env_int(
                "VLLM_HI_KV_COLLECT_CHUNK_BLOCKS_LARGE", 1 << 20
            ),
            transfer_backend=_env_str("VLLM_HI_TRANSFER_BACKEND"),
            request_filter=_env_str("VLLM_HI_REQUEST_FILTER"),
            shm_chunk_bytes=_env_int(
                "VLLM_HI_SHM_CHUNK_BYTES", 256 * 1024 * 1024
            ),
            shm_yield_seconds=_env_float("VLLM_HI_SHM_YIELD_SECONDS", 0.02),
            kv_restore_threads=_env_int("VLLM_HI_KV_RESTORE_THREADS", 4),
            kv_restore_yield_seconds=_env_float(
                "VLLM_HI_KV_RESTORE_YIELD_SECONDS", 0.01
            ),
        )


@dataclass(frozen=True)
class ProbeConfig:
    """Performance probe switches (root-cause validation apparatus, off by default / default-fixed behavior).

    Corresponds to the ``VLLM_HI_PERF_PROBE_*`` environment variables in features.md §4.6.
    All but ``d2h_mode`` have boolean semantics; ``pinned``/``full_throttle`` are
    "on by default" (setting off/0/false falls back to legacy behavior).
    """

    skip_d2h: bool = False
    skip_encode: bool = False
    skip_transfer: bool = False
    skip_poll: bool = False
    pinned: bool = True
    full_throttle: bool = True
    d2h_mode: str = "smooth"

    @classmethod
    def from_env(cls) -> "ProbeConfig":
        return cls(
            skip_d2h=_env_bool("VLLM_HI_PERF_PROBE_SKIP_D2H", False),
            skip_encode=_env_bool("VLLM_HI_PERF_PROBE_SKIP_ENCODE", False),
            skip_transfer=_env_bool("VLLM_HI_PERF_PROBE_SKIP_TRANSFER", False),
            skip_poll=_env_bool("VLLM_HI_PERF_PROBE_SKIP_POLL", False),
            pinned=_env_bool_on("VLLM_HI_PERF_PROBE_PINNED"),
            full_throttle=_env_bool_on("VLLM_HI_PERF_PROBE_FULL_THROTTLE"),
            d2h_mode=_env_str("VLLM_HI_PERF_PROBE_D2H_MODE", "smooth") or "smooth",
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes")


def _env_bool_on(name: str) -> bool:
    """"On by default" switch: returns True when unset, False when set to off/0/false/no."""
    raw = os.environ.get(name)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


# Module-level singletons, parsed once at process startup (migration config is invariant over the process lifetime).
RUNTIME_CONFIG = MigrationRuntimeConfig.from_env()
PROBE_CONFIG = ProbeConfig.from_env()

__all__ = [
    "MigrationRuntimeConfig",
    "ProbeConfig",
    "RUNTIME_CONFIG",
    "PROBE_CONFIG",
]
