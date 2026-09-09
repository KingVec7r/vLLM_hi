# SPDX-License-Identifier: Apache-2.0
"""Common utility functions for vllm_hi."""

from __future__ import annotations

import base64
import copy
import os
import pickle
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from vllm.logger import init_logger

from vllm_hi.common.types import Topology

# Note: the vllm. prefix is required, otherwise logs are dropped by vllm's propagate=False config.
logger = init_logger("vllm.vllm_hi.common.utils")


def generate_id(prefix: str = "hi") -> str:
    """Generate a unique ID with a prefix.

    Format: {prefix}-{first 8 hex chars of a uuid4}
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def parse_addr(s: str) -> tuple[str, int]:
    """Parse an ip:port string and return (ip, port).

    Raises:
        ValueError: Malformed format or port outside the 1-65535 range.
    """
    if ":" not in s:
        raise ValueError(f"Malformed address, missing port: {s!r}")
    ip, port_str = s.rsplit(":", 1)
    port = int(port_str)
    if not (1 <= port <= 65535):
        raise ValueError(f"Port out of range 1-65535: {port}")
    return ip, port


def now_iso8601() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# SamplingParams deserialization cache (base64 string -> object).
# During live migration most requests share the same SamplingParams (e.g. identical
# temperature/top_p/max_tokens); deserializing the same object per request via
# pickle.loads becomes the main CPU cost of scheduler-state rebuild.
# Bounded cache with FIFO eviction, to avoid unbounded memory growth from
# accumulating distinct SamplingParams across multiple migrations.
_SAMPLING_PARAMS_CACHE_MAXSIZE = 256
_SAMPLING_PARAMS_CACHE: OrderedDict[str, Any] = OrderedDict()


def deserialize_sampling_params(data: Any) -> Any:
    """Deserialize a base64-encoded SamplingParams (with a bounded cache).

    On a cache hit a shallow copy is returned so multiple restore requests do not
    share the same mutable SamplingParams instance (msgspec Struct is assignable),
    preventing in-place per-request modifications during scheduling/sampling from
    polluting each other. Returns None when ``data`` is empty; non-string
    (already-deserialized) values are returned as-is.
    """
    if not data:
        return None
    if not isinstance(data, str):
        return data
    cached = _SAMPLING_PARAMS_CACHE.get(data)
    if cached is None:
        try:
            cached = pickle.loads(base64.b64decode(data.encode("ascii")))
        except Exception:  # noqa: BLE001
            return None
        _SAMPLING_PARAMS_CACHE[data] = cached
        if len(_SAMPLING_PARAMS_CACHE) > _SAMPLING_PARAMS_CACHE_MAXSIZE:
            _SAMPLING_PARAMS_CACHE.popitem(last=False)
    return copy.copy(cached)


def get_device_cpu_affinity() -> list[int] | None:
    """Get the CPU affinity list of the NUMA node containing the first GPU visible to this process.

    Used to bind data-plane processing (writing shm) or the process to the NUMA
    node of its GPU, so that memcpy without affinity constraints does not drift to
    the node of another instance's GPU and contend for CPU/memory bandwidth with
    inference.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            try:
                first_gpu = int(visible.split(",")[0].strip())
            except ValueError:
                first_gpu = 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(first_gpu)
            cpu_count = os.cpu_count()
            if cpu_count is None:
                return None
            cpu_set_size = (cpu_count + 63) // 64
            mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)
        finally:
            pynvml.nvmlShutdown()
        cpu_ids: list[int] = []
        for i, m in enumerate(mask):
            for bit in range(64):
                cpu_id = i * 64 + bit
                if cpu_id >= cpu_count:
                    break
                if m & (1 << bit):
                    cpu_ids.append(cpu_id)
        return cpu_ids or None
    except Exception:  # noqa: BLE001
        return None


def bind_process_to_device_numa() -> None:
    """Bind the current process to the NUMA node of its visible GPU.

    During same-host live migration the A/B instances share CPU and memory
    bandwidth. vLLM only sets CPU affinity for the Worker processes; the
    APIServer and EngineCore can still drift across the whole machine (0-127).
    When B performs gigabyte-scale data-plane operations (writing shm / loading
    the model), A's unaffined EngineCore/APIServer may drift to B's nodes and be
    contended with, slowing inference. Pinning A's processes to the NUMA node of
    their GPU spatially isolates them from B and eliminates cross-instance
    CPU/memory contention.
    """
    try:
        cpus = get_device_cpu_affinity()
        if cpus:
            os.sched_setaffinity(0, cpus)
            logger.info(
                "[vllm_hi] Bound process CPU affinity to %d CPUs (device NUMA)",
                len(cpus),
            )
    except Exception:  # noqa: BLE001
        logger.warning("[vllm_hi] Failed to bind process CPU affinity (ignored)")


def visible_gpu_ids_from_env() -> list[int]:
    """Parse ``CUDA_VISIBLE_DEVICES`` into a list of visible GPU physical IDs (returns an empty list when unset)."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return []
    ids: list[int] = []
    for x in cuda_visible.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            ids.append(int(x))
        except ValueError:
            continue
    return ids


def resolve_register_ip(host: str) -> str:
    """Convert a listen address to the reachable IP to register with the Host (wildcard addresses fall back to loopback)."""
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def topology_dict_from_vllm_config(vllm_config: Any) -> dict[str, Any]:
    """Extract the topology dict from vLLM ``vllm_config`` (for remapper remapping).

    Unifies the extraction of the (tensor_parallel_size, pipeline_parallel_size,
    gpu_memory_utilization) triple, eliminating the duplication across
    activate_endpoint / standby / registration.
    """
    return {
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": vllm_config.parallel_config.pipeline_parallel_size,
        "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
    }


def topology_from_args(
    args: Any, visible_gpu_ids: list[int] | None = None
) -> Topology:
    """Build a Topology from CLI ``args`` (type B has no vllm_config yet at standby registration)."""
    return Topology(
        tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
        pipeline_parallel_size=getattr(args, "pipeline_parallel_size", 1),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
        visible_gpu_ids=list(visible_gpu_ids or []),
    )
