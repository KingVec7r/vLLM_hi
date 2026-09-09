# SPDX-License-Identifier: Apache-2.0
"""All per-instance state that vllm_hi stores on ``app.state.vllm_hi``.

Centralizing these fields into a single dataclass means:

- the field set is discoverable in one place, and renames are caught by
  type/attribute checks;
- new fields cannot be silently attached to ``app.state`` and lost;
- ``TransferContext`` and other layers still pass ``app.state`` around, but
  downstream code accesses it uniformly via ``state.vllm_hi.<field>`` for a
  clear boundary.

Fields such as ``engine_client`` cannot be statically typed (they patch vLLM
internals), so they remain ``Any``; at minimum the field names and meanings are
documented uniformly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VllmHiState:
    """All runtime state of a single vllm_hi instance (A/B)."""

    #: Back-reference to the app (set by the build_app patch; lets endpoints look up the app via state).
    app: Any = None
    #: Instance type "A"/"B" (corresponds to CLI --vllm-hi-type).
    instance_type: str | None = None
    #: Host scheduler address "ip:port" (--vllm-hi-host).
    host_addr: str | None = None
    #: Data-plane TCP port (--vllm-hi-data-port).
    data_port: int = 0
    #: vLLM AsyncLLM (used for A-side collection / B-side restore).
    engine_client: Any = None
    #: HostClient used to communicate with the Host (:class:`vllm_hi.instance.host_client.HostClient`).
    host_client: Any = None
    #: Whether the instance has registered with the Host.
    registered: bool = False
    #: Unique instance ID assigned by the Host.
    vllm_hi_id: str | None = None
    #: Whether generation is paused (A-side pause / before B-side activate).
    paused: bool = False
    #: A-side :class:`vllm_hi.migration.collector.StateCollector`.
    collector: Any = None
    #: B-side :class:`vllm_hi.migration.receiver.StateReceiver`.
    receiver: Any = None
    #: ID of the most recent snapshot (baseline for incremental collection).
    last_snapshot_id: str | None = None
    #: Output collector for resume requests (``request_id -> queue``, consumed by the resume endpoint).
    resumed_queues: dict = field(default_factory=dict)
    #: Data-plane backend name (``tcp_shm``/``cuda_ipc``/``rdma``).
    transfer_backend: str | None = None
    #: B-side data-plane receive service task.
    data_server_task: Any = None
    #: Serialization lock for data-plane full/incremental rebuild (asyncio.Lock).
    data_lock: Any = None
    #: Data-plane idle event (the activate endpoint waits on it for rebuild completion).
    data_idle: Any = None


__all__ = ["VllmHiState"]
