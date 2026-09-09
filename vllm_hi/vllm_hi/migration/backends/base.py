# SPDX-License-Identifier: Apache-2.0
"""Data-plane transfer backend abstraction (Strategy/Backend pattern).

``KVTransferBackend`` defines the "A sends -> B receives and restores" KV
data-plane contract, making the transfer backend pluggable. KV currently goes
through a CPU roundtrip (GPU->CPU->shm->TCP->CPU->GPU); it can later be replaced
with ICP (CUDA IPC) / RDMA (GPUDirect) direct transfer, and the design also
reserves room for direct-transfer implementations for Ascend/CANN/Lingqu NPUs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TransferContext:
    """Backend runtime context, bundling in-process references shared by both A/B sides.

    Attributes:
        state: FastAPI ``app.state`` (instance state, holding collector/receiver references).
        engine_client: vLLM AsyncLLM (used for A-side collection / B-side restore).
        collector: :class:`vllm_hi.migration.collector.StateCollector` (A-side).
        receiver: :class:`vllm_hi.migration.receiver.StateReceiver` (B-side).
    """

    def __init__(
        self,
        state: Any = None,
        engine_client: Any = None,
        collector: Any = None,
        receiver: Any = None,
    ) -> None:
        self.state = state
        self.engine_client = engine_client
        self.collector = collector
        self.receiver = receiver


class KVTransferBackend(ABC):
    """Abstract base class for KV data-plane transfer backends.

    Owns the full "KV bulk data transfer" data path: A-side :meth:`send_state`
    sends, B-side :meth:`start_data_server` starts a persistent receive service
    and calls back ``on_data`` to restore. Scheduler state (block table / request
    state) is small data and still goes over the control-plane HTTP, not through
    this backend.
    """

    #: Backend identifier (registry key; also used for ``VLLM_HI_TRANSFER_BACKEND`` explicit selection).
    name: str = ""
    #: Whether it is GPU-direct (KV does not roundtrip GPU->CPU->CPU->GPU).
    gpu_direct: bool = False
    #: Whether cross-node transfer is supported.
    cross_node: bool = True
    #: Priority in auto-selection (higher wins).
    priority: int = 0
    #: Whether the data path is implemented (skeleton/placeholder backends set False, so they are excluded from auto- and explicit selection).
    implemented: bool = True

    @classmethod
    def available(cls) -> bool:
        """Whether it is usable in the current environment (for auto-selection and fallback).

        Subclasses may override this to probe hardware/library dependencies (e.g.
        ICP needs P2P, RDMA needs a NIC).
        """
        return True

    @abstractmethod
    async def send_state(
        self,
        ctx: TransferContext,
        meta: dict,
        kv_cache: dict[int, dict[int, Any]],
        dst_addr: tuple[str, int],
    ) -> int:
        """A-side: send ``(meta, kv_cache)`` to ``dst_addr``, returning the byte count of the data.

        Args:
            ctx: Runtime context.
            meta: Control metadata ``{"type": "full"|"incremental", "snapshot"/"update": ...}``.
            kv_cache: KV bulk data, ``{rank: {layer: bytes-like or ipc_handle}}``;
                the layout is decided by the concrete backend.
            dst_addr: B-side data-plane address ``(ip, port)``.
        """
        raise NotImplementedError

    @abstractmethod
    async def start_data_server(
        self,
        ctx: TransferContext,
        listen_addr: tuple[str, int],
    ) -> None:
        """B-side: start a persistent data receive service.

        Each time a complete state is received, restore it via ``ctx.receiver`` /
        ``ctx.state`` (scheduler state + KV bulk data); the exact restore approach
        is decided by the backend.

        Args:
            ctx: Runtime context (containing receiver / state).
            listen_addr: Listen address ``(ip, port)``.
        """
        raise NotImplementedError


__all__ = ["KVTransferBackend", "TransferContext"]
