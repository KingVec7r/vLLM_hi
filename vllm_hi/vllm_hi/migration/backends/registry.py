# SPDX-License-Identifier: Apache-2.0
"""Data-plane transfer backend registry and auto-selection.

A standalone module (decoupled from the backend implementations); backend
implementations register via the ``@register_backend`` decorator.
"""

from __future__ import annotations

from typing import Type

from vllm.logger import init_logger

from vllm_hi.migration.backends.base import KVTransferBackend

logger = init_logger("vllm.vllm_hi.migration.backends.registry")

_REGISTRY: dict[str, Type[KVTransferBackend]] = {}


def register_backend(cls: Type[KVTransferBackend]) -> Type[KVTransferBackend]:
    """Register a backend implementation (keyed by ``cls.name``). Usable as a decorator."""
    if not cls.name:
        raise ValueError(f"backend {cls.__name__} is missing name")
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str | None) -> KVTransferBackend:
    """Instantiate a backend by name; fall back to ``tcp_shm`` when ``name`` is empty or unknown."""
    if not name:
        name = "tcp_shm"
    cls = _REGISTRY.get(name)
    if cls is None:
        logger.warning(
            "[vllm_hi Backend] unknown backend %r, falling back to tcp_shm", name
        )
        cls = _REGISTRY["tcp_shm"]
    return cls()


def registered_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def _is_same_node(src_ip: str, dst_ip: str) -> bool:
    """Determine whether A/B are on the same node (loopback or identical IP counts as same-node)."""
    loopback = {"127.0.0.1", "localhost", "::1"}
    if src_ip in loopback or dst_ip in loopback:
        return True
    return src_ip == dst_ip


def _resolve(name: str | None) -> KVTransferBackend | None:
    """Look up a backend by name; returns ``None`` when ``name`` is empty or unknown (for the fallback chain to decide)."""
    if not name:
        return None
    return _REGISTRY.get(name)


def select_backend(
    src_ip: str,
    dst_ip: str,
    explicit: str | None = None,
) -> KVTransferBackend:
    """Auto-select the best data-plane backend.

    Priority:
    1. ``explicit`` (``VLLM_HI_TRANSFER_BACKEND``) explicit selection;
    2. same node and ``cuda_ipc`` available -> CUDA IPC (GPU-direct);
    3. cross-node and ``rdma`` available -> GPUDirect RDMA (GPU-direct);
    4. fallback ``tcp_shm``.

    Explicit selection is a user override: it only checks "name valid + data path
    implemented", and does **not** run the ``available()`` probe. The
    ``cuda_ipc.available()`` probe (full P2P connectivity check) misjudges as
    unavailable in the Host process (which sees all 8 GPUs, including the
    cross-architecture Blackwell<->Ada pair without P2P); downgrading on that
    basis would silently turn an explicit GPU-direct transfer into a CPU
    roundtrip. Availability probes are only used on the auto-selection path.
    """
    def _usable(b: KVTransferBackend) -> bool:
        return b.implemented and b.available()

    if explicit:
        cls = _resolve(explicit)
        if cls is None:
            logger.warning(
                "[vllm_hi Backend] unknown backend %r, degrading to tcp_shm", explicit
            )
            return get_backend("tcp_shm")
        backend = cls()
        # Only degrade when the data path is unimplemented (e.g. rdma skeleton); availability is left to the explicit chooser.
        if not backend.implemented:
            logger.warning(
                "[vllm_hi Backend] backend %r is unimplemented (skeleton), degrading to tcp_shm",
                backend.name,
            )
            return get_backend("tcp_shm")
        logger.info(
            "[vllm_hi Backend] explicit data-plane backend: %s", backend.name
        )
        return backend

    same_node = _is_same_node(src_ip, dst_ip)

    if same_node:
        cuda_cls = _REGISTRY.get("cuda_ipc")
        if cuda_cls is not None:
            backend = cuda_cls()
            if _usable(backend):
                logger.info(
                    "[vllm_hi Backend] same node and CUDA IPC available, selecting cuda_ipc"
                )
                return backend

    if not same_node:
        rdma_cls = _REGISTRY.get("rdma")
        if rdma_cls is not None:
            backend = rdma_cls()
            if _usable(backend):
                logger.info(
                    "[vllm_hi Backend] cross-node and RDMA available, selecting rdma"
                )
                return backend

    logger.info("[vllm_hi Backend] selecting default backend tcp_shm")
    return get_backend("tcp_shm")


__all__ = [
    "register_backend",
    "get_backend",
    "select_backend",
    "registered_names",
]
