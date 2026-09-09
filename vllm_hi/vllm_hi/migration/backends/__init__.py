# SPDX-License-Identifier: Apache-2.0
"""Data-plane transfer backends.

Provides :func:`get_backend` (instantiate by name) and :func:`select_backend`
(auto-select the best backend), plus the ``KVTransferBackend`` abstraction. Each
backend implementation self-registers via the ``@register_backend`` decorator;
importing a backend module triggers registration (no circular imports).
"""

from __future__ import annotations

from vllm_hi.migration.backends.base import KVTransferBackend, TransferContext
from vllm_hi.migration.backends.registry import (
    get_backend,
    register_backend,
    registered_names,
    select_backend,
)

# Import backend modules to trigger ``@register_backend`` decorator registration.
from vllm_hi.migration.backends import tcp_shm  # noqa: F401
from vllm_hi.migration.backends import cuda_ipc  # noqa: F401
from vllm_hi.migration.backends import rdma  # noqa: F401

__all__ = [
    "KVTransferBackend",
    "TransferContext",
    "register_backend",
    "get_backend",
    "select_backend",
    "registered_names",
]
