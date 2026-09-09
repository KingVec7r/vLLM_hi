# SPDX-License-Identifier: Apache-2.0
"""RDMA cross-node GPU-direct data-plane backend (skeleton).

KV data is moved directly between cross-node GPU device memories via GPUDirect
RDMA, without a CPU roundtrip. This is currently a structurally complete
skeleton: ``available()`` probes RDMA devices/libraries, and when unavailable it
is excluded from auto-selection (this machine has no RDMA NIC, so it degrades
automatically and is not exercised). Enabling requires:
- an RDMA NIC on the host (``/sys/class/infiniband`` non-empty);
- the pyverbs or ucx-py library installed, and GPU/NIC supporting GPUDirect RDMA;
- explicitly setting ``VLLM_HI_TRANSFER_BACKEND=rdma`` or being selected
  cross-node by :func:`select_backend`.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

from vllm_hi.migration.backends.base import KVTransferBackend, TransferContext
from vllm_hi.migration.backends.registry import register_backend

logger = init_logger("vllm.vllm_hi.migration.backends.rdma")


def _rdma_hw_available() -> bool:
    """Probe whether RDMA hardware (InfiniBand / RoCE NIC) is present."""
    try:
        if os.path.isdir("/sys/class/infiniband") and os.listdir(
            "/sys/class/infiniband"
        ):
            return True
    except OSError:  # noqa: BLE001
        pass
    return False


def _rdma_lib_available() -> bool:
    """Probe whether an RDMA userspace library (pyverbs or ucx-py) is installed."""
    for mod in ("pyverbs", "ucp"):
        try:
            __import__(mod)
            return True
        except ImportError:  # noqa: BLE001
            continue
    return False


@register_backend
class RdmaTransferBackend(KVTransferBackend):
    """RDMA cross-node GPU-direct data-plane backend (skeleton).

    ``implemented=False``: the data path is not yet implemented
    (send/start_data_server are placeholders). Even if this machine has RDMA
    hardware + libraries, ``select_backend`` degrades to ``tcp_shm``, avoiding an
    error thrown by the placeholder halfway through the migration data path.
    """

    name = "rdma"
    gpu_direct = True
    cross_node = True
    priority = 50  # preferred over tcp_shm when cross-node
    implemented = False

    @classmethod
    def available(cls) -> bool:
        return _rdma_hw_available() and _rdma_lib_available()

    async def send_state(
        self,
        ctx: TransferContext,
        meta: dict,
        kv_cache: dict[int, dict[int, Any]],
        dst_addr: tuple[str, int],
    ) -> int:
        """Skeleton placeholder: reuse the control plane to transfer KV metadata; KV direct transfer is done via GPUDirect RDMA.

        A full implementation must export registered device memory inside the
        worker and perform remote writes, and depends on RDMA hardware and
        libraries.
        """
        raise RuntimeError(
            "RDMA backend skeleton has no data path implemented (no RDMA hardware on this machine); use tcp_shm/cuda_ipc instead"
        )

    async def start_data_server(
        self,
        ctx: TransferContext,
        listen_addr: tuple[str, int],
    ) -> None:
        raise RuntimeError(
            "RDMA backend skeleton has no data path implemented (no RDMA hardware on this machine); use tcp_shm/cuda_ipc instead"
        )


__all__ = ["RdmaTransferBackend"]
