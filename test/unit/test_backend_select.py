# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the data-plane backend registry and automatic selection.

Covers:
1. backend registration and lookup by name;
2. fallback to tcp_shm for unknown names;
3. select_backend: explicit selection takes precedence, plus default selection for
   same-node / cross-node / loopback.
"""

from __future__ import annotations

import unittest
from unittest import mock

from vllm_hi.migration.backends import (
    get_backend,
    registered_names,
    select_backend,
)
from vllm_hi.migration.backends.cuda_ipc import CudaIpcTransferBackend


class TestRegistry(unittest.TestCase):
    def test_registered_names(self):
        names = registered_names()
        self.assertIn("tcp_shm", names)

    def test_get_backend_by_name(self):
        b = get_backend("tcp_shm")
        self.assertEqual(b.name, "tcp_shm")
        self.assertFalse(b.gpu_direct)
        self.assertTrue(b.cross_node)

    def test_get_backend_unknown_falls_back(self):
        b = get_backend("no_such_backend")
        self.assertEqual(b.name, "tcp_shm")

    def test_get_backend_none_falls_back(self):
        self.assertEqual(get_backend(None).name, "tcp_shm")


class TestSelectBackend(unittest.TestCase):
    def test_explicit_override(self):
        b = select_backend("127.0.0.1", "10.0.0.2", explicit="tcp_shm")
        self.assertEqual(b.name, "tcp_shm")

    def test_loopback_same_node(self):
        # loopback is treated as same-node; choose cuda_ipc if available, else tcp_shm.
        b = select_backend("127.0.0.1", "127.0.0.1")
        self.assertIn(b.name, ("cuda_ipc", "tcp_shm"))

    def test_same_ip_same_node(self):
        b = select_backend("10.0.0.1", "10.0.0.1")
        self.assertIn(b.name, ("cuda_ipc", "tcp_shm"))

    def test_cross_node_falls_back_tcp(self):
        # Without RDMA hardware, cross-node should fall back to tcp_shm.
        b = select_backend("10.0.0.1", "10.0.0.2")
        self.assertEqual(b.name, "tcp_shm")

    def test_explicit_rdma_skeleton_falls_back(self):
        # The rdma backend is a skeleton (implemented=False); even when explicitly
        # requested it should be downgraded to tcp_shm to avoid the migration failing
        # halfway through the data plane because of a placeholder error.
        b = select_backend("10.0.0.1", "10.0.0.2", explicit="rdma")
        self.assertEqual(b.name, "tcp_shm")

    def test_explicit_unknown_falls_back(self):
        b = select_backend("10.0.0.1", "10.0.0.2", explicit="no_such")
        self.assertEqual(b.name, "tcp_shm")

    def test_explicit_cuda_ipc_honored_even_if_probe_false(self):
        # Regression test: when cuda_ipc is explicitly requested, the available()
        # probe must not run (the all-to-all P2P probe, when the Host process can see
        # all 8 cards, misjudges cross-architecture no-P2P as unavailable, silently
        # downgrading an explicit GPU-direct transfer to a CPU relay and causing severe
        # performance regression).
        with mock.patch.object(CudaIpcTransferBackend, "available", return_value=False):
            b = select_backend("127.0.0.1", "127.0.0.1", explicit="cuda_ipc")
        self.assertEqual(b.name, "cuda_ipc")
        self.assertTrue(b.gpu_direct)


if __name__ == "__main__":
    unittest.main()
