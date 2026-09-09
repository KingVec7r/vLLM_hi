# SPDX-License-Identifier: Apache-2.0
"""Unit tests for common-layer helpers (rank conversion / topology extraction / registry address / step enums)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from vllm_hi.common.parallel import (
    global_rank,
    layer_start_end,
    layer_start_index,
)
from vllm_hi.common.state import VllmHiState
from vllm_hi.common.types import MigrationStep
from vllm_hi.common.utils import (
    resolve_register_ip,
    topology_dict_from_vllm_config,
    topology_from_args,
    visible_gpu_ids_from_env,
)


class TestGlobalRank(unittest.TestCase):
    def test_tp_fastest_dim(self):
        # global_rank = pp_rank * tp_size + tp_rank
        self.assertEqual(global_rank(0, 0, 4), 0)
        self.assertEqual(global_rank(0, 3, 4), 3)
        self.assertEqual(global_rank(1, 2, 4), 6)
        self.assertEqual(global_rank(2, 0, 2), 4)

    def test_layer_start_index_without_config(self):
        # Without vllm_config, fall back to (0, 0) / 0 (single-PP case).
        self.assertEqual(layer_start_end(None, None), (0, 0))
        self.assertEqual(layer_start_index(None, None), 0)


class TestResolveRegisterIp(unittest.TestCase):
    def test_wildcard_maps_to_loopback(self):
        self.assertEqual(resolve_register_ip("0.0.0.0"), "127.0.0.1")
        self.assertEqual(resolve_register_ip("::"), "127.0.0.1")

    def test_concrete_ip_unchanged(self):
        self.assertEqual(resolve_register_ip("10.0.0.1"), "10.0.0.1")


class TestVisibleGpuIds(unittest.TestCase):
    def test_empty_env(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(visible_gpu_ids_from_env(), [])

    def test_parses_and_skips_bad_entries(self):
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0, 3, ,x,5"}):
            self.assertEqual(visible_gpu_ids_from_env(), [0, 3, 5])


class TestTopologyHelpers(unittest.TestCase):
    def test_topology_from_args(self):
        args = SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=2,
            gpu_memory_utilization=0.8,
        )
        topo = topology_from_args(args, visible_gpu_ids=[0, 1, 2, 3])
        self.assertEqual(topo.tensor_parallel_size, 4)
        self.assertEqual(topo.pipeline_parallel_size, 2)
        self.assertEqual(topo.gpu_memory_utilization, 0.8)
        self.assertEqual(topo.visible_gpu_ids, [0, 1, 2, 3])

    def test_topology_dict_from_vllm_config(self):
        parallel = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1)
        cache = SimpleNamespace(gpu_memory_utilization=0.9)
        vllm_config = SimpleNamespace(parallel_config=parallel, cache_config=cache)
        self.assertEqual(
            topology_dict_from_vllm_config(vllm_config),
            {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            },
        )


class TestMigrationStep(unittest.TestCase):
    def test_wire_value_stable(self):
        # The step value is the string on the wire; changing it would break client progress parsing, so lock it.
        self.assertEqual(MigrationStep.SWITCH_SERVICE_DONE.value, "switch_service_done")
        self.assertEqual(MigrationStep.DONE.value, "done")
        self.assertEqual(MigrationStep.B_INSTANTIATE.value, "b_instantiate")


class TestVllmHiState(unittest.TestCase):
    def test_defaults(self):
        s = VllmHiState()
        self.assertIsNone(s.engine_client)
        self.assertFalse(s.registered)
        self.assertFalse(s.paused)
        self.assertEqual(s.resumed_queues, {})


if __name__ == "__main__":
    unittest.main()
