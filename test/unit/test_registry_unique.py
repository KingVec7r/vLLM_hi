# SPDX-License-Identifier: Apache-2.0
"""Unit tests for InstanceRegistry unique model names ("A stays A" routing fix)."""

from __future__ import annotations

import asyncio
import unittest

from vllm_hi.common.types import InstanceType, Topology
from vllm_hi.common.protocol import RegisterInstanceRequest
from vllm_hi.coordinator.registry import InstanceRegistry


def _reg_req(model: str, itype: InstanceType = InstanceType.A):
    return RegisterInstanceRequest(
        ip="127.0.0.1",
        port=8000,
        instance_type=itype,
        served_model_name=model,
        inference_params={},
        topology=Topology(tensor_parallel_size=1),
        data_port=0,
    )


class TestUniqueModelName(unittest.TestCase):
    def test_no_conflict_returns_base(self):
        async def go():
            reg = InstanceRegistry()
            return await reg.ensure_unique_model_name("qwen")
        self.assertEqual(asyncio.run(go()), "qwen")

    def test_conflict_adds_suffix(self):
        async def go():
            reg = InstanceRegistry()
            await reg.register(_reg_req("qwen"))
            name = await reg.ensure_unique_model_name("qwen")
            self.assertNotEqual(name, "qwen")
            self.assertTrue(name.startswith("qwen@"))
            return name
        name = asyncio.run(go())
        self.assertNotEqual(name, "qwen")

    def test_unique_across_multiple(self):
        async def go():
            reg = InstanceRegistry()
            await reg.register(_reg_req("qwen"))
            n1 = await reg.ensure_unique_model_name("qwen")
            # Register a conflicting instance manually, then generate again
            await reg.register(_reg_req(n1))
            n2 = await reg.ensure_unique_model_name("qwen")
            return n1, n2
        n1, n2 = asyncio.run(go())
        self.assertNotEqual(n1, n2)
        self.assertNotEqual(n1, "qwen")
        self.assertNotEqual(n2, "qwen")

    def test_count_reflects_registered_instances(self):
        async def go():
            reg = InstanceRegistry()
            self.assertEqual(await reg.count(), 0)
            await reg.register(_reg_req("qwen"))
            await reg.register(_reg_req("standby", InstanceType.B))
            return await reg.count()
        self.assertEqual(asyncio.run(go()), 2)


if __name__ == "__main__":
    unittest.main()
