# SPDX-License-Identifier: Apache-2.0
"""Unit tests for state collection (KvCollector).

Covers:
1. slice_blocks contiguous-range slicing (zero GPU reordering) is semantically equivalent to fancy indexing;
2. StateCollector._assemble_kv_cache result assembly (shm / empty / error);
3. KvCollector dispatch and completion-query semantics (unique task_id, None before
   completion, result after completion, background exceptions written to error without
   propagating).

Run: source .../conda.sh && conda activate vllm_hi && \
      python -m unittest test.unit.test_kv_collector -v
"""

from __future__ import annotations

import time
import unittest
from multiprocessing import shared_memory
from types import SimpleNamespace

import torch

from vllm_hi.migration.collector import StateCollector
from vllm_hi.migration.kv_collector import (
    KvCollector,
    PinnedMemoryPool,
    slice_blocks,
)


class TestSliceBlocks(unittest.TestCase):
    def test_single_range(self):
        self.assertEqual(slice_blocks([2, 3, 4]), [(2, 5)])

    def test_multi_range(self):
        self.assertEqual(slice_blocks([1, 2, 5, 6]), [(1, 3), (5, 7)])

    def test_empty(self):
        self.assertEqual(slice_blocks([]), [])

    def test_equals_fancy_index(self):
        t = torch.randn(100, 4, 8)
        ids = sorted({0, 1, 2, 10, 11, 12, 50, 99})
        ranges = slice_blocks(ids)
        cat = torch.cat([t[s:e] for s, e in ranges], dim=0)
        self.assertTrue(torch.equal(cat, t[ids]))


class TestAssembleKvCache(unittest.TestCase):
    def test_assemble_shm(self):
        collector = StateCollector(None)
        t = torch.randn(4, 4, 8, dtype=torch.float32)
        tc = t.contiguous()
        nbytes = tc.numel() * tc.element_size()
        metas = [{
            "layer_idx": 0, "shape": list(tc.shape), "dtype": str(tc.dtype),
            "byte_offset": 0, "nbytes": nbytes, "head_dim": 1,
        }]
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            shm.buf[:nbytes] = tc.view(torch.uint8).reshape(-1).numpy().tobytes()
            results = [{
                "rank": 0, "shm_name": shm.name, "shm_size": nbytes,
                "metas": metas,
            }]
            kv_cache, metas_out = collector._assemble_kv_cache(results)
            self.assertIn(0, kv_cache)
            self.assertEqual(metas_out[0][0].nbytes, nbytes)
            src = torch.frombuffer(
                kv_cache[0][0], dtype=torch.float32, count=tc.numel(), offset=0
            ).reshape(tc.shape)
            self.assertTrue(torch.equal(src, t))
            # Release the memoryview references to shm.buf to avoid an "exported pointer" error on close
            del src
            del kv_cache
        finally:
            shm.close()
            shm.unlink()
            collector.release_kv_cache_shm()

    def test_assemble_error_raises(self):
        collector = StateCollector(None)
        with self.assertRaises(RuntimeError):
            collector._assemble_kv_cache([{"error": "boom"}])

    def test_assemble_empty(self):
        collector = StateCollector(None)
        kv_cache, metas = collector._assemble_kv_cache(
            [{"empty": True, "rank": 0, "metas": []}]
        )
        self.assertEqual(kv_cache, {0: {}})
        self.assertEqual(metas, {0: []})


class TestKvCollector(unittest.TestCase):
    def _make_collector(self, collect_fn=None, maxsize=4):
        model_runner = SimpleNamespace(
            kv_caches=[object()],
            device=torch.device("cuda:0"),
        )
        oc = KvCollector(model_runner, maxsize=maxsize)
        if collect_fn is not None:
            oc._collect = collect_fn
        return oc

    @staticmethod
    def _wait_result(oc, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = oc.try_get_result()
            if res is not None:
                return res
            time.sleep(0.01)
        return None

    def test_dispatch_returns_unique_task_id(self):
        oc = self._make_collector(lambda *a: {"empty": True})
        try:
            ev = torch.cuda.Event()
            self.assertNotEqual(
                oc.dispatch([1, 2], False, ev), oc.dispatch([3], False, ev)
            )
        finally:
            oc.shutdown()

    def test_try_get_result_none_before_done(self):
        def slow_collect(task_id, block_ids, boost_cpu, ev):
            time.sleep(0.5)
            return {"empty": True, "rank": 0, "metas": []}

        oc = self._make_collector(slow_collect)
        try:
            ev = torch.cuda.Event()
            oc.dispatch(None, False, ev)
            self.assertIsNone(oc.try_get_result())
        finally:
            oc.shutdown()

    def test_try_get_result_after_done(self):
        def collect_fn(task_id, block_ids, boost_cpu, ev):
            return {"rank": 0, "shm_name": "x", "shm_size": 0,
                    "metas": [], "task_id": task_id}

        oc = self._make_collector(collect_fn)
        try:
            ev = torch.cuda.Event()
            oc.dispatch(None, False, ev)
            res = self._wait_result(oc)
            self.assertIsNotNone(res)
            self.assertEqual(res["shm_name"], "x")
        finally:
            oc.shutdown()

    def test_collect_error_written_not_raised(self):
        def failing_collect(task_id, block_ids, boost_cpu, ev):
            raise RuntimeError("boom")

        oc = self._make_collector(failing_collect)
        try:
            ev = torch.cuda.Event()
            oc.dispatch(None, False, ev)
            res = self._wait_result(oc)
            self.assertIsNotNone(res)
            self.assertIn("error", res)
            self.assertIn("boom", res["error"])
        finally:
            oc.shutdown()


class TestPinnedMemoryPool(unittest.TestCase):
    def test_acquire_release_reuse(self):
        pool = PinnedMemoryPool(1024)
        self.assertTrue(pool.available)
        buf = pool.acquire()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.numel(), 1024)
        # Reuses the same buffer, no new allocation
        self.assertIs(pool.acquire(), buf)
        pool.release(buf)

    def test_zero_bytes_unavailable(self):
        pool = PinnedMemoryPool(0)
        self.assertFalse(pool.available)
        self.assertIsNone(pool.acquire())

    def test_buffer_is_pinned(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        pool = PinnedMemoryPool(1024)
        if pool.available:
            self.assertTrue(pool.buffer.is_pinned())


class TestStreamD2HEncode(unittest.TestCase):
    """End-to-end verification of streaming pinned D2H + encode-to-shm correctness (layer-major)."""

    @staticmethod
    def _make_kv_collector():
        model_runner = SimpleNamespace(
            kv_caches=[
                torch.arange(8 * 4 * 4, dtype=torch.float32,
                             device="cuda:0").reshape(8, 4, 4),
                torch.arange(8 * 4 * 2, dtype=torch.float32,
                             device="cuda:0").reshape(8, 4, 2),
            ],
            device=torch.device("cuda:0"),
        )
        return KvCollector(model_runner, maxsize=4)

    def test_stream_encode_matches_source(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        oc = self._make_kv_collector()
        try:
            kv_caches = oc._kv_caches
            ranges = [(0, 8)]
            layer_block_bytes = oc._layer_block_bytes()
            metas, total = oc._compute_metas(
                kv_caches, ranges, layer_block_bytes, 0
            )
            self.assertEqual(len(metas), 2)
            self.assertEqual(metas[0]["nbytes"], 8 * 4 * 4 * 4)
            self.assertEqual(metas[1]["nbytes"], 8 * 4 * 2 * 4)
            self.assertEqual(metas[1]["byte_offset"], metas[0]["nbytes"])

            shm = shared_memory.SharedMemory(create=True, size=total)
            try:
                batches = list(oc._smooth_throttler.batches(ranges))
                oc._stream_d2h_encode(
                    kv_caches, batches, layer_block_bytes, metas, shm,
                    boost_cpu=False, skip_d2h=False, skip_encode=False,
                    copy_stream=oc._copy_stream,
                    throttler=oc._smooth_throttler,
                )
                shm_buf = torch.frombuffer(shm.buf, dtype=torch.uint8)
                for li, t in enumerate(kv_caches):
                    src = t.detach().flatten().view(torch.uint8)
                    dst = shm_buf[
                        metas[li]["byte_offset"]:metas[li]["byte_offset"]
                        + metas[li]["nbytes"]
                    ]
                    self.assertTrue(torch.equal(src.cpu(), dst))
                del shm_buf
            finally:
                shm.close()
                shm.unlink()
        finally:
            oc.shutdown()


if __name__ == "__main__":
    unittest.main()
