# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KV cache depickle-free serialization (contiguous memory buffer + metadata).

Covers:
1. pickle round-trips of KVTensorMeta / HeadShard;
2. buffer encoding (torch tensor -> shm) and element-wise equivalence of
   torch.frombuffer zero-copy reconstruction;
3. StateCollector._aggregate_kv_chunks: chunked aggregation is semantically equivalent
   to one-shot collection.

Run: source .../conda.sh && conda activate vllm_hi && \
      python -m unittest test.unit.test_buffer_encoding -v
"""

from __future__ import annotations

import pickle
import unittest
from multiprocessing import shared_memory

import torch

from vllm_hi.migration.collector import StateCollector
from vllm_hi.migration.state_snapshot import HeadShard, KVTensorMeta


class TestMetadataPickle(unittest.TestCase):
    def test_kv_tensor_meta_roundtrip(self):
        meta = KVTensorMeta(
            layer_idx=3, shape=[128, 8, 128], dtype="torch.float16",
            byte_offset=4096, nbytes=262144, head_dim=1,
        )
        restored = pickle.loads(pickle.dumps(meta))
        self.assertEqual(restored.layer_idx, 3)
        self.assertEqual(restored.shape, [128, 8, 128])
        self.assertEqual(restored.dtype, "torch.float16")
        self.assertEqual(restored.byte_offset, 4096)
        self.assertEqual(restored.nbytes, 262144)

    def test_head_shard_roundtrip(self):
        shard = HeadShard(src_rank=1, head_offset=4, head_len=4, dst_head_offset=8)
        restored = pickle.loads(pickle.dumps(shard))
        self.assertEqual(restored.src_rank, 1)
        self.assertEqual(restored.head_offset, 4)
        self.assertEqual(restored.head_len, 4)
        self.assertEqual(restored.dst_head_offset, 8)


class TestBufferEncoding(unittest.TestCase):
    def test_frombuffer_roundtrip_single_layer(self):
        """torch tensor -> shm bytes -> torch.frombuffer reconstruction, element-wise equal."""
        t = torch.randn(4, 4, 8, dtype=torch.float32)
        tc = t.contiguous()
        nbytes = tc.numel() * tc.element_size()
        byte_arr = tc.view(torch.uint8).reshape(-1).numpy()

        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        try:
            shm.buf[:nbytes] = byte_arr.tobytes()
            src = torch.frombuffer(
                shm.buf, dtype=torch.float32, count=tc.numel(), offset=0
            ).reshape(tc.shape)
            self.assertTrue(torch.equal(t, src))
        finally:
            shm.close()
            shm.unlink()

    def test_frombuffer_roundtrip_multi_layer_offset(self):
        """Encode/decode round-trip of a multi-layer contiguous buffer with byte_offset."""
        t0 = torch.randn(4, 4, 8, dtype=torch.float32)
        t1 = torch.randn(4, 4, 8, dtype=torch.float32)
        tensors = [t0.contiguous(), t1.contiguous()]

        metas = []
        total = 0
        for i, tc in enumerate(tensors):
            nbytes = tc.numel() * tc.element_size()
            metas.append({
                "layer_idx": i,
                "shape": list(tc.shape),
                "dtype": str(tc.dtype),
                "byte_offset": total,
                "nbytes": nbytes,
            })
            total += nbytes

        shm = shared_memory.SharedMemory(create=True, size=total)
        try:
            for i, tc in enumerate(tensors):
                m = metas[i]
                byte_arr = tc.view(torch.uint8).reshape(-1).numpy()
                shm.buf[m["byte_offset"]:m["byte_offset"] + m["nbytes"]] = (
                    byte_arr.tobytes()
                )
            restored = []
            for m in metas:
                dtype = getattr(torch, str(m["dtype"]).split(".")[-1])
                numel = 1
                for d in m["shape"]:
                    numel *= d
                src = torch.frombuffer(
                    shm.buf, dtype=dtype, count=numel,
                    offset=int(m["byte_offset"]),
                ).reshape(m["shape"])
                restored.append(src)
            self.assertTrue(torch.equal(restored[0], t0))
            self.assertTrue(torch.equal(restored[1], t1))
        finally:
            shm.close()
            shm.unlink()


def _encode_chunk(t: torch.Tensor) -> tuple[bytes, KVTensorMeta]:
    """Simulates the encoding in _get_kv_cache: returns (bytes, meta)."""
    tc = t.contiguous()
    byte_arr = tc.view(torch.uint8).reshape(-1).numpy()
    return byte_arr.tobytes(), KVTensorMeta(
        layer_idx=0, shape=list(tc.shape), dtype=str(tc.dtype),
        byte_offset=0, nbytes=len(byte_arr), head_dim=1,
    )


class TestAggregateKvChunks(unittest.TestCase):
    def test_chunked_aggregate_equals_full(self):
        """Aggregating 2 chunks by rank/layer matches the one-shot full tensor."""
        full = torch.randn(4, 4, 8, dtype=torch.float32)
        chunks = [full[0:2].contiguous(), full[2:4].contiguous()]

        per_chunk = []
        for c in chunks:
            buf, meta = _encode_chunk(c)
            kv_cache = {0: {0: memoryview(buf)}}
            metas = {0: [meta]}
            per_chunk.append((kv_cache, metas))

        final_kv, final_metas = StateCollector._aggregate_kv_chunks(per_chunk)

        # Only one rank, and after aggregation shape[0] should be 4 (blocks summed across the two chunks)
        self.assertEqual(set(final_kv.keys()), {0})
        meta = final_metas[0][0]
        self.assertEqual(meta.shape, [4, 4, 8])
        self.assertEqual(meta.nbytes, full.numel() * full.element_size())

        # Decode the aggregated buffer and compare element-wise with the original full tensor
        buf = final_kv[0][0]
        src = torch.frombuffer(
            buf, dtype=torch.float32, count=full.numel(), offset=0
        ).reshape(full.shape)
        self.assertTrue(torch.equal(src, full))

    def test_chunked_aggregate_multi_layer(self):
        """Multi-layer (2 layers) chunked aggregation: each layer is concatenated across chunks without crossing layers."""
        full0 = torch.randn(4, 4, 8, dtype=torch.float32)
        full1 = torch.randn(4, 4, 8, dtype=torch.float32)

        per_chunk = []
        for lo, hi in ((0, 2), (2, 4)):
            metas = []
            total = 0
            parts = []
            for t in (full0, full1):
                c = t[lo:hi].contiguous()
                byte_arr = c.view(torch.uint8).reshape(-1).numpy().tobytes()
                metas.append(KVTensorMeta(
                    layer_idx=len(parts), shape=list(c.shape),
                    dtype=str(c.dtype), byte_offset=total,
                    nbytes=len(byte_arr), head_dim=1,
                ))
                parts.append(byte_arr)
                total += len(byte_arr)
            kv_cache = {0: {0: memoryview(b"".join(parts))}}
            per_chunk.append((kv_cache, {0: metas}))

        final_kv, final_metas = StateCollector._aggregate_kv_chunks(per_chunk)

        self.assertEqual(len(final_metas[0]), 2)
        self.assertEqual(final_metas[0][0].shape, [4, 4, 8])
        self.assertEqual(final_metas[0][1].shape, [4, 4, 8])

        buf = final_kv[0][0]
        m0 = final_metas[0][0]
        m1 = final_metas[0][1]
        r0 = torch.frombuffer(buf, dtype=torch.float32, count=full0.numel(),
                              offset=m0.byte_offset).reshape(full0.shape)
        r1 = torch.frombuffer(buf, dtype=torch.float32, count=full1.numel(),
                              offset=m1.byte_offset).reshape(full1.shape)
        self.assertTrue(torch.equal(r0, full0))
        self.assertTrue(torch.equal(r1, full1))

    def test_aggregate_preserves_global_layer_idx(self):
        """Under PP, the source meta's global layer indices must be preserved after chunked aggregation (not reset to local indices).

        When the source has PP>1, each rank holds only a subset of layers and layer_idx
        records the global layer index (e.g. 16/17); chunked aggregation must not reset
        layer_idx to local indices 0/1.
        """
        full0 = torch.randn(4, 4, 8, dtype=torch.float32)
        full1 = torch.randn(4, 4, 8, dtype=torch.float32)

        per_chunk = []
        for lo, hi in ((0, 2), (2, 4)):
            metas = []
            total = 0
            parts = []
            for global_li, t in ((16, full0), (17, full1)):
                c = t[lo:hi].contiguous()
                byte_arr = c.view(torch.uint8).reshape(-1).numpy().tobytes()
                metas.append(KVTensorMeta(
                    layer_idx=global_li, shape=list(c.shape),
                    dtype=str(c.dtype), byte_offset=total,
                    nbytes=len(byte_arr), head_dim=1,
                ))
                parts.append(byte_arr)
                total += len(byte_arr)
            kv_cache = {0: {0: memoryview(b"".join(parts))}}
            per_chunk.append((kv_cache, {0: metas}))

        _, final_metas = StateCollector._aggregate_kv_chunks(per_chunk)

        self.assertEqual(len(final_metas[0]), 2)
        self.assertEqual(final_metas[0][0].layer_idx, 16)
        self.assertEqual(final_metas[0][1].layer_idx, 17)


if __name__ == "__main__":
    unittest.main()
