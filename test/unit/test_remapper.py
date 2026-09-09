# SPDX-License-Identifier: Apache-2.0
"""Unit tests for head-shard mapping of TopologyRemapper.remap_kv_cache.

Covers TP expansion (2->4), TP shrink (4->2), and TP unchanged (2->2), plus the
scenario where a non-integer ratio raises an exception.

Run: source .../conda.sh && conda activate vllm_hi && \
      python -m unittest test.unit.test_remapper -v
"""

from __future__ import annotations

import unittest

from vllm_hi.migration.remapper import TopologyRemapper
from vllm_hi.migration.state_snapshot import KVTensorMeta


def _topo(tp: int, pp: int = 1) -> dict:
    return {"tensor_parallel_size": tp, "pipeline_parallel_size": pp}


def _metas(src_tp: int, src_heads: int, n_layers: int = 2) -> dict[int, list[KVTensorMeta]]:
    """Constructs src_tp KV tensor metas, each rank/layer with shape=[100, src_heads, 128]."""
    metas: dict[int, list[KVTensorMeta]] = {}
    for rank in range(src_tp):
        metas[rank] = [
            KVTensorMeta(
                layer_idx=li,
                shape=[100, src_heads, 128],
                dtype="torch.float16",
                byte_offset=0,
                nbytes=100 * src_heads * 128 * 2,
                head_dim=1,
            )
            for li in range(n_layers)
        ]
    return metas


def _pp_metas(
    src_tp: int, src_pp: int, num_layers: int, src_heads: int,
) -> dict[int, list[KVTensorMeta]]:
    """Constructs KV metadata with a PP dimension (keys are source global ranks, layer_idx is the global layer index).

    The layer range is split contiguously using ``get_pp_indices``, consistent with
    vLLM, matching the ``layer_idx = start_layer + i`` that the collector actually produces.
    """
    from vllm.distributed.utils import get_pp_indices

    metas: dict[int, list[KVTensorMeta]] = {}
    for pp_rank in range(src_pp):
        start, end = get_pp_indices(num_layers, pp_rank, src_pp)
        for tp_rank in range(src_tp):
            rank = pp_rank * src_tp + tp_rank
            metas[rank] = [
                KVTensorMeta(
                    layer_idx=li,
                    shape=[100, src_heads, 128],
                    dtype="torch.float16",
                    byte_offset=0,
                    nbytes=100 * src_heads * 128 * 2,
                    head_dim=1,
                )
                for li in range(start, end)
            ]
    return metas


class TestRemapKvCache(unittest.TestCase):
    def test_tp_expand_2_to_4(self):
        remapper = TopologyRemapper(_topo(2), _topo(4))
        kv_cache = {r: {0: b"data"} for r in range(2)}
        metas = _metas(2, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        # Data returned as-is, not copied: still src_tp=2 parts
        self.assertEqual(set(kv.keys()), {0, 1})
        # Mapping covers all 4 target ranks
        self.assertEqual(set(shard_map.keys()), {0, 1, 2, 3})

        # ratio = 4 // 2 = 2, dst_head = 8 // 2 = 4
        self.assertEqual(shard_map[0][0].src_rank, 0)
        self.assertEqual(shard_map[0][0].head_offset, 0)
        self.assertEqual(shard_map[0][0].head_len, 4)

        self.assertEqual(shard_map[1][0].src_rank, 0)
        self.assertEqual(shard_map[1][0].head_offset, 4)
        self.assertEqual(shard_map[1][0].head_len, 4)

        self.assertEqual(shard_map[2][0].src_rank, 1)
        self.assertEqual(shard_map[2][0].head_offset, 0)
        self.assertEqual(shard_map[2][0].head_len, 4)

        self.assertEqual(shard_map[3][0].src_rank, 1)
        self.assertEqual(shard_map[3][0].head_offset, 4)
        self.assertEqual(shard_map[3][0].head_len, 4)

    def test_tp_shrink_4_to_2(self):
        remapper = TopologyRemapper(_topo(4), _topo(2))
        kv_cache = {r: {0: b"data"} for r in range(4)}
        metas = _metas(4, src_heads=4)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(kv.keys()), {0, 1, 2, 3})
        self.assertEqual(set(shard_map.keys()), {0, 1})

        # ratio = 4 // 2 = 2, each target rank merges the head segments of 2 source ranks
        shards0 = shard_map[0]
        self.assertEqual(len(shards0), 2)
        self.assertEqual(shards0[0].src_rank, 0)
        self.assertEqual(shards0[0].head_offset, 0)
        self.assertEqual(shards0[0].head_len, 4)
        self.assertEqual(shards0[0].dst_head_offset, 0)

        self.assertEqual(shards0[1].src_rank, 1)
        self.assertEqual(shards0[1].head_len, 4)
        self.assertEqual(shards0[1].dst_head_offset, 4)

        shards1 = shard_map[1]
        self.assertEqual([s.src_rank for s in shards1], [2, 3])
        self.assertEqual([s.dst_head_offset for s in shards1], [0, 4])

    def test_tp_unchanged_2_to_2(self):
        remapper = TopologyRemapper(_topo(2), _topo(2))
        kv_cache = {r: {0: b"data"} for r in range(2)}
        metas = _metas(2, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(shard_map.keys()), {0, 1})
        # head_len=0 means "full head segment", resolved by the worker against the actual tensor head count
        self.assertEqual(shard_map[0][0].src_rank, 0)
        self.assertEqual(shard_map[0][0].head_offset, 0)
        self.assertEqual(shard_map[0][0].head_len, 0)
        self.assertEqual(shard_map[1][0].src_rank, 1)

    def test_tp_expand_non_integer_ratio_raises(self):
        remapper = TopologyRemapper(_topo(2), _topo(3))
        kv_cache = {r: {0: b"data"} for r in range(2)}
        metas = _metas(2, src_heads=8)
        with self.assertRaises(ValueError):
            remapper.remap_kv_cache(kv_cache, metas)

    def test_tp_shrink_non_integer_ratio_raises(self):
        remapper = TopologyRemapper(_topo(3), _topo(2))
        kv_cache = {r: {0: b"data"} for r in range(3)}
        metas = _metas(3, src_heads=6)
        with self.assertRaises(ValueError):
            remapper.remap_kv_cache(kv_cache, metas)

    def test_needs_remap(self):
        self.assertTrue(TopologyRemapper(_topo(2), _topo(4)).needs_remap)
        self.assertFalse(TopologyRemapper(_topo(2), _topo(2)).needs_remap)

    def test_needs_remap_pp(self):
        self.assertTrue(
            TopologyRemapper(_topo(1, 2), _topo(1, 1)).needs_remap
        )
        self.assertTrue(
            TopologyRemapper(_topo(1, 1), _topo(1, 2)).needs_remap
        )
        self.assertFalse(
            TopologyRemapper(_topo(1, 2), _topo(1, 2)).needs_remap
        )

    def test_pp_shrink_2_to_1(self):
        # source pp=2 -> target pp=1, TP unchanged: a single target rank reads data from both source stages.
        remapper = TopologyRemapper(_topo(1, 2), _topo(1, 1))
        kv_cache = {r: {0: b"data"} for r in range(2)}
        metas = _pp_metas(1, 2, num_layers=4, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(kv.keys()), {0, 1})
        # Only 1 global rank on the target, and its shards cover both source stages (global ranks 0 and 1).
        self.assertEqual(set(shard_map.keys()), {0})
        self.assertEqual(
            [s.src_rank for s in shard_map[0]], [0, 1]
        )
        # TP unchanged: head_len=0 (full head).
        self.assertTrue(all(s.head_len == 0 for s in shard_map[0]))

    def test_pp_expand_1_to_2(self):
        # source pp=1 -> target pp=2: both target stages read from the single source stage (filtered by layer index).
        remapper = TopologyRemapper(_topo(1, 1), _topo(1, 2))
        kv_cache = {0: {0: b"data"}}
        metas = _pp_metas(1, 1, num_layers=4, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(shard_map.keys()), {0, 1})
        self.assertEqual([s.src_rank for s in shard_map[0]], [0])
        self.assertEqual([s.src_rank for s in shard_map[1]], [0])

    def test_pp_unchanged_2_to_2(self):
        # source pp=2 -> target pp=2 (PP topology unchanged): each target stage reads only its corresponding source stage.
        remapper = TopologyRemapper(_topo(1, 2), _topo(1, 2))
        kv_cache = {r: {0: b"data"} for r in range(2)}
        metas = _pp_metas(1, 2, num_layers=4, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(shard_map.keys()), {0, 1})
        self.assertEqual([s.src_rank for s in shard_map[0]], [0])
        self.assertEqual([s.src_rank for s in shard_map[1]], [1])

    def test_tp_pp_double_shrink_2x2_to_1x1(self):
        # source tp=2 pp=2 -> target tp=1 pp=1: TP shrink (head merge) + PP shrink (two stages).
        remapper = TopologyRemapper(_topo(2, 2), _topo(1, 1))
        kv_cache = {r: {0: b"data"} for r in range(4)}
        metas = _pp_metas(2, 2, num_layers=4, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        self.assertEqual(set(shard_map.keys()), {0})
        shards = shard_map[0]
        # 2 tp ranks within each source stage merge into one target rank, for a total of 4 source ranks.
        self.assertEqual([s.src_rank for s in shards], [0, 1, 2, 3])
        # Head merge within each stage: rank0->offset0, rank1->offset8; rank2->0, rank3->8.
        self.assertEqual([s.dst_head_offset for s in shards], [0, 8, 0, 8])
        self.assertEqual([s.head_len for s in shards], [8, 8, 8, 8])

    def test_tp_pp_double_expand_1x1_to_2x2(self):
        # source tp=1 pp=1 -> target tp=2 pp=2: TP expansion (head split) + PP expansion.
        remapper = TopologyRemapper(_topo(1, 1), _topo(2, 2))
        kv_cache = {0: {0: b"data"}}
        metas = _pp_metas(1, 1, num_layers=4, src_heads=8)
        kv, shard_map = remapper.remap_kv_cache(kv_cache, metas)

        # 4 target global ranks, all reading from the single source rank 0.
        self.assertEqual(set(shard_map.keys()), {0, 1, 2, 3})
        # ratio = 2, dst_head = 8 // 2 = 4
        self.assertEqual(shard_map[0][0].head_offset, 0)
        self.assertEqual(shard_map[0][0].head_len, 4)
        self.assertEqual(shard_map[1][0].head_offset, 4)
        self.assertEqual(shard_map[1][0].head_len, 4)
        self.assertEqual(shard_map[2][0].head_offset, 0)
        self.assertEqual(shard_map[3][0].head_offset, 4)


if __name__ == "__main__":
    unittest.main()
