# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the D2HThrottler rate limiter.

Covers:
1. correctness of splitting the batches interval (split by batch_blocks, intervals may span batches);
2. yield_interval decreases as hbm_budget_ratio increases;
3. an empty interval list yields an empty iteration;
4. invalid parameters fall back to defaults.

Run: source .../conda.sh && conda activate vllm_hi && \
      python -m unittest test.unit.test_d2h_throttler -v
"""

from __future__ import annotations

import unittest

from vllm_hi.migration.kv_collector import (
    D2HThrottler,
    SmoothD2HThrottler,
)


class TestD2HThrottler(unittest.TestCase):
    def test_batches_single_range_split(self):
        th = D2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=3, peak_hbm_bw=1000e9,
        )
        self.assertEqual(
            list(th.batches([(0, 10)])),
            [[(0, 3)], [(3, 6)], [(6, 9)], [(9, 10)]],
        )

    def test_batches_multi_range(self):
        th = D2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=4, peak_hbm_bw=1000e9,
        )
        self.assertEqual(
            list(th.batches([(0, 3), (5, 6), (10, 14)])),
            [[(0, 3), (5, 6)], [(10, 14)]],
        )

    def test_batches_empty(self):
        th = D2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=3, peak_hbm_bw=1000e9,
        )
        self.assertEqual(list(th.batches([])), [])

    def test_yield_interval_decreases_with_ratio(self):
        bw = 1000e9
        lo = D2HThrottler(
            hbm_budget_ratio=0.1, batch_blocks=3, peak_hbm_bw=bw,
        )
        hi = D2HThrottler(
            hbm_budget_ratio=0.8, batch_blocks=3, peak_hbm_bw=bw,
        )
        self.assertGreater(lo.yield_interval(1e6), hi.yield_interval(1e6))

    def test_yield_interval_zero_for_zero_bytes(self):
        th = D2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=3, peak_hbm_bw=1000e9,
        )
        self.assertEqual(th.yield_interval(0), 0.0)

    def test_invalid_params_fallback_default(self):
        th = D2HThrottler(
            hbm_budget_ratio=0.0, batch_blocks=-1, peak_hbm_bw=1000e9,
        )
        self.assertEqual(th.hbm_budget_ratio, 0.25)
        self.assertEqual(th.batch_blocks, 64)
        th2 = D2HThrottler(
            hbm_budget_ratio=2.0, batch_blocks=0, peak_hbm_bw=1000e9,
        )
        self.assertEqual(th2.hbm_budget_ratio, 0.25)
        self.assertEqual(th2.batch_blocks, 64)


class TestSmoothD2HThrottler(unittest.TestCase):
    def test_default_batch_blocks_is_16(self):
        th = SmoothD2HThrottler(peak_hbm_bw=1000e9)
        self.assertEqual(th.batch_blocks, 16)

    def test_batches_splits_by_default_16(self):
        th = SmoothD2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=16, peak_hbm_bw=1000e9,
        )
        batches = list(th.batches([(0, 40)]))
        self.assertEqual(
            batches,
            [[(0, 16)], [(16, 32)], [(32, 40)]],
        )

    def test_batches_empty(self):
        th = SmoothD2HThrottler(peak_hbm_bw=1000e9)
        self.assertEqual(list(th.batches([])), [])

    def test_next_gap_converges_to_zero_when_budget_met(self):
        bw = 1000e9
        th = SmoothD2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=16, peak_hbm_bw=bw,
        )
        batch_bytes = 250e6  # 250MB, budget_time = 1ms
        # measured time == budget time -> no yield needed
        self.assertEqual(th.next_gap(1.0, int(batch_bytes)), 0.0)

    def test_next_gap_positive_when_faster_than_budget(self):
        bw = 1000e9
        th = SmoothD2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=16, peak_hbm_bw=bw,
        )
        batch_bytes = int(250e6)
        # measured 0.5ms is faster than the 1ms budget -> yield 0.5ms
        self.assertAlmostEqual(
            th.next_gap(0.5, batch_bytes), 0.0005, places=6
        )

    def test_next_gap_zero_when_slower_than_budget(self):
        bw = 1000e9
        th = SmoothD2HThrottler(
            hbm_budget_ratio=0.25, batch_blocks=16, peak_hbm_bw=bw,
        )
        batch_bytes = int(250e6)
        # measured 2ms is slower than the 1ms budget -> no yield
        self.assertEqual(th.next_gap(2.0, batch_bytes), 0.0)

    def test_next_gap_zero_for_zero_bytes(self):
        th = SmoothD2HThrottler(peak_hbm_bw=1000e9)
        self.assertEqual(th.next_gap(1.0, 0), 0.0)

    def test_invalid_params_fallback(self):
        th = SmoothD2HThrottler(
            hbm_budget_ratio=0.0, batch_blocks=-1, peak_hbm_bw=1000e9,
        )
        self.assertEqual(th.hbm_budget_ratio, 0.25)
        self.assertEqual(th.batch_blocks, 64)


if __name__ == "__main__":
    unittest.main()
