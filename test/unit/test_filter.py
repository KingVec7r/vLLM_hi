# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the request filter.

Covers:
1. default MigrateAllFilter migrates everything;
2. load_request_filter falls back to default when unspecified or on an invalid path;
3. a custom filter is loaded via its fully-qualified class name.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from vllm_hi.migration.filter import (
    MigrateAllFilter,
    RequestFilter,
    load_request_filter,
)


class TestMigrateAllFilter(unittest.TestCase):
    def test_all_requests_migrate(self):
        f = MigrateAllFilter()
        rs = SimpleNamespace(output_token_ids=[1, 2, 3])
        self.assertTrue(f.should_migrate("seq-1", rs))
        self.assertTrue(f.should_migrate("seq-2", None))


class TestLoadRequestFilter(unittest.TestCase):
    def test_none_returns_default(self):
        self.assertIsInstance(load_request_filter(None), MigrateAllFilter)

    def test_empty_returns_default(self):
        self.assertIsInstance(load_request_filter(""), MigrateAllFilter)

    def test_invalid_path_returns_default(self):
        self.assertIsInstance(
            load_request_filter("no_such_module.Filter"), MigrateAllFilter
        )

    def test_load_custom_filter(self):
        f = load_request_filter(
            "test.unit.test_filter._OnlyLongRequests"
        )
        self.assertIsInstance(f, RequestFilter)
        self.assertTrue(f.should_migrate("a", SimpleNamespace(output_token_ids=list(range(100)))))
        self.assertFalse(f.should_migrate("b", SimpleNamespace(output_token_ids=[1])))


class _OnlyLongRequests(RequestFilter):
    """Example custom filter: only migrate requests whose length exceeds the threshold (output token count > 10)."""

    def should_migrate(self, seq_id, request_state) -> bool:
        tokens = list(getattr(request_state, "output_token_ids", None) or [])
        return len(tokens) > 10


if __name__ == "__main__":
    unittest.main()
