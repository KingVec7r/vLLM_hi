#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hot-migration test with the TCP backend and test-1 topology (B tp=2 pp=1).

Reads test_config.json, starts the server, runs inference + migration, and saves the
results to tcp_1_result.json.
"""

from __future__ import annotations

import sys

from test_runner import run_test

if __name__ == "__main__":
    sys.exit(run_test("tcp", 1))
