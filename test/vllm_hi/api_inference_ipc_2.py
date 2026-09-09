#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hot-migration test with the IPC backend and test-2 topology (B tp=1 pp=2).

Reads test_config.json, starts the server, runs inference + migration, and saves the
results to ipc_2_result.json.
"""

from __future__ import annotations

import sys

from test_runner import run_test

if __name__ == "__main__":
    sys.exit(run_test("ipc", 2))
