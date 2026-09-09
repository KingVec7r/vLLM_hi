# SPDX-License-Identifier: Apache-2.0
"""vllm_hi's vllm.general_plugins entry point.

This function is called in every vllm process (process0, engine core, worker),
ensuring the engine_internals patch also takes effect in worker subprocesses.

Note: this function is called multiple times (once per process) and must be
idempotent.
"""

from __future__ import annotations


def load() -> None:
    """vllm.general_plugins entry point: apply the vllm_hi patches."""
    # Only apply the engine_internals patch (the others are applied by
    # process0's `import vllm_hi`). This way worker subprocesses also get
    # methods such as _vllm_hi_export_kv_ipc.
    import os
    import logging

    logger = logging.getLogger("vllm.vllm_hi.patches.plugin")
    from vllm_hi.common.rpc import RPC_EXPORT_KV_IPC
    from vllm_hi.patches import patch_engine_internals

    patch_engine_internals()

    # Verify the method has been injected
    from vllm.v1.worker.gpu_worker import Worker
    has_method = hasattr(Worker, RPC_EXPORT_KV_IPC)
    logger.info(
        "[vllm_hi plugin] pid=%s Worker.%s=%s",
        os.getpid(), RPC_EXPORT_KV_IPC, has_method,
    )
