# SPDX-License-Identifier: Apache-2.0
"""RPC method-name contract injected by vllm_hi into EngineCore / Worker.

These ``_vllm_hi_*`` attribute names are injected onto ``EngineCore``/``Worker``
by the patch layer, and referenced by string on the data plane (collector /
receiver) via ``call_utility_async`` / ``collective_rpc_async``. Centralizing the
strings as constants avoids the cross-module "magic string contract" being
missed on rename.

Note: these attribute names are a cross-process RPC contract, so the constant
values themselves must not change; this module only declares them centrally.
"""

from __future__ import annotations

# --- EngineCore (call_utility_async) ---
RPC_GET_SCHEDULER_STATE = "_vllm_hi_get_scheduler_state"
RPC_SET_SCHEDULER_STATE = "_vllm_hi_set_scheduler_state"
RPC_APPLY_SCHEDULER_INCREMENTAL = "_vllm_hi_apply_scheduler_incremental"

# --- Worker (collective_rpc_async) ---
RPC_EXPORT_KV_IPC = "_vllm_hi_export_kv_ipc"
RPC_SET_KV_IPC = "_vllm_hi_set_kv_ipc"
RPC_COLLECT_KV_CACHE = "_vllm_hi_collect_kv_cache"
RPC_TRY_GET_RESULT = "_vllm_hi_try_get_result"


__all__ = [
    "RPC_GET_SCHEDULER_STATE",
    "RPC_SET_SCHEDULER_STATE",
    "RPC_APPLY_SCHEDULER_INCREMENTAL",
    "RPC_EXPORT_KV_IPC",
    "RPC_SET_KV_IPC",
    "RPC_COLLECT_KV_CACHE",
    "RPC_TRY_GET_RESULT",
]
