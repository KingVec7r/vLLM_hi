# SPDX-License-Identifier: Apache-2.0
"""vllm_hi's monkey-patching logic for vllm.

All patches are implemented by "wrapping the original function", without
modifying the vllm-main source, so this package can be maintained
independently as vllm versions evolve.

This module only handles "wrapping + injection": the concrete patch
implementations are moved down into
- ``vllm_hi.patches.engine_rpc`` (EngineCore/Worker state machine)
- ``vllm_hi.instance.registration`` (type-A auto-registration)
- ``vllm_hi.common.utils`` (NUMA affinity binding)
"""

from __future__ import annotations

import functools
import json
from typing import Any

from vllm.logger import init_logger

# Note: the "vllm." prefix is required, otherwise logs are dropped by vllm's
# propagate=False configuration
logger = init_logger("vllm.vllm_hi")

# Names of patches already applied (idempotency guard)
_APPLIED: set[str] = set()

# Patch status exposed to users/tests
PATCH_STATUS: dict[str, bool] = {}


def _mark(name: str) -> None:
    _APPLIED.add(name)
    PATCH_STATUS[name] = True


def apply() -> dict[str, bool]:
    """Apply all patches (idempotent). Returns the patch application status."""
    patch_build_app()
    patch_chat_completion_logging()
    patch_cli_args()
    patch_build_async_engine_client()
    patch_init_app_state()
    patch_run_server()
    patch_engine_internals()
    logger.info(
        "[vllm_hi] vllm customizations active, patch list: %s",
        sorted(_APPLIED),
    )
    return dict(PATCH_STATUS)


def patch_cli_args() -> None:
    """Add vllm_hi-specific arguments to the vllm CLI:
    --vllm-hi-type: instance type (A/B)
    --vllm-hi-host: Host scheduler address (ip:port)
    --vllm-hi-data-port: data-plane TCP port
    """
    name = "cli_args"
    if name in _APPLIED:
        return

    from vllm.entrypoints.openai.cli_args import make_arg_parser as _orig_make_arg_parser

    @functools.wraps(_orig_make_arg_parser)
    def make_arg_parser_with_vllm_hi(parser):
        parser = _orig_make_arg_parser(parser)
        # Avoid duplicate additions (idempotent)
        existing = {tuple(a.option_strings) for a in parser._actions}
        vllm_hi_args = [
            (["--vllm-hi-type"], {
                "type": str, "default": None,
                "choices": ["A", "B"],
                "help": "vllm_hi instance type (A=active inference, B=standby)",
            }),
            (["--vllm-hi-host"], {
                "type": str, "default": None,
                "help": "vllm_hi Host scheduler address (ip:port)",
            }),
            (["--vllm-hi-data-port"], {
                "type": int, "default": 0,
                "help": "vllm_hi data-plane TCP port (for state transfer)",
            }),
        ]
        for opts, kwargs in vllm_hi_args:
            if tuple(opts) not in existing:
                parser.add_argument(*opts, **kwargs)
        return parser

    import vllm.entrypoints.openai.cli_args as cli_args_module

    cli_args_module.make_arg_parser = make_arg_parser_with_vllm_hi
    _mark(name)


def patch_init_app_state() -> None:
    """Wrap init_app_state to inject vllm_hi state and skip serving state for B.

    - type A: call the native init_app_state (full serving state), then inject
      ``app.state.vllm_hi`` and auto-register with the Host.
    - type B: skip the native init_app_state (the placeholder engine has no
      model), inject ``app.state.vllm_hi`` (instance_type="B"), set the
      placeholder engine, and register with the Host.

    The migration endpoint routes are mounted in patch_build_app (unified A/B).
    """
    name = "init_app_state"
    if name in _APPLIED:
        return

    import asyncio

    from vllm.entrypoints.launchers.api_server import app_state as app_state_module

    original_init_app_state = app_state_module.init_app_state

    @functools.wraps(original_init_app_state)
    async def init_app_state_with_vllm_hi(
        engine_client, state, args, supported_tasks=None,
    ):
        from vllm_hi.common.state import VllmHiState
        from vllm_hi.instance.host_client import HostClient

        vllm_hi_type = getattr(args, "vllm_hi_type", None)
        vllm_hi_host = getattr(args, "vllm_hi_host", None)
        vllm_hi_data_port = getattr(args, "vllm_hi_data_port", 0) or 0

        if vllm_hi_type == "B":
            # B standby: skip the serving state (placeholder engine has no
            # model); inject vllm_hi state + set the placeholder engine +
            # register with the Host.
            try:
                state.engine_client = engine_client  # placeholder (serve_http watchdog)
                hi = getattr(state, "vllm_hi", None)
                if hi is None:
                    hi = VllmHiState()
                    state.vllm_hi = hi
                hi.instance_type = "B"
                hi.host_addr = vllm_hi_host
                hi.data_port = vllm_hi_data_port
                hi.engine_client = engine_client
                if vllm_hi_host and hi.host_client is None:
                    hi.host_client = HostClient(vllm_hi_host)
                hi.registered = False
                hi.paused = False
                if vllm_hi_host:
                    from vllm_hi.instance.standby import _register_b_to_host

                    asyncio.create_task(_register_b_to_host(state, args, vllm_hi_host))
                logger.info(
                    "[vllm_hi] init_app_state skipped serving state for B: host=%s data_port=%s",
                    vllm_hi_host, vllm_hi_data_port,
                )
            except Exception:  # noqa: BLE001
                logger.exception("[vllm_hi] failed to initialize B standby state")
            return

        await original_init_app_state(
            engine_client, state, args, supported_tasks=supported_tasks
        )
        # Inject vllm_hi state (centralized in the single app.state.vllm_hi
        # dataclass, see common.state)
        try:
            hi = getattr(state, "vllm_hi", None)
            if hi is None:
                hi = VllmHiState()
                state.vllm_hi = hi
            hi.instance_type = vllm_hi_type
            hi.host_addr = vllm_hi_host
            hi.data_port = vllm_hi_data_port
            hi.engine_client = engine_client
            if vllm_hi_host and hi.host_client is None:
                hi.host_client = HostClient(vllm_hi_host)
            hi.registered = False
            hi.paused = False

            # Type-A instance: auto-register with the Host after startup
            if vllm_hi_type == "A" and vllm_hi_host:
                from vllm_hi.instance.registration import (
                    auto_register_active_instance,
                )

                asyncio.create_task(
                    auto_register_active_instance(state, args)
                )

            logger.info(
                "[vllm_hi] init_app_state extended: type=%s host=%s data_port=%s",
                vllm_hi_type, vllm_hi_host, vllm_hi_data_port,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[vllm_hi] failed to inject vllm_hi state; native functionality is unaffected")

    # Keep a reference to the original init_app_state so the B instance can fill
    # serving state with the real engine later (see _init_serving_state), without
    # re-triggering the vllm_hi injection.
    app_state_module._vllm_hi_original_init_app_state = original_init_app_state
    app_state_module.init_app_state = init_app_state_with_vllm_hi
    _mark(name)


def patch_build_async_engine_client() -> None:
    """For a type-B instance, yield a placeholder engine instead of a real one.

    ``build_async_engine_client`` is an async generator used as
    ``async with build_async_engine_client(args) as engine:`` by the native
    ``run_server_worker``. Wrapping it here lets B go through the native pipeline
    while skipping weight loading / worker spawn; the real engine is created later
    on the ``instantiate`` migration endpoint.
    """
    name = "build_async_engine_client"
    if name in _APPLIED:
        return

    from contextlib import asynccontextmanager

    from vllm.entrypoints.launchers.api_server import entry as entry_module

    original = entry_module.build_async_engine_client

    @asynccontextmanager
    async def build_async_engine_client_with_vllm_hi(*args, **kwargs):
        from vllm_hi.instance.standby import _StandbyEngineClient

        # The CLI namespace is the first positional argument.
        namespace = args[0] if args else kwargs.get("args")
        if getattr(namespace, "vllm_hi_type", None) == "B":
            yield _StandbyEngineClient()
            return
        async with original(*args, **kwargs) as engine:
            yield engine

    entry_module.build_async_engine_client = build_async_engine_client_with_vllm_hi
    _mark(name)


def patch_run_server() -> None:
    """Wrap run_server: bind the process to its GPU's NUMA node, then run the
    native pipeline.

    Both A and B now go through the native ``run_server``; the type-specific
    differences are pushed down into ``build_async_engine_client`` (placeholder
    engine for B) and ``init_app_state`` (skip serving state for B).
    """
    name = "run_server"
    if name in _APPLIED:
        return

    from vllm.entrypoints.launchers.api_server import entry as entry_module

    original_run_server = entry_module.run_server

    @functools.wraps(original_run_server)
    async def run_server_with_vllm_hi(args, **uvicorn_kwargs):
        if getattr(args, "vllm_hi_type", None) in ("A", "B"):
            from vllm_hi.common.utils import bind_process_to_device_numa

            bind_process_to_device_numa()
        await original_run_server(args, **uvicorn_kwargs)

    entry_module.run_server = run_server_with_vllm_hi
    _mark(name)


def patch_build_app() -> None:
    """For the vllm API server:
    1. attach both A-side and B-side migration endpoints (unified A/B app);
    2. add a custom route ``GET /vllm_hi/status``;
    3. attach the ``X-VLLM-HI`` response header to all responses.
    """
    name = "build_app"
    if name in _APPLIED:
        return

    import sys

    from vllm.entrypoints.launchers import app as app_module

    original_build_app = app_module.build_app

    @functools.wraps(original_build_app)
    def build_app_with_vllm_hi(*args: Any, **kwargs: Any):
        app = original_build_app(*args, **kwargs)

        # Set up the app <-> state back-reference for use by the
        # init_app_state patch
        try:
            from vllm_hi.common.state import VllmHiState

            hi = getattr(app.state, "vllm_hi", None)
            if hi is None:
                hi = VllmHiState()
                app.state.vllm_hi = hi
            hi.app = app
        except Exception:  # noqa: BLE001
            pass

        # Attach both A-side and B-side migration endpoints (unified A/B app).
        # Each endpoint reads state.vllm_hi at request time, so registering them
        # here — before init_app_state fully populates the state — is safe.
        try:
            from vllm_hi.instance.activate_endpoint import attach_active_endpoints
            from vllm_hi.instance.standby import attach_standby_endpoints

            attach_active_endpoints(app.state)
            attach_standby_endpoints(app)
        except Exception:  # noqa: BLE001
            pass

        # ---- 1. Custom status route ----
        @app.get("/vllm_hi/status")
        async def vllm_hi_status():
            hi = getattr(app.state, "vllm_hi", None)
            instance_type = getattr(hi, "instance_type", None)
            return {
                "active": True,
                "package": "vllm_hi",
                "role": (
                    "active" if instance_type == "A"
                    else "standby" if instance_type == "B"
                    else "unknown"
                ),
                "type": instance_type,
                "patches": sorted(_APPLIED),
                "message": "this vllm API server has been customized by vllm_hi",
            }

        # ---- 2. Pure ASGI middleware: attach the X-VLLM-HI response header ----
        # Use a pure ASGI middleware rather than BaseHTTPMiddleware to avoid
        # breaking SSE streaming responses.
        # Note: middleware_stack may already be built after build_app, so wrap
        # the existing stack directly instead of calling app.add_middleware.
        from starlette.datastructures import MutableHeaders

        class _VllmHiMiddleware:
            def __init__(self, inner_app):
                self.inner_app = inner_app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    await self.inner_app(scope, receive, send)
                    return

                async def send_with_header(message):
                    if message["type"] == "http.response.start":
                        headers = MutableHeaders(scope=message)
                        headers["X-VLLM-HI"] = "active"
                    await send(message)

                await self.inner_app(scope, receive, send_with_header)

        app.middleware_stack = _VllmHiMiddleware(app.middleware_stack)

        logger.info(
            "[vllm_hi] build_app customized: added route /vllm_hi/status "
            "and attached the X-VLLM-HI response header"
        )
        return app

    app_module.build_app = build_app_with_vllm_hi
    # If the relevant entry modules already bound the original name via
    # `from ..app import build_app`, replace their references too, so the patch
    # takes effect regardless of import order.
    for mod_name in (
        "vllm.entrypoints.launchers.api_server.entry",
        "vllm.entrypoints.launchers.render.entry",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "build_app", None) is original_build_app:
            mod.build_app = build_app_with_vllm_hi
    _mark(name)


def patch_chat_completion_logging() -> None:
    """Log each chat request on the server side (including source IP and key sampling parameters)."""
    name = "chat_completion_logging"
    if name in _APPLIED:
        return

    from vllm.entrypoints.openai.chat_completion import serving as serving_module

    cls = serving_module.OpenAIServingChat
    original = cls.create_chat_completion

    @functools.wraps(original)
    async def create_chat_completion_with_log(self, request, raw_request=None):
        client = "unknown"
        if raw_request is not None and raw_request.client is not None:
            client = raw_request.client.host
        logger.info(
            "[vllm_hi] received chat request model=%s max_tokens=%s temperature=%s "
            "stream=%s source=%s",
            request.model,
            request.max_tokens,
            request.temperature,
            request.stream,
            client,
        )
        # Also log the first 80 chars of the last user message, to confirm the
        # customization took effect
        if request.messages:
            last = request.messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                logger.info("[vllm_hi] last message: %s...", content[:80])
        return await original(self, request, raw_request=raw_request)

    cls.create_chat_completion = create_chat_completion_with_log
    _mark(name)


def patch_engine_internals() -> None:
    """Inject the RPC methods required by vllm_hi into EngineCore and Worker.

    See :mod:`vllm_hi.patches.engine_rpc` for the implementations; here we only
    inject them onto the classes under their RPC contract names (``_vllm_hi_*``).
    These names are referenced by string on the data plane via
    ``call_utility_async`` / ``collective_rpc_async`` and must not be changed.

    EngineCore:
    - _vllm_hi_get_scheduler_state: return a scheduler state snapshot
    - _vllm_hi_set_scheduler_state: restore scheduler state
    - _vllm_hi_apply_scheduler_incremental: apply incremental state
    Worker:
    - _vllm_hi_export_kv_ipc / _vllm_hi_set_kv_ipc: CUDA IPC direct KV export/restore
    - _vllm_hi_collect_kv_cache / _vllm_hi_try_get_result: background KV collection dispatch/query
    """
    name = "engine_internals"
    if name in _APPLIED:
        return

    from vllm.v1.engine.core import EngineCore
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_hi.common.rpc import (
        RPC_APPLY_SCHEDULER_INCREMENTAL,
        RPC_COLLECT_KV_CACHE,
        RPC_EXPORT_KV_IPC,
        RPC_GET_SCHEDULER_STATE,
        RPC_SET_KV_IPC,
        RPC_SET_SCHEDULER_STATE,
        RPC_TRY_GET_RESULT,
    )
    from vllm_hi.patches import engine_rpc

    setattr(EngineCore, RPC_GET_SCHEDULER_STATE, engine_rpc.engine_core_get_scheduler_state)
    setattr(EngineCore, RPC_SET_SCHEDULER_STATE, engine_rpc.engine_core_set_scheduler_state)
    setattr(EngineCore, RPC_APPLY_SCHEDULER_INCREMENTAL, engine_rpc.engine_core_apply_scheduler_incremental)

    setattr(Worker, RPC_EXPORT_KV_IPC, engine_rpc.worker_export_kv_ipc)
    setattr(Worker, RPC_SET_KV_IPC, engine_rpc.worker_set_kv_ipc)
    setattr(Worker, RPC_COLLECT_KV_CACHE, engine_rpc.worker_collect_kv_cache)
    setattr(Worker, RPC_TRY_GET_RESULT, engine_rpc.worker_try_get_result)

    _mark(name)
