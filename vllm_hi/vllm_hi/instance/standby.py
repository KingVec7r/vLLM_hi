# SPDX-License-Identifier: Apache-2.0
"""StandbyInstance: standby startup and on-demand instantiation of the B instance.

When a B instance starts:
- Only initialize FastAPI (migration endpoints + HostClient)
- Do not instantiate the inference backend
- Register with the Host (type=B)
- Enter standby and wait for a migration command

After receiving a migration command:
- Instantiate the vLLM inference backend with the target topology/model
- Receive state data, perform remapping and recovery
- Convert to an A instance
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_hi.common.protocol import (
    ActivateRequest,
    ActivateResponse,
    InstantiateRequest,
    InstantiateResponse,
    ReceiveStateRequest,
    ReceiveStateResponse,
)
from vllm_hi.common.state import VllmHiState
from vllm_hi.common.utils import (
    resolve_register_ip,
    topology_from_args,
    visible_gpu_ids_from_env,
)
from vllm_hi.instance.host_client import HostClient
from vllm_hi.migration.receiver import StateReceiver
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.instance.standby")


async def run_standby_server(args, **uvicorn_kwargs) -> None:
    """Standby startup of the B instance.

    Skip build_async_engine_client; only start FastAPI (migration endpoints +
    HostClient).
    """
    from vllm.entrypoints.launchers.launcher import serve_http, setup_server
    from vllm.entrypoints.launchers.utils.server_utils import (
        get_uvicorn_log_config,
    )
    import vllm.envs as envs

    try:
        from vllm.utils.system_utils import decorate_logs
        decorate_logs("APIServer", skip_if_decorated=True)
    except Exception:  # noqa: BLE001
        pass

    # During same-host migration, B's data plane (receiving + writing shm)
    # shares CPU/memory bandwidth with A's inference. Bind the whole B-side
    # APIServer process to the NUMA node of B's GPU so the socket recv buffer,
    # tmpfs page faults, and shm memcpy all land on the B-side node, spatially
    # isolated from A's inference (on the other NUMA node), eliminating
    # cross-NUMA memory contention.
    try:
        from vllm_hi.common.utils import get_device_cpu_affinity

        _b_cpu_affinity = get_device_cpu_affinity()
        if _b_cpu_affinity:
            os.sched_setaffinity(0, _b_cpu_affinity)
            logger.info(
                "[vllm_hi] B side bound process CPU affinity to %d CPUs",
                len(_b_cpu_affinity),
            )
    except Exception:  # noqa: BLE001
        logger.warning("[vllm_hi] B side CPU affinity binding failed (ignoring)")

    import signal

    def _interrupt_init(*_):
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)

    listen_address, sock = setup_server(args, reuse_port=False)

    app = _build_standby_app(args)

    # The data-plane server is started lazily on the first receive request,
    # using the selected backend. The backend is auto-selected by the migration
    # coordinator based on whether A and B are on the same host and delivered
    # via ReceiveStateRequest.

    host_addr = getattr(args, "vllm_hi_host", None)
    if host_addr:
        asyncio.create_task(_register_b_to_host(app.state, args, host_addr))

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    logger.info("[vllm_hi] B instance standby started at %s", listen_address)

    shutdown_task = await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        ssl_ciphers=args.ssl_ciphers,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
        **uvicorn_kwargs,
    )
    try:
        await shutdown_task
    finally:
        data_server_task = app.state.vllm_hi.data_server_task
        if data_server_task is not None:
            data_server_task.cancel()
        sock.close()


class _StandbyVllmConfig:
    """Placeholder vllm_config during standby (only shutdown_timeout is accessed on shutdown)."""

    shutdown_timeout = 0


class _StandbyEngineClient:
    """Placeholder engine_client during B standby, satisfying serve_http's
    watchdog/shutdown interface.

    Replaced by the real AsyncLLM after instantiation (app.state.engine_client
    = real_engine).
    """

    errored = False
    is_running = True

    @property
    def vllm_config(self):
        return _StandbyVllmConfig()

    async def shutdown(self, timeout: float = 0.0) -> None:
        logger.info("[vllm_hi] StandbyEngineClient shutdown (no-op)")


async def _init_serving_state(app: FastAPI, state, engine) -> None:
    """Attach the full vLLM serving state and routes so that B can serve new
    requests after activation.

    During standby, B only has migration endpoints (no /v1/chat/completions).
    After instantiation, call vLLM's native ``init_app_state`` to initialize
    serving state (openai_serving_chat etc.) and register inference routes +
    the exception handler, giving B the same inference capability as A. The
    migration endpoints are unaffected.

    Don't use ``register_api_routers``: it also registers the metrics
    instrumentator, which calls ``app.add_middleware`` and raises
    ``RuntimeError: Cannot add middleware after an application has started``
    on an already-started app. Register only the inference routes
    (models/chat/completions) here and skip metrics/lora/profile.
    """
    from vllm.entrypoints.generate.api_router import register_generate_api_routers
    from vllm.entrypoints.launchers.api_server import app_state as app_state_module
    from vllm.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )
    from vllm.entrypoints.serve.exception_handling.register import (
        init_exception_handler,
    )

    # Use the original (not vllm_hi-wrapped) init_app_state: here we only want
    # to attach vLLM's native serving state and must not inject vllm_hi state
    # again (standby manages state.vllm_hi itself; re-injecting would overwrite
    # the already-established receiver / resumed_queues etc.).
    init_app_state = getattr(
        app_state_module, "_vllm_hi_original_init_app_state",
        app_state_module.init_app_state,
    )

    supported_tasks = await engine.get_supported_tasks()
    await init_app_state(engine, state, state.args, supported_tasks)
    register_models_api_router(app)
    if "generate" in supported_tasks:
        register_generate_api_routers(app)
    init_exception_handler(app)
    logger.info(
        "[vllm_hi] B instance attached vLLM serving routes (tasks=%s)",
        sorted(supported_tasks),
    )


def _build_standby_app(args) -> FastAPI:
    """Build the B instance's FastAPI app (migration endpoints only)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("[vllm_hi] B instance FastAPI started")
        try:
            yield
        finally:
            host_client = app.state.vllm_hi.host_client
            if host_client is not None:
                await host_client.close()
            logger.info("[vllm_hi] B instance FastAPI shut down")

    app = FastAPI(title="vllm_hi StandbyInstance", lifespan=lifespan)
    app.state.args = args
    host_addr = getattr(args, "vllm_hi_host", None)
    app.state.vllm_hi = VllmHiState(
        app=app,
        instance_type="B",
        host_addr=host_addr,
        data_port=getattr(args, "vllm_hi_data_port", 0) or 0,
        engine_client=None,
        host_client=HostClient(host_addr) if host_addr else None,
        registered=False,
        paused=False,
    )
    # serve_http needs app.state.engine_client (for watchdog and shutdown).
    # During B standby use a placeholder; replace it with the real engine_client
    # after instantiation.
    app.state.engine_client = _StandbyEngineClient()

    @app.get("/vllm_hi/status")
    async def status():
        return {
            "active": True,
            "package": "vllm_hi",
            "role": "standby",
            "type": "B",
            "instantiated": app.state.vllm_hi.engine_client is not None,
        }

    @app.post("/vllm_hi/migration/instantiate")
    async def instantiate(request: InstantiateRequest):
        try:
            engine = await instantiate_engine(request, app.state)
            app.state.vllm_hi.engine_client = engine
            app.state.vllm_hi.receiver = StateReceiver(engine)
            # Attach the full serving state and routes so B can serve new
            # requests like A after activation (not just resume in-flight
            # streams).
            await _init_serving_state(app, app.state, engine)
            # Pause immediately after instantiation: keep it frozen while
            # restoring the scheduler state, avoiding duplicate generation with
            # A during incremental migration; resume only on activate.
            await engine.pause_generation(mode="keep", clear_cache=False)
            app.state.vllm_hi.resumed_queues = {}
            logger.info("[vllm_hi] B instance instantiated the inference backend (paused, on standby)")
            return InstantiateResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[vllm_hi] B instantiation failed")
            return InstantiateResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.post("/vllm_hi/migration/receive")
    async def receive(request: ReceiveStateRequest):
        receiver = app.state.vllm_hi.receiver
        if receiver is None:
            return ReceiveStateResponse(
                success=False, error="receiver unavailable (not instantiated?)"
            ).model_dump()
        try:
            src_topo = request.src_topology.model_dump()
            engine = app.state.vllm_hi.engine_client
            dst_topo = {}
            if engine is not None and hasattr(engine, "vllm_config"):
                dst_topo = {
                    "tensor_parallel_size": engine.vllm_config.parallel_config.tensor_parallel_size,
                    "pipeline_parallel_size": engine.vllm_config.parallel_config.pipeline_parallel_size,
                }
            receiver.setup_remap(src_topo, dst_topo)
            # Lazily start the data-plane receive service with the selected
            # backend (started once on the first receive)
            await _ensure_data_server(app.state, request.transfer_backend)
            return ReceiveStateResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            return ReceiveStateResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.post("/vllm_hi/migration/activate")
    async def activate(request: ActivateRequest):
        try:
            hi = app.state.vllm_hi
            hi.instance_type = "A"
            hi.paused = False
            engine = hi.engine_client
            if engine is not None:
                # Wait for all full/incremental state reconstruction on the data
                # plane to finish; otherwise register_resumed_outputs may get an
                # empty snapshot (zero resumed requests, no continuation after
                # migration).
                data_idle = hi.data_idle
                if data_idle is not None:
                    await data_idle.wait()
                # Register output processors for the resumed requests, then
                # resume to start serving
                receiver = hi.receiver
                request_states = {}
                snapshot = getattr(receiver, "_current_snapshot", None)
                if snapshot is not None:
                    request_states = snapshot.request_states
                from vllm_hi.migration.resume import register_resumed_outputs

                queues = register_resumed_outputs(engine, request_states)
                hi.resumed_queues = queues
                try:
                    await engine.resume_generation()
                except Exception:  # noqa: BLE001
                    pass
            logger.info("[vllm_hi] B instance converted to type A, starting to serve")
            return ActivateResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            return ActivateResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.get("/vllm_hi/migration/resume_stream")
    async def resume_stream(request_id: str):
        """Provide a resume SSE stream for a single in-flight request (continue
        emitting new tokens from the state inherited by B)."""
        from vllm.outputs import STREAM_FINISHED

        queues = app.state.vllm_hi.resumed_queues or {}
        queue = queues.get(request_id)
        if queue is None:
            return JSONResponse(
                status_code=404, content={"error": "request_not_found"}
            )

        async def generate():
            try:
                while True:
                    try:
                        out = queue.get_nowait()
                        if out is None:
                            out = await queue.get()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[vllm_hi] resume stream %s failed to read output: %s",
                            request_id, exc,
                        )
                        break
                    if out is STREAM_FINISHED:
                        break
                    finished = bool(getattr(out, "finished", False))
                    for o in getattr(out, "outputs", []) or []:
                        text = getattr(o, "text", "") or ""
                        if not text:
                            continue
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "choices": [{
                                "index": getattr(o, "index", 0),
                                "delta": {"content": text},
                                "finish_reason": getattr(o, "finish_reason", None),
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    if finished:
                        break
            finally:
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "ok", "type": "B"}

    return app


def _clamp_util_to_free_gpu_memory(requested_util: float) -> float:
    """Clamp gpu_memory_utilization to the actual free GPU memory to avoid OOM
    during B-side instantiation.

    When the host GPU is partially occupied by external processes and free
    memory is insufficient, vLLM instantiation fails outright with
    ``Free memory ... is less than desired``. Compute the usable util from the
    smallest free ratio across visible GPUs and always reserve a margin,
    automatically lowering the target utilization (migration correctness only
    requires B's KV capacity >= the migrated data volume, so lowering util does
    not affect the migrated data itself).

    The reserved margin must cover transient allocations during vLLM
    instantiation such as ``compile_or_warm_up_model`` (CUDA graph capture) and
    activation buffers: in practice, at util=0.97 the GPU is nearly full, with
    only ~10 MiB left during warmup, causing intermittent ``CUDA out of memory``
    failures (migration hangs at b_instantiate). So the margin must be large
    enough (8% ~ 6GB), not just 1-2%.
    """
    import torch

    try:
        visible = [
            int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if x.strip() != ""
        ]
        if not visible:
            visible = [0]
        min_free_ratio = 1.0
        # `visible` holds physical GPU numbers, but torch.cuda.mem_get_info
        # expects the device index within the current process's
        # CUDA_VISIBLE_DEVICES (0..N-1). Querying by physical number goes out
        # of bounds for non-contiguous numbers (e.g. the default B side
        # "4,5,6,7"), gets swallowed by the except below, and silently returns
        # the original util, so util clamping never takes effect under the
        # default config. Iterate by index over all devices visible to the
        # current process instead.
        for i in range(len(visible)):
            free, total = torch.cuda.mem_get_info(i)
            if total <= 0:
                continue
            min_free_ratio = min(min_free_ratio, free / total)
        # Always reserve an 8% margin for CUDA graph warmup + activation
        # buffers (see docstring).
        margin = 0.08
        max_util = min_free_ratio - margin
        if max_util < requested_util:
            clamped = max(max_util, 0.1)
            logger.warning(
                "[vllm_hi] B instantiation util lowered from %.3f to %.3f "
                "(GPU min free ratio %.3f)",
                requested_util, clamped, min_free_ratio,
            )
            return clamped
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[vllm_hi] probing GPU free memory failed, keeping util=%.3f: %s",
            requested_util, exc,
        )
    return requested_util


async def instantiate_engine(request: InstantiateRequest, state) -> Any:
    """Instantiate the vLLM inference backend with the target topology/model."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    orig_args = state.args
    model = request.target_model
    served_model_name = request.served_model_name
    target_util = _clamp_util_to_free_gpu_memory(
        request.target_gpu_memory_utilization
    )

    engine_args = AsyncEngineArgs(
        model=model,
        served_model_name=[served_model_name],
        tensor_parallel_size=request.target_tp,
        pipeline_parallel_size=request.target_pp,
        gpu_memory_utilization=target_util,
        max_model_len=getattr(orig_args, "max_model_len", 4096),
        disable_custom_all_reduce=getattr(
            orig_args, "disable_custom_all_reduce", True
        ),
        enable_log_requests=False,
    )
    for k, v in request.extra_args.items():
        if hasattr(engine_args, k):
            setattr(engine_args, k, v)

    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    async_llm = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        enable_log_requests=False,
        aggregate_engine_logging=False,
        disable_log_stats=True,
    )
    await async_llm.reset_mm_cache()

    logger.info(
        "[vllm_hi] B instantiation complete: model=%s tp=%d pp=%d util=%.3f",
        model, request.target_tp, request.target_pp,
        target_util,
    )
    return async_llm


async def _register_b_to_host(state, args, host_addr) -> None:
    """Register the B instance with the Host."""
    from vllm_hi.common.types import InstanceType

    await asyncio.sleep(1.0)

    hi = state.vllm_hi
    host_client = hi.host_client
    if host_client is None:
        logger.warning("[vllm_hi] B registration: host_client unavailable")
        return

    register_ip = resolve_register_ip(getattr(args, "host", "0.0.0.0"))
    port = getattr(args, "port", 8001)

    served_model_name = f"standby:{register_ip}:{port}"

    topology = topology_from_args(args, visible_gpu_ids_from_env())

    result = await host_client.register(
        ip=register_ip,
        port=port,
        instance_type=InstanceType.B,
        served_model_name=served_model_name,
        inference_params={},
        topology=topology,
        data_port=hi.data_port,
    )
    if result.success:
        hi.registered = True
        hi.vllm_hi_id = result.vllm_hi_id
        logger.info("[vllm_hi] B instance registered with Host: id=%s", result.vllm_hi_id)
    else:
        logger.error("[vllm_hi] B instance registration failed: %s", result.error)


async def _ensure_data_server(state, transfer_backend: str | None) -> None:
    """Lazily start the data-plane receive service with the selected backend
    (idempotent, started once on the first receive).

    The backend is auto-selected by the migration coordinator based on whether
    A and B are on the same host and delivered via ``ReceiveStateRequest``; the
    data-plane server receives KV chunks from A and restores them (TCP via shm,
    IPC via direct GPU transfer).
    """
    hi = state.vllm_hi
    if hi.data_server_task is not None:
        return
    data_port = hi.data_port
    if data_port <= 0:
        return

    from vllm_hi.migration.backends import get_backend
    from vllm_hi.migration.backends.base import TransferContext

    backend = get_backend(transfer_backend)
    hi.transfer_backend = backend.name
    ctx = TransferContext(
        state=state,
        engine_client=hi.engine_client,
        receiver=hi.receiver,
    )
    task = asyncio.create_task(
        backend.start_data_server(ctx, ("0.0.0.0", data_port))
    )
    hi.data_server_task = task
    # Yield the event loop to ensure the server task runs through socket
    # bind/listen before returning, so the port is already listening when A
    # connects immediately afterwards.
    await asyncio.sleep(0)
    logger.info(
        "[vllm_hi] B data-plane backend %s server started on port %d",
        backend.name, data_port,
    )
