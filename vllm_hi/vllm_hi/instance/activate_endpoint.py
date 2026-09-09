# SPDX-License-Identifier: Apache-2.0
"""Migration-related HTTP endpoints for the A instance.

    POST /vllm_hi/migration/collect       - state collection (full/incremental)
    POST /vllm_hi/migration/pause         - pause generation
    POST /vllm_hi/migration/resume        - resume generation
    POST /vllm_hi/migration/destroy       - destroy the instance
    POST /vllm_hi/migration/convert_to_b  - convert to type B

B instance endpoints are defined by ``_build_standby_app`` in
``instance.standby``, not in this module. The data-plane TCP endpoints (on both
A and B) listen on data_port and receive/send state data.
"""

from __future__ import annotations

from fastapi import FastAPI

from vllm_hi.common.protocol import (
    CollectStateRequest,
    CollectStateResponse,
    PauseResponse,
    SimpleResponse,
)
from vllm_hi.common.utils import parse_addr, topology_dict_from_vllm_config
from vllm_hi.migration.backends import get_backend
from vllm_hi.migration.backends.base import TransferContext
from vllm_hi.migration.collector import (
    StateCollector,
    _KV_COLLECT_CHUNK_BLOCKS,
    _KV_COLLECT_CHUNK_BLOCKS_LARGE,
)
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.instance.activate_endpoint")


def attach_active_endpoints(state) -> None:
    """Register the A instance's migration endpoint routes on the app (resolved
    from state).

    Called from patch_init_app_state. The app is provided via the
    ``state.vllm_hi.app`` back-reference set up by the build_app patch.
    """
    hi = getattr(state, "vllm_hi", None)
    app = getattr(hi, "app", None) if hi is not None else None
    if app is None:
        logger.warning("[vllm_hi] app not found, skipping migration endpoint registration")
        return
    _attach_active_endpoints(app, state)


def _attach_active_endpoints(app: FastAPI, state) -> None:
    """Attach the A instance's migration endpoints."""

    @app.post("/vllm_hi/migration/collect")
    async def collect(request: CollectStateRequest):
        hi = state.vllm_hi
        engine_client = hi.engine_client
        if engine_client is None:
            return CollectStateResponse(
                success=False, error="engine_client unavailable"
            ).model_dump()

        collector = hi.collector
        if collector is None:
            collector = StateCollector(engine_client)
            hi.collector = collector

        try:
            dst_ip, dst_port = parse_addr(request.dst_addr)
            backend = get_backend(request.transfer_backend)
            kv_mode = "ipc" if backend.gpu_direct else "cpu"
            ctx = TransferContext(
                state=state,
                engine_client=engine_client,
                collector=collector,
            )

            if request.mode == "full":
                vllm_config = getattr(engine_client, "vllm_config", None)
                src_topo = (
                    topology_dict_from_vllm_config(vllm_config)
                    if vllm_config is not None else {}
                )
                snapshot = await collector.collect_full(src_topo, kv_mode=kv_mode)
                hi.last_snapshot_id = snapshot.snapshot_id
                meta = {"type": "full", "snapshot": snapshot}
                kv_cache = snapshot.kv_cache
                snapshot_id = snapshot.snapshot_id
                block_count = len(snapshot.kv_cache_blocks)
            else:
                # The final increment (A is paused, no inference needs to stay
                # smooth) collects in one large chunk, eliminating multiple RPC
                # round-trips and cross-chunk aggregation overhead to shorten
                # downtime.
                paused = hi.paused
                chunk_size = (
                    _KV_COLLECT_CHUNK_BLOCKS_LARGE if paused
                    else _KV_COLLECT_CHUNK_BLOCKS
                )
                update = await collector.collect_incremental(
                    chunk_size=chunk_size,
                    boost_cpu=paused,
                    kv_mode=kv_mode,
                )
                meta = {"type": "incremental", "update": update}
                kv_cache = update.kv_cache_delta
                snapshot_id = update.update_id
                block_count = len(update.kv_cache_blocks)

            data_size_bytes = await backend.send_state(
                ctx, meta, kv_cache, (dst_ip, dst_port)
            )
            return CollectStateResponse(
                success=True, snapshot_id=snapshot_id,
                data_size_bytes=data_size_bytes,
                block_count=block_count,
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[vllm_hi] state collection failed")
            return CollectStateResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()
        finally:
            # Release shared memory left over from the collection phase on both
            # success and failure to avoid leaks.
            collector.release_kv_cache_shm()

    @app.post("/vllm_hi/migration/pause")
    async def pause():
        hi = state.vllm_hi
        engine_client = hi.engine_client
        if engine_client is None:
            return PauseResponse(success=False, error="engine_client unavailable").model_dump()
        try:
            # Use keep mode: pause all generation, don't clear cache, don't wait for drain
            await engine_client.pause_generation(mode="keep", clear_cache=False)
            hi.paused = True
            logger.info("[vllm_hi] A instance paused generation")
            return PauseResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[vllm_hi] pause failed")
            return PauseResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.post("/vllm_hi/migration/resume")
    async def resume():
        hi = state.vllm_hi
        engine_client = hi.engine_client
        if engine_client is None:
            return PauseResponse(success=False, error="engine_client unavailable").model_dump()
        try:
            await engine_client.resume_generation()
            hi.paused = False
            logger.info("[vllm_hi] A instance resumed generation")
            return PauseResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            return PauseResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.post("/vllm_hi/migration/destroy")
    async def destroy():
        hi = state.vllm_hi
        engine_client = hi.engine_client
        try:
            if engine_client is not None:
                engine_client.shutdown()
            hi.engine_client = None
            logger.info("[vllm_hi] A instance destroyed")
            return SimpleResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            return SimpleResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    @app.post("/vllm_hi/migration/convert_to_b")
    async def convert_to_b():
        hi = state.vllm_hi
        engine_client = hi.engine_client
        try:
            if engine_client is not None:
                # Pause then unload (keep the communication layer)
                await engine_client.pause_generation(mode="keep", clear_cache=False)
                engine_client.shutdown()
            hi.engine_client = None
            hi.instance_type = "B"
            hi.paused = False
            logger.info("[vllm_hi] A instance converted to type B (standby)")
            return SimpleResponse(success=True).model_dump()
        except Exception as exc:  # noqa: BLE001
            return SimpleResponse(
                success=False, error=f"{type(exc).__name__}: {exc}"
            ).model_dump()

    logger.info("[vllm_hi] A instance migration endpoints registered")
