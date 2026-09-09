# SPDX-License-Identifier: Apache-2.0
"""Host scheduler (SchedulerServer): FastAPI app providing instance registration, request routing, and migration scheduling."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm_hi.common.protocol import (
    MigrationCommand,
    MigrationResult,
    RegisterInstanceRequest,
    RegisterInstanceResponse,
)
from vllm_hi.common.types import MigrationStep
from vllm_hi.coordinator.registry import InstanceRegistry
from vllm_hi.coordinator.router import RequestRouter
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.coordinator")


class SchedulerServer:
    """Host scheduler, serving requests via FastAPI.

    Endpoints:
        POST /vllm_hi/register           - instance registration
        GET  /vllm_hi/instances          - instance list
        POST /vllm_hi/migrate            - live-migration command
        GET  /vllm_hi/migration/{id}/status - migration status query
        POST /v1/chat/completions        - inference request routing (passthrough)
        POST /v1/completions             - inference request routing (passthrough)
        GET  /vllm_hi/status             - Host status
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8002) -> None:
        self.host = host
        self.port = port
        self.registry = InstanceRegistry()
        self._http_client: httpx.AsyncClient | None = None
        self.router: RequestRouter | None = None
        self._migration_results: dict[str, MigrationResult] = {}
        self._migration_tasks: dict[str, asyncio.Task] = {}
        self._app: FastAPI | None = None
        self._inflight: dict[str, Any] = {}

    @property
    def app(self) -> FastAPI:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Connection pool: supports 4096 concurrent streaming requests
            # (each streaming request holds one connection). Raise the cap to
            # 8192 to avoid httpx.PoolTimeout when concurrency exceeds the pool.
            limits = httpx.Limits(
                max_connections=8192,
                max_keepalive_connections=2048,
                keepalive_expiry=120.0,
            )
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0, read=None, write=30.0, pool=120.0
                ),
                limits=limits,
                follow_redirects=False,
            )
            self.router = RequestRouter(
                self.registry, self._http_client, self._inflight
            )
            logger.info(
                "[vllm_hi Host] scheduler started at %s:%d", self.host, self.port
            )
            try:
                yield
            finally:
                # Cancel all in-progress migration tasks
                for task in self._migration_tasks.values():
                    task.cancel()
                await self.router.close()
                await self._http_client.aclose()

        app = FastAPI(
            title="vllm_hi SchedulerServer",
            lifespan=lifespan,
        )

        # --- Status endpoints ---
        @app.get("/vllm_hi/status")
        async def status():
            return {
                "active": True,
                "package": "vllm_hi",
                "role": "scheduler",
                "instances": await self.registry.count(),
            }

        # --- Registration ---
        @app.post("/vllm_hi/register")
        async def register(request: RegisterInstanceRequest):
            response = await self.registry.register(request)
            status_code = 200 if response.success else 409
            return JSONResponse(
                status_code=status_code, content=response.model_dump()
            )

        # --- Instance list ---
        @app.get("/vllm_hi/instances")
        async def list_instances():
            items = await self.registry.list_all_models()
            return [item.model_dump() for item in items]

        # --- Migration ---
        @app.post("/vllm_hi/migrate")
        async def migrate(command: MigrationCommand):
            from vllm_hi.coordinator.migration import MigrationCoordinator

            coordinator = MigrationCoordinator(
                self.registry, self._http_client
            )
            # Run the migration asynchronously, returning migration_id immediately
            result = coordinator.create_pending_result(command)
            # Store it in _migration_results right away so the status endpoint can query it
            self._migration_results[result.migration_id] = result
            task = asyncio.create_task(
                self._run_migration(coordinator, command, result.migration_id)
            )
            self._migration_tasks[result.migration_id] = task
            return result.model_dump()

        @app.get("/vllm_hi/migration/{migration_id}/status")
        async def migration_status(migration_id: str):
            result = self._migration_results.get(migration_id)
            if result is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": "migration_not_found"},
                )
            return result.model_dump()

        # --- Inference request routing ---
        @app.post("/v1/chat/completions")
        async def proxy_chat(request: Request):
            return await self.router.proxy_request(request, "/v1/chat/completions")

        @app.post("/v1/completions")
        async def proxy_completion(request: Request):
            return await self.router.proxy_request(request, "/v1/completions")

        # OpenAI-compatible models endpoint (returns the list of registered models)
        @app.get("/v1/models")
        async def list_models():
            instances = await self.registry.list_all()
            seen = set()
            data = []
            for info in instances:
                if info.instance_type.value == "A" and info.served_model_name not in seen:
                    seen.add(info.served_model_name)
                    data.append({
                        "id": info.served_model_name,
                        "object": "model",
                        "created": int(info.registered_at),
                        "owned_by": "vllm_hi",
                    })
            return {"object": "list", "data": data}

        return app

    async def _run_migration(
        self,
        coordinator: Any,
        command: MigrationCommand,
        migration_id: str,
    ) -> None:
        """Run the migration task and save its result."""

        async def _progress_callback(step: str, **extra: Any) -> None:
            # Update current_step and step_history in real time so the status
            # endpoint can query progress
            import time as _time
            stored = self._migration_results.get(migration_id)
            if stored is not None:
                stored.current_step = step
                entry = {"step": step, "timestamp": _time.time()}
                entry.update(extra)
                stored.step_history.append(entry)
            # After service switchover completes, reconnect in-flight streaming
            # requests to the new A instance (B)
            if step == MigrationStep.SWITCH_SERVICE_DONE.value and self.router is not None:
                try:
                    dst = await self.registry.get_by_id(command.dst_vllm_hi_id)
                    if dst is not None:
                        await self.router.switch_inflight(dst.base_url)
                        logger.info(
                            "[vllm_hi Host] in-flight streams reconnected to %s", dst.base_url
                        )
                except Exception:  # noqa: BLE001
                    logger.exception("[vllm_hi Host] failed to reconnect in-flight streams")

        try:
            result = await coordinator.execute_migration(
                command, migration_id,
                progress_callback=_progress_callback,
            )
            # Backfill the final fields into the original pending result,
            # preserving the accumulated step_history and current_step
            stored = self._migration_results.get(migration_id)
            if stored is not None:
                stored.status = result.status
                stored.end_time = result.end_time
                stored.duration = result.duration
                stored.new_active_instance = result.new_active_instance
                stored.error = result.error
            else:
                self._migration_results[migration_id] = result
        except Exception as exc:  # noqa: BLE001
            logger.exception("[vllm_hi Host] migration task error")
            from vllm_hi.common.types import MigrationStatus

            stored = self._migration_results.get(migration_id)
            if stored is not None:
                stored.status = MigrationStatus.FAILED
                stored.error = f"{type(exc).__name__}: {exc}"
            else:
                from vllm_hi.common.utils import now_iso8601

                self._migration_results[migration_id] = MigrationResult(
                    migration_id=migration_id,
                    status=MigrationStatus.FAILED,
                    start_time=now_iso8601(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        finally:
            self._migration_tasks.pop(migration_id, None)

    async def start(self) -> None:
        """Start the uvicorn server."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    def serve_sync(self) -> None:
        """Synchronous startup (for script entrypoints)."""
        import uvloop

        uvloop.run(self.start())
