# SPDX-License-Identifier: Apache-2.0
"""Request router: routes by served_model_name to the matching A instance and
passes the response through (including streaming SSE).

Supports live-migration switchover: after migration completes, the Host
seamlessly reconnects in-flight streaming requests to B's resume endpoint; the
client connection stays open and only sees a transient throughput drop.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm_hi.coordinator.registry import InstanceRegistry
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.coordinator.router")


class InflightStream:
    """Client-connection state of one in-flight streaming request."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.a_done = asyncio.Event()
        self.switched = False
        self.finished = False


class RequestRouter:
    """Request router, proxying HTTP requests via httpx.AsyncClient."""

    def __init__(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient | None = None,
        inflight: dict[str, InflightStream] | None = None,
    ) -> None:
        self.registry = registry
        self._external_client = http_client is not None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
            follow_redirects=False,
        )
        self._inflight: dict[str, InflightStream] = inflight if inflight is not None else {}

    async def close(self) -> None:
        if not self._external_client:
            await self._http_client.aclose()

    async def proxy_request(
        self,
        request: Request,
        path: str,
    ) -> Response:
        """Generic request routing passthrough.

        Args:
            request: FastAPI Request object
            path: Target path (e.g. /v1/chat/completions)

        Returns:
            Response (StreamingResponse or JSONResponse)
        """
        body = await request.body()
        try:
            payload: dict[str, Any] = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_request", "message": "request body is not valid JSON"},
            )

        model = payload.get("model")
        if not model:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_model",
                    "message": "request is missing the model field",
                },
            )

        info = await self.registry.get_by_model(model)
        if info is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "model_not_found",
                    "message": f"no instance found for served_model_name={model!r}",
                },
            )

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }

        is_stream = bool(payload.get("stream", False))

        # Inject a stable request_id so the migration switchover can reconnect
        # the resume stream by request_id. The vLLM serving layer converts it
        # to ``chatcmpl-{request_id}`` as the internal ID (with
        # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1).
        raw_rid = payload.get("request_id")
        if not raw_rid:
            raw_rid = uuid.uuid4().hex
            payload["request_id"] = raw_rid
            body = json.dumps(payload).encode("utf-8")
        internal_rid = f"chatcmpl-{raw_rid}"

        if is_stream:
            return await self._proxy_stream(
                request.method, model, path, headers, body, internal_rid
            )
        else:
            return await self._proxy_once(
                request.method, model, path, headers, body
            )

    async def _resolve_target(self, model: str, path: str) -> str | None:
        """Resolve the target instance URL for the current model (may change from A to B after switchover)."""
        info = await self.registry.get_by_model(model)
        if info is None:
            return None
        return f"{info.base_url}{path}"

    async def _send(self, method, model, path, headers, body):
        """Send the request; if the connection fails because A was destroyed
        during switchover, re-resolve the route and retry once."""
        url = await self._resolve_target(model, path)
        if url is None:
            raise KeyError(f"model {model!r} not found")
        req = self._http_client.build_request(method, url, headers=headers, content=body)
        try:
            return await self._http_client.send(req, stream=True)
        except httpx.TransportError:
            # Transport errors (connection reset/refused) are transient during
            # the switchover window: A is destroyed or B has just taken over and
            # is not stable yet. Re-resolve the route (it may have changed from A
            # to B) and retry once.
            new_url = await self._resolve_target(model, path)
            if new_url is None:
                raise
            req = self._http_client.build_request(
                method, new_url, headers=headers, content=body
            )
            return await self._http_client.send(req, stream=True)

    async def _proxy_stream(
        self,
        method: str,
        model: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        request_id: str,
    ) -> StreamingResponse:
        """Streaming passthrough: first relay A's SSE, then reconnect to B's
        resume stream after migration switchover."""
        entry = InflightStream(request_id)
        self._inflight[request_id] = entry

        resp = await self._send(method, model, path, headers, body)

        async def pump_a():
            try:
                async for chunk in resp.aiter_raw():
                    await entry.queue.put(chunk)
            except Exception:  # noqa: BLE001
                # A's stream is destroyed/disconnected after switchover; ignore
                # it and let B's resume stream take over.
                pass
            finally:
                try:
                    await resp.aclose()
                except Exception:  # noqa: BLE001
                    pass
                entry.a_done.set()
                if entry.switched:
                    return
                entry.finished = True
                await entry.queue.put(None)
                # The stream fully ended; clean up the in-flight registration to
                # avoid leaks accumulating in a long-running Host.
                self._inflight.pop(request_id, None)

        asyncio.create_task(pump_a())

        async def generate():
            while True:
                chunk = await entry.queue.get()
                if chunk is None:
                    break
                yield chunk

        return StreamingResponse(
            generate(),
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding")
            },
            media_type=resp.headers.get("content-type"),
        )

    async def switch_inflight(self, dst_url: str) -> None:
        """After migration switchover, reconnect in-flight streaming requests to B's resume endpoint."""
        for request_id, entry in list(self._inflight.items()):
            if entry.finished or entry.switched:
                continue
            entry.switched = True
            asyncio.create_task(self._pump_b(request_id, entry, dst_url))

    async def _pump_b(
        self, request_id: str, entry: InflightStream, dst_url: str
    ) -> None:
        url = f"{dst_url}/vllm_hi/migration/resume_stream"
        try:
            req = self._http_client.build_request(
                "GET", url, params={"request_id": request_id}
            )
            resp = await self._http_client.send(req, stream=True)
            async for chunk in resp.aiter_raw():
                await entry.queue.put(chunk)
            await resp.aclose()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[vllm_hi Router] resume stream %s failed: %s", request_id, exc
            )
        finally:
            entry.finished = True
            await entry.queue.put(None)
            self._inflight.pop(request_id, None)

    async def _proxy_once(
        self,
        method: str,
        model: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> Response:
        """Non-streaming passthrough: return JSON in one shot."""
        resp = await self._send(method, model, path, headers, body)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers={
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding")
            },
            media_type=resp.headers.get("content-type"),
        )
