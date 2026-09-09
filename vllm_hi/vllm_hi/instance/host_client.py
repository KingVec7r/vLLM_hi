# SPDX-License-Identifier: Apache-2.0
"""HostClient: client for the vLLM instance to communicate with the Host scheduler."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from vllm_hi.common.types import InstanceInfo, InstanceType, Topology
from vllm_hi.common.utils import parse_addr
from vllm_hi.common.protocol import (
    RegisterInstanceRequest,
    RegisterInstanceResponse,
)
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.instance.host_client")


class HostClient:
    """Client for communicating with the Host scheduler.

    Used to:
    - Register the instance with the Host after startup
    - Query the instance list
    - Interact with the Host during migration
    """

    def __init__(
        self,
        host_addr: str,
        timeout: float = 30.0,
        retry: int = 3,
    ) -> None:
        self.host_addr = host_addr
        self.base_url = f"http://{host_addr}"
        self.timeout = timeout
        self.retry = retry
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def register(
        self,
        ip: str,
        port: int,
        instance_type: InstanceType,
        served_model_name: str,
        inference_params: dict[str, Any],
        topology: Topology,
        data_port: int = 0,
    ) -> RegisterInstanceResponse:
        """Register the instance with the Host.

        Retry up to `retry` times on failure.
        """
        request = RegisterInstanceRequest(
            ip=ip,
            port=port,
            instance_type=instance_type,
            served_model_name=served_model_name,
            inference_params=inference_params,
            topology=topology,
            data_port=data_port,
        )
        url = f"{self.base_url}/vllm_hi/register"
        last_error: str | None = None
        for attempt in range(1, self.retry + 1):
            try:
                resp = await self.client.post(url, json=request.model_dump())
                data = resp.json()
                result = RegisterInstanceResponse(**data)
                if result.success:
                    logger.info(
                        "[vllm_hi HostClient] registration succeeded: type=%s model=%s id=%s",
                        instance_type.value, served_model_name, result.vllm_hi_id,
                    )
                    return result
                # Registration was rejected (e.g. served_model_name conflict); don't retry
                logger.warning(
                    "[vllm_hi HostClient] registration rejected: %s", result.error
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "[vllm_hi HostClient] registration failed (attempt %d/%d): %s",
                    attempt, self.retry, last_error,
                )
                if attempt < self.retry:
                    await asyncio.sleep(1.0 * attempt)
        return RegisterInstanceResponse(success=False, error=last_error)

    async def list_instances(self) -> list[InstanceInfo]:
        """Query the Host's registered instance list."""
        url = f"{self.base_url}/vllm_hi/instances"
        resp = await self.client.get(url)
        data = resp.json()
        return [InstanceInfo.model_validate(item) for item in data]

    async def heartbeat(self, vllm_hi_id: str) -> bool:
        """Send a heartbeat to the Host (placeholder; the Host currently doesn't
        require heartbeats)."""
        return True
