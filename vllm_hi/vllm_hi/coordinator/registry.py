# SPDX-License-Identifier: Apache-2.0
"""Thread-safe instance registry."""

from __future__ import annotations

import asyncio
import itertools
import time

from vllm_hi.common.types import (
    InstanceInfo,
    InstanceStatus,
    InstanceType,
    Topology,
)
from vllm_hi.common.utils import generate_id
from vllm_hi.common.protocol import (
    RegisterInstanceRequest,
    RegisterInstanceResponse,
)


class InstanceRegistry:
    """Thread-safe instance registry.

    Maintains two indexes:
    - _instances: vllm_hi_id -> InstanceInfo
    - _model_index: served_model_name -> vllm_hi_id (A instances only, for routing)
    """

    def __init__(self) -> None:
        self._instances: dict[str, InstanceInfo] = {}
        self._model_index: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._id_counter = itertools.count(1)

    async def register(
        self, request: RegisterInstanceRequest
    ) -> RegisterInstanceResponse:
        """Register an instance.

        - Check served_model_name uniqueness (A instances only; B instances do
          not participate in routing)
        - Assign a vllm_hi_id
        - Store it in the registry
        """
        async with self._lock:
            # A instances must have a unique served_model_name (used for routing)
            # B instances do not participate in routing, so a duplicate
            # served_model_name is allowed (standby)
            if request.instance_type == InstanceType.A:
                if request.served_model_name in self._model_index:
                    existing_id = self._model_index[request.served_model_name]
                    return RegisterInstanceResponse(
                        success=False,
                        error=(
                            f"served_model_name {request.served_model_name!r} "
                            f"is already taken by instance {existing_id}"
                        ),
                    )

            vllm_hi_id = generate_id("inst")
            topology = request.topology
            info = InstanceInfo(
                vllm_hi_id=vllm_hi_id,
                ip=request.ip,
                port=request.port,
                instance_type=request.instance_type,
                served_model_name=request.served_model_name,
                inference_params=dict(request.inference_params),
                status=(
                    InstanceStatus.ACTIVE
                    if request.instance_type == InstanceType.A
                    else InstanceStatus.STANDBY
                ),
                registered_at=time.time(),
                topology=topology,
                data_port=request.data_port,
            )
            self._instances[vllm_hi_id] = info
            if request.instance_type == InstanceType.A:
                self._model_index[request.served_model_name] = vllm_hi_id
            return RegisterInstanceResponse(success=True, vllm_hi_id=vllm_hi_id)

    async def deregister(self, vllm_hi_id: str) -> bool:
        """Remove an instance from the registry. Return whether it succeeded.

        Only remove the route if the route index still points to this instance:
        after migration, the source A's route may have switched to the new A
        (dst), so deleting the source A must not wrongly remove the new A's
        route.
        """
        async with self._lock:
            info = self._instances.pop(vllm_hi_id, None)
            if info is None:
                return False
            if (
                info.instance_type == InstanceType.A
                and self._model_index.get(info.served_model_name) == vllm_hi_id
            ):
                self._model_index.pop(info.served_model_name, None)
            return True

    async def get_by_model(self, served_model_name: str) -> InstanceInfo | None:
        """Look up an A instance by served_model_name."""
        async with self._lock:
            vllm_hi_id = self._model_index.get(served_model_name)
            if vllm_hi_id is None:
                return None
            return self._instances.get(vllm_hi_id)

    async def get_by_id(self, vllm_hi_id: str) -> InstanceInfo | None:
        """Look up an instance by vllm_hi_id."""
        async with self._lock:
            return self._instances.get(vllm_hi_id)

    async def ensure_unique_model_name(self, base: str) -> str:
        """Return a unique model name that does not collide with existing instances.

        When "A stays type A" after migration, the new A instance needs a
        unique route id under the same model name; here we append a unique
        suffix as ``{base}@{suffix}`` so routing by model name stays unique.
        """
        async with self._lock:
            existing = {info.served_model_name for info in self._instances.values()}
            if base not in existing:
                return base
            name = f"{base}@{generate_id('mig')[4:]}"
            while name in existing:
                name = f"{base}@{generate_id('mig')[4:]}"
            return name

    async def update_type(
        self, vllm_hi_id: str, new_type: InstanceType
    ) -> InstanceInfo | None:
        """Update the instance type (A->B or B->A) and keep the model index in sync.

        Return the updated InstanceInfo, or None if the instance does not exist.
        """
        async with self._lock:
            info = self._instances.get(vllm_hi_id)
            if info is None:
                return None
            old_type = info.instance_type
            info.instance_type = new_type
            # Maintain the model index
            if old_type == InstanceType.A and new_type == InstanceType.B:
                # A->B: remove from the route index
                self._model_index.pop(info.served_model_name, None)
                info.status = InstanceStatus.STANDBY
            elif old_type == InstanceType.B and new_type == InstanceType.A:
                # B->A: add to the route index (check for conflicts)
                if info.served_model_name in self._model_index:
                    # Conflict: don't update the index, but still mark as A (the
                    # caller should clean up the old A first)
                    pass
                else:
                    self._model_index[info.served_model_name] = vllm_hi_id
                info.status = InstanceStatus.ACTIVE
            return info

    async def promote_standby_to_active(
        self,
        vllm_hi_id: str,
        served_model_name: str,
        target_tp: int,
        target_pp: int,
        target_util: float,
    ) -> bool:
        """Atomically promote B -> A: set type/status/model name/topology and
        take over routing.

        After migration, the target instance takes over service: perform the
        atomic update of "type B->A + status ACTIVE + model name + topology +
        route index" under a single lock, avoiding inconsistencies from callers
        bypassing the lock to mutate InstanceInfo fields directly. Preserve the
        visible GPU list (the GPUs the target instance registered with).
        """
        async with self._lock:
            info = self._instances.get(vllm_hi_id)
            if info is None:
                return False
            info.instance_type = InstanceType.A
            info.status = InstanceStatus.ACTIVE
            info.served_model_name = served_model_name
            info.topology = Topology(
                tensor_parallel_size=target_tp,
                pipeline_parallel_size=target_pp,
                gpu_memory_utilization=target_util,
                visible_gpu_ids=info.topology.visible_gpu_ids,
            )
            self._model_index[served_model_name] = vllm_hi_id
            return True

    async def update_status(
        self, vllm_hi_id: str, new_status: InstanceStatus
    ) -> InstanceInfo | None:
        """Update the instance status."""
        async with self._lock:
            info = self._instances.get(vllm_hi_id)
            if info is None:
                return None
            info.status = new_status
            return info

    async def update_topology(
        self, vllm_hi_id: str, topology: Topology
    ) -> InstanceInfo | None:
        """Update the instance topology (after migration)."""
        async with self._lock:
            info = self._instances.get(vllm_hi_id)
            if info is None:
                return None
            info.topology = topology
            return info

    async def list_all(self) -> list[InstanceInfo]:
        """Return all instance info."""
        async with self._lock:
            return list(self._instances.values())

    async def count(self) -> int:
        """Return the total number of registered instances (under the lock, to
        avoid callers accessing ``_instances`` directly)."""
        async with self._lock:
            return len(self._instances)

    async def list_all_models(self) -> list[InstanceInfo]:
        """Return a snapshot of all instance info (deep copy, to avoid data races
        under concurrent migration)."""
        async with self._lock:
            return [
                info.model_copy(deep=True)
                for info in self._instances.values()
            ]
