# SPDX-License-Identifier: Apache-2.0
"""Shared type definitions for vllm_hi.

Includes instance type/status, migration type/status enums, plus the topology
and instance-info data structures.

``Topology`` / ``InstanceInfo`` use pydantic models as the single source of
truth: they serve both as internal state and as Host control-plane request/
response models, eliminating the dual dataclass/pydantic models and hand-written
conversion.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InstanceType(str, Enum):
    """vLLM instance type."""

    A = "A"  # Instantiated and currently serving inference
    B = "B"  # Registered and on standby; instantiated upon receiving a migration command


class InstanceStatus(str, Enum):
    """Instance runtime status."""

    REGISTERING = "registering"
    ACTIVE = "active"
    STANDBY = "standby"
    MIGRATING = "migrating"
    DESTROYED = "destroyed"


class MigrationType(str, Enum):
    """Migration type."""

    PRESERVE_SOURCE = "preserve_source"  # Preserve source: A converted to type B (standby)
    DESTROY_SOURCE = "destroy_source"  # Destroy source: A is destroyed
    KEEP_SOURCE_ACTIVE = "keep_source_active"  # Preserve source: A stays type A and keeps handling un-migrated requests


class MigrationStatus(str, Enum):
    """Migration status."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


class MigrationStep(str, Enum):
    """Migration step (used by ``MigrationResult.current_step`` and ``step_history``).

    The values are the on-wire strings that clients use to observe migration
    progress; code references the enum members to avoid bare strings scattered
    around (see ``_notify`` in :mod:`vllm_hi.coordinator.migration`).
    """

    VALIDATE = "validate"
    B_INSTANTIATE = "b_instantiate"
    B_INSTANTIATE_DONE = "b_instantiate_done"
    FULL_TRANSFER = "full_transfer"
    FULL_TRANSFER_DONE = "full_transfer_done"
    INCREMENTAL_TRANSFER = "incremental_transfer"
    INCREMENTAL_TRANSFER_DONE = "incremental_transfer_done"
    PAUSE_A = "pause_a"
    PAUSE_A_DONE = "pause_a_done"
    FINAL_INCREMENTAL = "final_incremental"
    FINAL_INCREMENTAL_DONE = "final_incremental_done"
    SWITCH_SERVICE = "switch_service"
    SWITCH_SERVICE_DONE = "switch_service_done"
    HANDLE_SOURCE = "handle_source"
    DONE = "done"


class Topology(BaseModel):
    """Instance topology structure."""

    tensor_parallel_size: int = Field(1, ge=1, description="Tensor-parallel size (TP)")
    pipeline_parallel_size: int = Field(1, ge=1, description="Pipeline-parallel size (PP)")
    gpu_memory_utilization: float = Field(
        0.9, gt=0, le=1, description="GPU memory utilization ratio per device"
    )
    visible_gpu_ids: list[int] = Field(
        default_factory=list, description="List of visible GPU physical IDs"
    )


class InstanceInfo(BaseModel):
    """Full information of a registered instance."""

    vllm_hi_id: str = Field(description="Unique ID assigned by the Host")
    ip: str = Field(description="Instance service IP")
    port: int = Field(description="Instance service port")
    instance_type: InstanceType = Field(description="Instance type A/B")
    served_model_name: str = Field(description="Served model name (unique in the registry)")
    inference_params: dict[str, Any] = Field(
        default_factory=dict, description="Full inference service parameters (passed through to vLLM)"
    )
    status: InstanceStatus = InstanceStatus.REGISTERING
    registered_at: float = 0.0  # Unix wall-clock timestamp (time.time())
    topology: Topology = Field(default_factory=Topology)
    data_port: int = 0  # Data-plane TCP port (used for state transfer)

    @property
    def addr(self) -> str:
        """Return the ip:port string."""
        return f"{self.ip}:{self.port}"

    @property
    def base_url(self) -> str:
        """Return the http://ip:port string."""
        return f"http://{self.ip}:{self.port}"
