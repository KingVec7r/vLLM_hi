# SPDX-License-Identifier: Apache-2.0
"""Control-plane wire contract: request/response Pydantic models shared by the Host / A / B processes.

Placed under ``common`` rather than ``coordinator`` because both the Host
scheduler (coordinator) and the instance side (instance) use these DTOs; it is a
cross-process communication contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from vllm_hi.common.types import (
    InstanceInfo,
    InstanceStatus,
    InstanceType,
    MigrationStatus,
    MigrationType,
    Topology,
)


# --- Registration ---

class RegisterInstanceRequest(BaseModel):
    """Instance registration request."""

    ip: str = Field(..., description="Instance service IP")
    port: int = Field(..., ge=1, le=65535, description="Instance service port")
    instance_type: InstanceType = Field(..., description="Instance type A/B")
    served_model_name: str = Field(..., min_length=1, description="Served model name")
    inference_params: dict[str, Any] = Field(
        default_factory=dict, description="Full inference parameters"
    )
    topology: Topology = Field(..., description="Topology information")
    data_port: int = Field(0, ge=0, le=65535, description="Data-plane TCP port")


class RegisterInstanceResponse(BaseModel):
    """Instance registration response."""

    success: bool
    vllm_hi_id: str | None = None
    error: str | None = None


# --- Migration command ---

class MigrationCommand(BaseModel):
    """live migration command."""

    src_vllm_hi_id: str = Field(..., description="Source instance ID (type A)")
    dst_vllm_hi_id: str = Field(..., description="Destination instance ID (type B)")
    target_tp: int | None = Field(None, ge=1, description="Target tensor-parallel size")
    target_pp: int | None = Field(None, ge=1, description="Target pipeline-parallel size")
    target_gpu_memory_utilization: float | None = Field(
        None, gt=0, le=1, description="Target GPU memory utilization"
    )
    target_model: str | None = Field(None, description="Target model")
    migration_type: MigrationType = Field(
        MigrationType.DESTROY_SOURCE, description="Migration type"
    )
    tn: int = Field(2, ge=1, description="Number of incremental transfer rounds")


class MigrationResult(BaseModel):
    """Migration result."""

    migration_id: str
    status: MigrationStatus
    start_time: str  # ISO 8601
    end_time: str | None = None
    duration: float | None = None
    new_active_instance: InstanceInfo | None = None
    error: str | None = None
    # Current migration step (for clients to observe migration progress).
    # Values: validate, b_instantiate, full_transfer, incremental_transfer,
    #         pause_a, final_incremental, switch_service, handle_source, done
    current_step: str = "validate"
    # Full step history (with Unix timestamps), so polling intervals do not
    # miss intermediate events. Each element: {"step": str, "timestamp": float}
    step_history: list[dict[str, Any]] = Field(default_factory=list)


# --- State collection (A side) ---

class CollectStateRequest(BaseModel):
    """State collection request (Host → A)."""

    mode: Literal["full", "incremental"] = Field(..., description="Collection mode")
    dst_addr: str = Field(..., description="B data-plane address ip:port")
    round_idx: int = Field(0, ge=0, description="Incremental round index")
    last_snapshot_id: str | None = Field(None, description="Last snapshot ID")
    transfer_backend: str | None = Field(None, description="Data-plane backend name")


class CollectStateResponse(BaseModel):
    """State collection response."""

    success: bool
    snapshot_id: str | None = None
    data_size_bytes: int | None = None
    block_count: int | None = None
    error: str | None = None


# --- B instantiation ---

class InstantiateRequest(BaseModel):
    """Type-B instantiation request (Host → B)."""

    target_tp: int = Field(..., ge=1)
    target_pp: int = Field(..., ge=1)
    target_gpu_memory_utilization: float = Field(..., gt=0, le=1)
    target_model: str = Field(..., description="Model name or weights path")
    served_model_name: str = Field(..., description="Served model name")
    # Pass through other vLLM arguments
    extra_args: dict[str, Any] = Field(default_factory=dict)


class InstantiateResponse(BaseModel):
    """Type-B instantiation response."""

    success: bool
    error: str | None = None


# --- State receive notification (Host → B) ---

class ReceiveStateRequest(BaseModel):
    """Notify B to receive state data."""

    mode: Literal["full", "incremental"] = Field(...)
    src_addr: str = Field(..., description="A data-plane address ip:port")
    src_topology: Topology = Field(...)
    round_idx: int = Field(0, ge=0)
    last_snapshot_id: str | None = None
    transfer_backend: str | None = Field(None, description="Data-plane backend name")


class ReceiveStateResponse(BaseModel):
    success: bool
    snapshot_id: str | None = None
    error: str | None = None


# --- Pause / destroy / activate ---

class PauseResponse(BaseModel):
    success: bool
    error: str | None = None


class ActivateRequest(BaseModel):
    """Request to convert B to type A."""

    served_model_name: str = Field(...)


class ActivateResponse(BaseModel):
    success: bool
    error: str | None = None


class SimpleResponse(BaseModel):
    success: bool
    error: str | None = None
