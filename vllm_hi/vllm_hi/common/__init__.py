# SPDX-License-Identifier: Apache-2.0
"""vllm_hi common base module: type definitions, configuration, and utility functions."""

from vllm_hi.common.types import (
    InstanceInfo,
    InstanceStatus,
    InstanceType,
    MigrationStatus,
    MigrationStep,
    MigrationType,
    Topology,
)
from vllm_hi.common.utils import (
    bind_process_to_device_numa,
    generate_id,
    get_device_cpu_affinity,
    now_iso8601,
    parse_addr,
    resolve_register_ip,
    topology_dict_from_vllm_config,
    topology_from_args,
    visible_gpu_ids_from_env,
)

__all__ = [
    "InstanceType",
    "InstanceStatus",
    "MigrationType",
    "MigrationStatus",
    "MigrationStep",
    "Topology",
    "InstanceInfo",
    "generate_id",
    "parse_addr",
    "now_iso8601",
    "get_device_cpu_affinity",
    "bind_process_to_device_numa",
    "resolve_register_ip",
    "topology_dict_from_vllm_config",
    "topology_from_args",
    "visible_gpu_ids_from_env",
]
