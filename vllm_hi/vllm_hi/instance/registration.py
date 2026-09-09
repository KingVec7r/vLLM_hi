# SPDX-License-Identifier: Apache-2.0
"""Auto-registration logic for the A instance after startup.

Extract instance info from vLLM's ``args`` / ``vllm_config`` and call
HostClient to register with the Host scheduler.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_hi.common.types import InstanceType, Topology
from vllm_hi.common.utils import resolve_register_ip, visible_gpu_ids_from_env

logger = init_logger("vllm.vllm_hi.instance.registration")


async def auto_register_active_instance(state: Any, args: Any) -> None:
    """Automatically register with the Host after the A instance starts up.

    Extract instance info from args and vllm_config, then call
    HostClient.register.
    """
    try:
        hi = state.vllm_hi
        # Wait for the engine to be fully ready
        engine_client = hi.engine_client
        if engine_client is None:
            logger.warning("[vllm_hi] A auto-registration: engine_client unavailable")
            return

        host_client = hi.host_client
        if host_client is None:
            logger.warning("[vllm_hi] A auto-registration: host_client unavailable")
            return

        # Extract instance info
        vllm_config = getattr(engine_client, "vllm_config", None)
        if vllm_config is None:
            logger.warning("[vllm_hi] A auto-registration: vllm_config unavailable")
            return

        # Serving address (if host is a wildcard address, use the loopback IP
        # when registering)
        register_ip = resolve_register_ip(getattr(args, "host", "0.0.0.0"))
        port = getattr(args, "port", 8000)

        # served_model_name
        served_model_name = (
            args.served_model_name[0]
            if getattr(args, "served_model_name", None)
            else args.model
        )

        # Topology
        topology = Topology(
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.parallel_config.pipeline_parallel_size,
            gpu_memory_utilization=vllm_config.cache_config.gpu_memory_utilization,
            visible_gpu_ids=visible_gpu_ids_from_env(),
        )

        # inference_params (pass through key params)
        inference_params = {
            "model": args.model,
            "tensor_parallel_size": topology.tensor_parallel_size,
            "pipeline_parallel_size": topology.pipeline_parallel_size,
            "gpu_memory_utilization": topology.gpu_memory_utilization,
            "max_model_len": getattr(args, "max_model_len", None),
        }

        data_port = hi.data_port

        result = await host_client.register(
            ip=register_ip,
            port=port,
            instance_type=InstanceType.A,
            served_model_name=served_model_name,
            inference_params=inference_params,
            topology=topology,
            data_port=data_port,
        )
        if result.success:
            hi.registered = True
            hi.vllm_hi_id = result.vllm_hi_id
            logger.info(
                "[vllm_hi] A instance registered with Host: id=%s model=%s",
                result.vllm_hi_id, served_model_name,
            )
        else:
            logger.error("[vllm_hi] A instance registration failed: %s", result.error)
    except Exception:  # noqa: BLE001
        logger.exception("[vllm_hi] A auto-registration error")
