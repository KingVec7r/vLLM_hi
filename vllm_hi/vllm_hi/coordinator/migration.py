# SPDX-License-Identifier: Apache-2.0
"""MigrationCoordinator: live-migration scheduler that orchestrates the full
migration flow.

Flow:
1. Validate the command (src=A, dst=B, IDs exist)
2. Fill in defaults (topology/model/type/TN)
3. Trigger B instantiation (HTTP -> B)
4. Establish the state migration channel (A -> B)
5. Full transfer (round=0)
6. Incremental transfer x TN-1 (A keeps serving)
7. Pause A from generating new tokens
8. Incremental transfer TN (final)
9. Service switchover (B -> A, update the registry)
10. Source instance handling (destroy/preserve)
11. Return MigrationResult
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from vllm_hi.common.types import (
    InstanceStatus,
    InstanceType,
    MigrationStatus,
    MigrationStep,
    MigrationType,
    Topology,
)
from vllm_hi.common.utils import generate_id, now_iso8601, parse_addr
from vllm_hi.common.protocol import (
    ActivateRequest,
    CollectStateRequest,
    InstantiateRequest,
    MigrationCommand,
    MigrationResult,
    ReceiveStateRequest,
)
from vllm_hi.coordinator.registry import InstanceRegistry
from vllm_hi.migration.backends import select_backend
from vllm_hi.migration.runtime_config import RUNTIME_CONFIG
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.coordinator.migration")

# Maximum incremental rounds before pause (including the tn-1 base rounds), to
# guard against an infinite loop under abnormal conditions.
_MAX_INCR_ROUNDS = RUNTIME_CONFIG.max_incr_rounds
# How many consecutive incremental rounds not dropping below the historical
# minimum count as "stalled", triggering an early final round.
# Rationale: incremental data volume should decrease round over round as the
# transfer time shrinks (exponential convergence); if consecutive rounds stop
# hitting new lows, the KV cache is no longer converging (it keeps churning
# under full load), so further increments add little value and we should pause +
# final sooner to shorten the total migration time.
_INCR_STALL_ROUNDS = RUNTIME_CONFIG.incr_stall_rounds


class MigrationCoordinator:
    """Live-migration scheduler, running in the Host process."""

    def __init__(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.registry = registry
        self.http_client = http_client

    async def _post(
        self,
        url: str,
        json: Any = None,
        timeout: float = 60.0,
    ) -> httpx.Response:
        """Send a migration control request.

        Force a short-lived connection (``Connection: close``) to avoid reusing
        a stale connection already closed by the peer's keep-alive timeout,
        which would hang the request forever (uvicorn disconnects after ~5
        seconds idle by default, while migration steps are often tens of
        seconds apart).
        """
        return await self.http_client.post(
            url,
            json=json,
            headers={"Connection": "close"},
            timeout=httpx.Timeout(
                connect=10.0, read=timeout, write=30.0, pool=30.0
            ),
        )

    def create_pending_result(
        self, command: MigrationCommand
    ) -> MigrationResult:
        """Create a pending migration result (returned to the client immediately)."""
        return MigrationResult(
            migration_id=generate_id("mig"),
            status=MigrationStatus.IN_PROGRESS,
            start_time=now_iso8601(),
        )

    async def execute_migration(
        self, command: MigrationCommand, migration_id: str,
        progress_callback: Any = None,
    ) -> MigrationResult:
        """Execute the full migration flow.

        Args:
            progress_callback: Optional callback ``async (step: str) -> None``
                invoked on each completed step, so the client can observe
                migration progress.
        """
        start_ts = time.perf_counter()
        start_iso = now_iso8601()
        logger.info(
            "[vllm_hi Mig] migration started %s: src=%s dst=%s",
            migration_id, command.src_vllm_hi_id, command.dst_vllm_hi_id,
        )

        async def _notify(step: MigrationStep, **extra: Any) -> None:
            if progress_callback is not None:
                try:
                    await progress_callback(step.value, **extra)
                except Exception:  # noqa: BLE001
                    pass

        try:
            # 1. Validate the command
            src, dst = await self._validate_command(command)

            # Select the data-plane backend (auto or explicit); both A and B
            # use the same backend.
            backend = select_backend(
                src.ip, dst.ip, RUNTIME_CONFIG.transfer_backend
            )
            backend_name = backend.name
            logger.info(
                "[vllm_hi Mig] data-plane backend: %s (same_host=%s)",
                backend.name, src.ip == dst.ip,
            )

            # 2. Fill in defaults
            target_tp = command.target_tp or src.topology.tensor_parallel_size
            target_pp = command.target_pp or src.topology.pipeline_parallel_size
            target_util = (
                command.target_gpu_memory_utilization
                or src.topology.gpu_memory_utilization
            )
            target_model = command.target_model or src.inference_params.get(
                "model", ""
            )
            tn = command.tn
            migration_type = command.migration_type

            logger.info(
                "[vllm_hi Mig] migration params: tp=%d pp=%d util=%.3f model=%s tn=%d type=%s",
                target_tp, target_pp, target_util, target_model, tn,
                migration_type.value,
            )

            # Mark instance status
            await self.registry.update_status(
                src.vllm_hi_id, InstanceStatus.MIGRATING
            )
            await self.registry.update_status(
                dst.vllm_hi_id, InstanceStatus.MIGRATING
            )

            # 3. Trigger B instantiation
            await _notify(MigrationStep.B_INSTANTIATE)
            logger.info("[vllm_hi Mig] step 3: trigger B instantiation")
            await self._trigger_b_instantiation(
                dst, target_tp, target_pp, target_util, target_model,
                src.served_model_name,
            )
            logger.info("[vllm_hi Mig] step 3 done")
            await _notify(MigrationStep.B_INSTANTIATE_DONE)

            # 4. Establish the channel
            src_data_addr = (src.ip, src.data_port) if src.data_port else None
            dst_data_addr = (dst.ip, dst.data_port) if dst.data_port else None

            # 5. Full transfer
            await _notify(MigrationStep.FULL_TRANSFER)
            logger.info("[vllm_hi Mig] step 5: full transfer")
            full_bytes = await self._full_transfer(
                src, dst, src_data_addr, dst_data_addr,
                src.topology, target_tp, target_pp, target_util,
                backend_name,
            )
            logger.info("[vllm_hi Mig] step 5 done")
            await _notify(MigrationStep.FULL_TRANSFER_DONE, data_size_bytes=full_bytes)

            # 6. Incremental transfer (A keeps serving), adaptive rounds until
            # "stalled" or the round cap is reached.
            round_idx = await self._run_incremental_rounds(
                src, dst, dst_data_addr, full_bytes, tn, backend_name, _notify,
            )

            # 7. Pause A
            await _notify(MigrationStep.PAUSE_A)
            logger.info("[vllm_hi Mig] step 7: pause A from generating new tokens")
            await self._pause_a(src)
            logger.info("[vllm_hi Mig] step 7 done")
            await _notify(MigrationStep.PAUSE_A_DONE)

            # 8. Final incremental transfer
            await _notify(MigrationStep.FINAL_INCREMENTAL, round_idx=round_idx)
            logger.info("[vllm_hi Mig] final incremental transfer round %d", round_idx)
            final_bytes, final_blocks = await self._incremental_transfer(
                src, dst, dst_data_addr, round_idx, backend_name,
            )
            await _notify(
                MigrationStep.FINAL_INCREMENTAL_DONE,
                round_idx=round_idx, data_size_bytes=final_bytes,
                block_count=final_blocks,
            )

            # 9. Service switchover
            await _notify(MigrationStep.SWITCH_SERVICE)
            await self._switch_service(
                src, dst, target_tp, target_pp, target_util, migration_type
            )
            await _notify(MigrationStep.SWITCH_SERVICE_DONE)

            # 10. Handle the source instance
            await _notify(MigrationStep.HANDLE_SOURCE)
            await self._handle_source(src, migration_type)
            await _notify(MigrationStep.DONE)

            duration = time.perf_counter() - start_ts
            new_active = await self.registry.get_by_id(dst.vllm_hi_id)

            result = MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.SUCCESS,
                start_time=start_iso,
                end_time=now_iso8601(),
                duration=duration,
                new_active_instance=new_active,
            )
            logger.info(
                "[vllm_hi Mig] migration complete %s: duration=%.3fs",
                migration_id, duration,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("[vllm_hi Mig] migration failed %s", migration_id)
            # Restore A's status
            try:
                await self.registry.update_status(
                    command.src_vllm_hi_id, InstanceStatus.ACTIVE
                )
                await self.registry.update_status(
                    command.dst_vllm_hi_id, InstanceStatus.STANDBY
                )
            except Exception:  # noqa: BLE001
                pass
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                start_time=start_iso,
                end_time=now_iso8601(),
                duration=time.perf_counter() - start_ts,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _validate_command(
        self, command: MigrationCommand
    ) -> tuple[Any, Any]:
        """Validate the migration command and return the source and destination instance info."""
        src = await self.registry.get_by_id(command.src_vllm_hi_id)
        if src is None:
            raise ValueError(f"source instance not found: {command.src_vllm_hi_id}")
        if src.instance_type != InstanceType.A:
            raise ValueError(
                f"source instance is not type A: {command.src_vllm_hi_id} (type={src.instance_type.value})"
            )

        dst = await self.registry.get_by_id(command.dst_vllm_hi_id)
        if dst is None:
            raise ValueError(f"target instance not found: {command.dst_vllm_hi_id}")
        if dst.instance_type != InstanceType.B:
            raise ValueError(
                f"target instance is not type B: {command.dst_vllm_hi_id} (type={dst.instance_type.value})"
            )

        return src, dst

    async def _trigger_b_instantiation(
        self,
        dst: Any,
        target_tp: int,
        target_pp: int,
        target_util: float,
        target_model: str,
        served_model_name: str,
    ) -> None:
        """Notify the B instance via HTTP to instantiate with the target topology/model."""
        request = InstantiateRequest(
            target_tp=target_tp,
            target_pp=target_pp,
            target_gpu_memory_utilization=target_util,
            target_model=target_model,
            served_model_name=served_model_name,
        )
        url = f"{dst.base_url}/vllm_hi/migration/instantiate"
        resp = await self._post(url, json=request.model_dump(), timeout=300.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"B instantiation failed: {data.get('error', 'unknown error')}"
            )
        logger.info("[vllm_hi Mig] B instantiation complete")

    async def _full_transfer(
        self,
        src: Any,
        dst: Any,
        src_data_addr: tuple[str, int] | None,
        dst_data_addr: tuple[str, int] | None,
        src_topology: Topology,
        target_tp: int,
        target_pp: int,
        target_util: float,
        backend_name: str | None,
    ) -> int | None:
        """Perform a full state transfer.

        1. Notify B to prepare for receiving (set up topology remapping)
        2. Notify A to collect the full state and send it to B
        """
        # 1. Notify B to prepare for receiving
        if dst_data_addr:
            receive_req = ReceiveStateRequest(
                mode="full",
                src_addr=f"{src_data_addr[0]}:{src_data_addr[1]}" if src_data_addr else "",
                src_topology=src_topology,
                transfer_backend=backend_name,
            )
            url = f"{dst.base_url}/vllm_hi/migration/receive"
            try:
                await self._post(url, json=receive_req.model_dump(), timeout=30.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[vllm_hi Mig] failed to notify B to receive: %s", exc)

        # 2. Notify A to collect the full state and send it to B
        collect_req = CollectStateRequest(
            mode="full",
            dst_addr=f"{dst_data_addr[0]}:{dst_data_addr[1]}" if dst_data_addr else "",
            transfer_backend=backend_name,
        )
        url = f"{src.base_url}/vllm_hi/migration/collect"
        resp = await self._post(url, json=collect_req.model_dump(), timeout=600.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"full state collection failed: {data.get('error', 'unknown error')}"
            )
        full_bytes = data.get("data_size_bytes")
        logger.info(
            "[vllm_hi Mig] full transfer complete, data_size=%s bytes", full_bytes
        )
        return full_bytes

    async def _run_incremental_rounds(
        self,
        src: Any,
        dst: Any,
        dst_data_addr: tuple[str, int] | None,
        full_bytes: int | None,
        tn: int,
        backend_name: str | None,
        notify: Any,
    ) -> int:
        """Run the incremental transfer rounds before pause; return the final
        round index (used by the final increment).

        Adaptive multi-round transfer until "stalled" or the round cap is hit:
        under full load, A keeps producing new KV during each transfer round. If
        we only did tn-1 incremental rounds before pause, the final increment
        would accumulate to tens of GB and stretch downtime to tens of seconds.
        Multiple rounds let the incremental volume converge as the transfer time
        shrinks, moving as much data as possible before pause (A is still
        serving, so this does not count as downtime).
        """
        round_idx = 1
        # Seed the historical minimum with the full-transfer volume: the first
        # incremental round is already compared against the full transfer. If the
        # full transfer is the smallest and later increments stay larger, it is
        # counted as "stalled" from the first round.
        min_incr_bytes: int | None = full_bytes
        stall_rounds = 0  # consecutive rounds not below the historical minimum
        while True:
            # Record a start event (with round_idx) inside the loop for each
            # round, so rounds 1..N-1 before pause each get a step_history entry
            # for their start time.
            await notify(MigrationStep.INCREMENTAL_TRANSFER, round_idx=round_idx)
            logger.info(
                "[vllm_hi Mig] step 6: incremental transfer round %d (A keeps serving)",
                round_idx,
            )
            incr_bytes, block_count = await self._incremental_transfer(
                src, dst, dst_data_addr, round_idx, backend_name,
            )
            await notify(
                MigrationStep.INCREMENTAL_TRANSFER_DONE,
                round_idx=round_idx, data_size_bytes=incr_bytes,
                block_count=block_count,
            )
            round_idx += 1

            # Shortcut: this round's increment is <=1 block (no new KV, or just
            # 16 tokens), meaning the transfer has caught up with decoding; the
            # next round becomes the final round directly (pause + final).
            if block_count is not None and block_count <= 1:
                logger.info(
                    "[vllm_hi Mig] incremental round %d has only %d block(s) (<=1), "
                    "entering the final round early",
                    round_idx - 1, block_count,
                )
                break

            # Update the historical minimum incremental volume and the "stall" count.
            if incr_bytes is not None:
                if min_incr_bytes is None or incr_bytes < min_incr_bytes:
                    min_incr_bytes = incr_bytes
                    stall_rounds = 0
                else:
                    stall_rounds += 1

            # Stop conditions: (1) the maximum round cap is reached, or
            # (2) the minimum rounds tn-1 are met and consecutive rounds have
            #     not dropped below the historical minimum (KV is no longer
            #     converging, so enter pause + final as soon as possible).
            if round_idx > _MAX_INCR_ROUNDS:
                break
            if round_idx > (tn - 1) and stall_rounds >= _INCR_STALL_ROUNDS:
                logger.info(
                    "[vllm_hi Mig] incremental did not drop below the historical "
                    "minimum for %d consecutive rounds (%.0f bytes), entering "
                    "the final round early",
                    stall_rounds,
                    (min_incr_bytes if min_incr_bytes is not None else 0),
                )
                break
        logger.info("[vllm_hi Mig] step 6 done")
        return round_idx

    async def _incremental_transfer(
        self,
        src: Any,
        dst: Any,
        dst_data_addr: tuple[str, int] | None,
        round_idx: int,
        backend_name: str | None,
    ) -> tuple[int | None, int | None]:
        """Perform one incremental state transfer; return ``(data_size_bytes, block_count)``."""
        collect_req = CollectStateRequest(
            mode="incremental",
            dst_addr=f"{dst_data_addr[0]}:{dst_data_addr[1]}" if dst_data_addr else "",
            round_idx=round_idx,
            transfer_backend=backend_name,
        )
        url = f"{src.base_url}/vllm_hi/migration/collect"
        resp = await self._post(url, json=collect_req.model_dump(), timeout=600.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"incremental state collection failed (round={round_idx}): {data.get('error', 'unknown error')}"
            )
        incr_bytes = data.get("data_size_bytes")
        block_count = data.get("block_count")
        logger.info(
            "[vllm_hi Mig] incremental transfer round %d complete, data_size=%s bytes, blocks=%s",
            round_idx, incr_bytes, block_count,
        )
        return incr_bytes, block_count

    async def _pause_a(self, src: Any) -> None:
        """Notify the A instance to pause generating new tokens."""
        url = f"{src.base_url}/vllm_hi/migration/pause"
        resp = await self._post(url, timeout=30.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"failed to pause A: {data.get('error', 'unknown error')}"
            )

    async def _switch_service(
        self,
        src: Any,
        dst: Any,
        target_tp: int,
        target_pp: int,
        target_util: float,
        migration_type: MigrationType,
    ) -> None:
        """Service switchover: B -> A, update the registry route."""
        if migration_type == MigrationType.KEEP_SOURCE_ACTIVE:
            await self._switch_service_keep_source_active(
                src, dst, target_tp, target_pp, target_util
            )
            return

        # 1. Notify B to become type A
        activate_req = ActivateRequest(served_model_name=src.served_model_name)
        url = f"{dst.base_url}/vllm_hi/migration/activate"
        resp = await self._post(url, json=activate_req.model_dump(), timeout=120.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"B activation failed: {data.get('error', 'unknown error')}"
            )

        # 2. Atomically promote dst -> A and take over src's route
        ok = await self.registry.promote_standby_to_active(
            dst.vllm_hi_id, src.served_model_name,
            target_tp, target_pp, target_util,
        )
        if not ok:
            raise RuntimeError(f"target instance not found: {dst.vllm_hi_id}")
        logger.info("[vllm_hi Mig] service switchover complete: route %s -> %s",
                     src.served_model_name, dst.vllm_hi_id)

    async def _switch_service_keep_source_active(
        self,
        src: Any,
        dst: Any,
        target_tp: int,
        target_pp: int,
        target_util: float,
    ) -> None:
        """Service switchover for "A stays A" mode: dst becomes the new A with a
        unique model-name suffix, while src's route is left unchanged.

        After migration there are two A instances of the same model: the
        original A keeps serving unmigrated requests (original-name route),
        while the new A (dst) serves migrated requests (unique-suffix route).
        """
        # 1. Assign dst a unique model name (original name + unique suffix)
        new_name = await self.registry.ensure_unique_model_name(
            src.served_model_name
        )
        # 2. Notify B to become type A
        activate_req = ActivateRequest(served_model_name=new_name)
        url = f"{dst.base_url}/vllm_hi/migration/activate"
        resp = await self._post(url, json=activate_req.model_dump(), timeout=120.0)
        data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"B activation failed: {data.get('error', 'unknown error')}"
            )

        # 3. Atomically promote dst -> A (unique new-name route); src keeps its
        # original-name route unchanged.
        ok = await self.registry.promote_standby_to_active(
            dst.vllm_hi_id, new_name, target_tp, target_pp, target_util
        )
        if not ok:
            raise RuntimeError(f"target instance not found: {dst.vllm_hi_id}")
        logger.info(
            "[vllm_hi Mig] service switchover complete (source preserved): "
            "original A %s keeps its original-name route, new A %s routes via "
            "unique name %s",
            src.vllm_hi_id, dst.vllm_hi_id, new_name,
        )

    async def _handle_source(
        self, src: Any, migration_type: MigrationType
    ) -> None:
        """Source instance handling: destroy / convert to B / keep as A."""
        if migration_type == MigrationType.DESTROY_SOURCE:
            url = f"{src.base_url}/vllm_hi/migration/destroy"
            try:
                await self._post(url, timeout=30.0)
                await self.registry.deregister(src.vllm_hi_id)
                logger.info("[vllm_hi Mig] source instance destroyed: %s", src.vllm_hi_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[vllm_hi Mig] failed to destroy source: %s", exc)
        elif migration_type == MigrationType.PRESERVE_SOURCE:
            url = f"{src.base_url}/vllm_hi/migration/convert_to_b"
            try:
                await self._post(url, timeout=30.0)
                await self.registry.update_type(
                    src.vllm_hi_id, InstanceType.B
                )
                logger.info(
                    "[vllm_hi Mig] source instance converted to type B: %s", src.vllm_hi_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[vllm_hi Mig] failed to convert source to B: %s", exc)
        else:  # KEEP_SOURCE_ACTIVE
            # A stays type A; resume generation to keep serving unmigrated requests.
            url = f"{src.base_url}/vllm_hi/migration/resume"
            try:
                await self._post(url, timeout=30.0)
                await self.registry.update_status(
                    src.vllm_hi_id, InstanceStatus.ACTIVE
                )
                logger.info(
                    "[vllm_hi Mig] source instance stays type A and resumed generation: %s",
                    src.vllm_hi_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[vllm_hi Mig] failed to resume source generation: %s", exc)
