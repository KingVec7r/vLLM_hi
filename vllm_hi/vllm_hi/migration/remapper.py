# SPDX-License-Identifier: Apache-2.0
"""TopologyRemapper: topology remapper.

When the source topology ≠ target topology, remap the transferred state data:
- TP change (integer ratio): re-split/merge the KV Cache along the head dim
- PP change: route KV by the layer-range overlap of "source PP stage ↔ target PP stage"
- Block table / request state: physical block IDs are rebuilt by B's ``block_mapping``, no remap needed

**rank basis convention** (uniform across the whole pipeline): every rank key uses the
**global worker rank** within the vLLM world group, related to (pp_rank, tp_rank) by
``global_rank = pp_rank * tp_size + tp_rank`` (TP is the fastest-varying dimension,
consistent with ``pp_rank = rank // tp_size`` in
``ModelConfig.get_layers_start_end_indices``). Source and target each convert using
their own tp_size. Layer alignment is done on the restore side via the **global layer
index** carried in ``KVTensorMeta.layer_idx``; this module only produces the head-shard
description ``dst_global_rank -> [HeadShard(src_global_rank, ...)]``.
"""

from __future__ import annotations

from typing import Any

from vllm_hi.common.parallel import global_rank
from vllm_hi.migration.state_snapshot import (
    HeadShard,
    IncrementalStateUpdate,
    StateSnapshot,
)
from vllm.logger import init_logger

logger = init_logger("vllm.vllm_hi.migration.remapper")


class TopologyRemapper:
    """Topology remapper.

    Attributes:
        src_topology: Source topology (dict with tensor_parallel_size, pipeline_parallel_size, ...)
        dst_topology: Target topology
    """

    def __init__(
        self,
        src_topology: dict[str, Any],
        dst_topology: dict[str, Any],
    ) -> None:
        self.src_topology = src_topology
        self.dst_topology = dst_topology
        self.src_tp = int(src_topology.get("tensor_parallel_size", 1))
        self.dst_tp = int(dst_topology.get("tensor_parallel_size", 1))
        self.src_pp = int(src_topology.get("pipeline_parallel_size", 1))
        self.dst_pp = int(dst_topology.get("pipeline_parallel_size", 1))
        self.src_world = self.src_tp * self.src_pp
        self.dst_world = self.dst_tp * self.dst_pp

    @property
    def needs_remap(self) -> bool:
        """Whether topology remapping is needed."""
        return (self.src_tp != self.dst_tp) or (self.src_pp != self.dst_pp)

    # ------------------------------------------------------------------
    # PP layer mapping
    # ------------------------------------------------------------------

    @staticmethod
    def pp_layer_range(
        pp_rank: int, pp_size: int, num_layers: int
    ) -> tuple[int, int]:
        """Return the global layer range ``[start, end)`` owned by pp_rank (identical to vLLM)."""
        from vllm.distributed.utils import get_pp_indices

        return get_pp_indices(num_layers, pp_rank, pp_size)

    @staticmethod
    def _num_layers(kv_tensor_metas: dict[int, list] | None) -> int | None:
        """Infer the total layer count from KV metadata (max(layer_idx) + 1); return None without metadata."""
        if not kv_tensor_metas:
            return None
        max_layer = -1
        for metas in kv_tensor_metas.values():
            for m in metas or []:
                max_layer = max(max_layer, int(getattr(m, "layer_idx", -1)))
        return (max_layer + 1) if max_layer >= 0 else None

    def compute_pp_layer_mapping(
        self, num_layers: int | None
    ) -> dict[int, list[int]]:
        """Compute the PP layer mapping: ``{dst_pp_rank: [src_pp_rank, ...]}``.

        Each target PP stage reads only the source PP stages whose global layer
        range overlaps its own. Layer ranges are computed with vLLM's
        ``get_pp_indices``; when ``num_layers`` is unknown, degrade to "all source
        stages feed all target stages", and the restore side filters by global
        layer index (layer_idx).
        """
        if num_layers is None:
            return {
                dst_pp: list(range(self.src_pp))
                for dst_pp in range(self.dst_pp)
            }
        mapping: dict[int, list[int]] = {}
        for dst_pp in range(self.dst_pp):
            dst_start, dst_end = self.pp_layer_range(
                dst_pp, self.dst_pp, num_layers
            )
            src_pps: list[int] = []
            for src_pp in range(self.src_pp):
                src_start, src_end = self.pp_layer_range(
                    src_pp, self.src_pp, num_layers
                )
                if src_end > dst_start and dst_end > src_start:
                    src_pps.append(src_pp)
            mapping[dst_pp] = src_pps
        return mapping

    # ------------------------------------------------------------------
    # TP head shards (within a single source PP stage)
    # ------------------------------------------------------------------

    def _tp_shards(
        self, src_pp: int, dst_tp_rank: int, src_head: int | None
    ) -> list[HeadShard]:
        """Return the head-shard list a target TP rank reads from a single source PP stage.

        ``src_rank`` in each shard is converted to a global rank
        (``global_rank(src_pp, tp, src_tp)``). The expand/shrink/unchanged cases are
        orthogonal to the TP-independent PP layer count and reused per stage.
        """
        shards: list[HeadShard] = []
        if self.dst_tp > self.src_tp:
            if self.dst_tp % self.src_tp != 0:
                raise ValueError(
                    f"TP expansion ratio is not an integer: dst_tp={self.dst_tp} does not divide "
                    f"src_tp={self.src_tp}"
                )
            ratio = self.dst_tp // self.src_tp
            dst_head = (src_head // ratio) if src_head else 0
            src_tp_rank = dst_tp_rank // ratio
            head_offset = (dst_tp_rank % ratio) * dst_head
            shards.append(
                HeadShard(
                    src_rank=global_rank(src_pp, src_tp_rank, self.src_tp),
                    head_offset=head_offset,
                    head_len=dst_head,
                )
            )
        elif self.dst_tp < self.src_tp:
            if self.src_tp % self.dst_tp != 0:
                raise ValueError(
                    f"TP shrink ratio is not an integer: src_tp={self.src_tp} does not divide "
                    f"dst_tp={self.dst_tp}"
                )
            ratio = self.src_tp // self.dst_tp
            for k in range(ratio):
                src_tp_rank = dst_tp_rank * ratio + k
                shards.append(
                    HeadShard(
                        src_rank=global_rank(src_pp, src_tp_rank, self.src_tp),
                        head_offset=0,
                        head_len=src_head or 0,
                        dst_head_offset=(k * src_head) if src_head else 0,
                    )
                )
        else:
            shards.append(
                HeadShard(
                    src_rank=global_rank(src_pp, dst_tp_rank, self.src_tp),
                    head_offset=0,
                    head_len=0,
                )
            )
        return shards

    # ------------------------------------------------------------------
    # KV Cache remapping
    # ------------------------------------------------------------------

    def remap_kv_cache(
        self,
        kv_cache: dict[int, dict[int, bytes]],
        kv_tensor_metas: dict[int, list] | None = None,
    ) -> tuple[dict[int, dict[int, bytes]], dict[int, list[HeadShard]]]:
        """Compute the KV Cache head-shard mapping, returning (kv_cache unchanged, head_shard_map).

        This method **copies no data**: kv_cache stays as ``src_world`` copies (each
        a full buffer of one source global rank) returned unchanged, and only the
        shard description ``dst_global_rank -> [HeadShard, ...]`` is produced, from
        which the B-side worker locates the shm by ``src_rank`` (global rank) and
        moves only the head segment it owns.

        head_len semantics: >0 means split exactly this segment; ==0 means the full
        head segment (fallback when TP is unchanged or the head count is unknown),
        resolved on the worker side by the actual tensor head count.

        PP dimension: a target global rank reads only from source PP stages whose
        global layer range overlaps its own; the exact layer-to-source-stage mapping
        is resolved on the restore side by the global ``layer_idx``.
        """
        src_head = self._src_heads_per_rank(kv_tensor_metas)
        num_layers = self._num_layers(kv_tensor_metas)
        pp_map = self.compute_pp_layer_mapping(num_layers)

        head_shard_map: dict[int, list[HeadShard]] = {}
        for dst_pp in range(self.dst_pp):
            for dst_tp_rank in range(self.dst_tp):
                dst_rank = global_rank(dst_pp, dst_tp_rank, self.dst_tp)
                shards: list[HeadShard] = []
                for src_pp in pp_map.get(dst_pp, []):
                    shards.extend(self._tp_shards(src_pp, dst_tp_rank, src_head))
                head_shard_map[dst_rank] = shards

        logger.info(
            "[vllm_hi Remapper] KV Cache head-shard mapping: src(tp=%d,pp=%d) → "
            "dst(tp=%d,pp=%d), data copies stay at %d (no copy), mapping covers %d target ranks",
            self.src_tp, self.src_pp, self.dst_tp, self.dst_pp,
            len(kv_cache), len(head_shard_map),
        )
        return kv_cache, head_shard_map

    def _src_heads_per_rank(
        self, kv_tensor_metas: dict[int, list] | None
    ) -> int | None:
        """Read the source per-rank head count from KV tensor metadata (taking shape[1] of the first layer)."""
        if not kv_tensor_metas:
            return None
        for rank in sorted(kv_tensor_metas):
            metas = kv_tensor_metas[rank]
            if not metas:
                continue
            shape = getattr(metas[0], "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) > 1:
                return int(shape[1])
        return None

    def remap_block_table(
        self,
        block_table: dict[str, Any],
    ) -> dict[str, Any]:
        """Block table needs no remap: physical block IDs are topology-dependent runtime allocations. B reallocates blocks in ``_vllm_hi_set_scheduler_state`` and produces the ``a_block→b_block`` mapping, and the KV restore phase moves blocks per that mapping, so block_id can stay as-is here."""
        return block_table

    def remap_request_states(
        self,
        request_states: dict[str, Any],
    ) -> dict[str, Any]:
        """Request execution state needs no remap: block references are rebuilt on B, and the rest (prompt/output tokens, etc.) is topology-independent and stays as-is."""
        return request_states

    def remap_snapshot(self, snapshot: StateSnapshot) -> StateSnapshot:
        """Remap a full snapshot (fills the head-shard mapping, copies no KV data)."""
        snapshot.kv_cache, snapshot.head_shard_map = self.remap_kv_cache(
            snapshot.kv_cache, snapshot.kv_tensor_metas
        )
        snapshot.block_table = self.remap_block_table(snapshot.block_table)
        snapshot.request_states = self.remap_request_states(
            snapshot.request_states
        )
        return snapshot

    def remap_incremental(
        self, update: IncrementalStateUpdate
    ) -> IncrementalStateUpdate:
        """Remap an incremental update (fills the head-shard mapping, copies no KV data)."""
        update.kv_cache_delta, update.head_shard_map = self.remap_kv_cache(
            update.kv_cache_delta, update.kv_tensor_metas
        )
        update.block_table_added = self.remap_block_table(
            update.block_table_added
        )
        update.block_table_modified = self.remap_block_table(
            update.block_table_modified
        )
        update.request_states_added = self.remap_request_states(
            update.request_states_added
        )
        update.request_states_modified = self.remap_request_states(
            update.request_states_modified
        )
        return update
