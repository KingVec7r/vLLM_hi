# SPDX-License-Identifier: Apache-2.0
"""State snapshot data models (full/incremental).

Here dataclass is deliberately used instead of pydantic (unlike ``common.types``):
snapshots are serialized across processes on the live-migration data plane via
pickle/cloudpickle, and the receiver accesses fields directly via ``__dict__`` /
``vars()`` (see ``receiver.py``); dataclass is lighter, has zero validation
overhead, and guarantees fields stay consistent across a pickle round-trip. This
is an intentional data-plane vs control-plane difference — do not switch this to
pydantic for the sake of "consistency".

The KV data-plane fields shared by ``StateSnapshot`` and
``IncrementalStateUpdate`` are factored into ``_KVPayloadMixin`` (``kw_only`` to
avoid default-value field ordering conflicts after inheritance), reducing
duplicate declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlockTableInfo:
    """A single block table entry."""

    block_ids: list[int] = field(default_factory=list)
    ref_count: int = 0
    block_hash: str | None = None


@dataclass
class RequestState:
    """Request execution state."""

    seq_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    sampling_params: Any = None  # base64-serialized SamplingParams
    status: str = "waiting"  # running/waiting/finished


@dataclass
class KVTensorMeta:
    """Layout metadata of a KV tensor within a contiguous memory buffer.

    Used in place of pickle serialization: each layer's KV tensor is carried in a
    contiguous memory buffer, and its position in the buffer is described by
    shape/dtype/byte_offset/nbytes; downstream can reconstruct it zero-copy via
    ``torch.frombuffer`` / ``np.frombuffer``.
    """

    layer_idx: int
    shape: list[int] = field(default_factory=list)
    dtype: str = "torch.float16"
    byte_offset: int = 0
    nbytes: int = 0
    head_dim: int = 1


@dataclass
class HeadShard:
    """Mapping of a head segment from a source global rank to a target global rank.

    ``src_rank`` is the source **global worker rank** (``pp_rank * src_tp + tp_rank``);
    ``head_offset``/``head_len`` describe the head segment on the head dim (dim=1)
    of the source rank's KV tensor that it owns; ``dst_head_offset`` describes the
    starting position on the head dim of the target rank's KV tensor to write into
    (used when TP shrink merges multiple source ranks). Under PP, the exact
    "layer → source stage" mapping is resolved on the restore side by the global
    ``KVTensorMeta.layer_idx``.
    """

    src_rank: int
    head_offset: int
    head_len: int
    dst_head_offset: int = 0


@dataclass(kw_only=True)
class _KVPayloadMixin:
    """KV data-plane fields shared by ``StateSnapshot`` / ``IncrementalStateUpdate``."""

    # A-side physical block ID order corresponding to dim 0 of the KV. Empty means
    # full (dim 0 = all preallocated blocks); non-empty means only these used
    # blocks were collected.
    kv_cache_blocks: list[int] = field(default_factory=list)

    # Layout metadata of each rank's per-layer KV tensor in the contiguous memory buffer:
    # {src_global_rank: [KVTensorMeta, ...]} (ascending by layer_idx, which is the global layer index).
    kv_tensor_metas: dict[int, list[KVTensorMeta]] = field(default_factory=dict)

    # Head-shard mapping for TP/PP changes: {dst_global_rank: [HeadShard, ...]}.
    # Filled by the B-side TopologyRemapper; used by the worker to locate the shm by src_rank and split heads.
    head_shard_map: dict[int, list[HeadShard]] = field(default_factory=dict)

    # GPU handles for the CUDA IPC direct backend: {(src_global_rank, global_layer_idx): handle_info}.
    # handle_info carries the IPC handle bytes + shape/stride/dtype, used by the
    # B-side worker to open shared VRAM for a GPU→GPU copy (no CPU relay).
    kv_ipc_handles: dict = field(default_factory=dict)

    # Scheduler queue state (running/waiting seq_id lists)
    running_seq_ids: list[str] = field(default_factory=list)
    waiting_seq_ids: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class StateSnapshot(_KVPayloadMixin):
    """Full state snapshot (for full transfer).

    kv_cache: {global_rank: {layer_idx: bytes}}
    block_table: {seq_id: BlockTableInfo}
    request_states: {seq_id: RequestState}
    """

    snapshot_id: str
    timestamp: float
    src_topology: dict[str, Any] = field(default_factory=dict)

    # KV Cache: {global_rank: {layer_idx: bytes}} (bytes = torch.save-serialized).
    # ``global_rank`` is the source global worker rank (``pp_rank * src_tp + tp_rank``).
    kv_cache: dict[int, dict[int, bytes]] = field(default_factory=dict)

    # Block Table
    block_table: dict[str, BlockTableInfo] = field(default_factory=dict)

    # Request execution state
    request_states: dict[str, RequestState] = field(default_factory=dict)


@dataclass(kw_only=True)
class IncrementalStateUpdate(_KVPayloadMixin):
    """Incremental state update."""

    update_id: str
    timestamp: float
    base_snapshot_id: str

    # Incremental KV Cache: only the changed layers and ranks (rank is the global worker rank).
    kv_cache_delta: dict[int, dict[int, bytes]] = field(default_factory=dict)

    # Incremental block table
    block_table_added: dict[str, BlockTableInfo] = field(default_factory=dict)
    block_table_modified: dict[str, BlockTableInfo] = field(default_factory=dict)
    block_table_removed: list[str] = field(default_factory=list)

    # Incremental request state
    request_states_added: dict[str, RequestState] = field(default_factory=dict)
    request_states_modified: dict[str, RequestState] = field(default_factory=dict)
    request_states_removed: list[str] = field(default_factory=list)
