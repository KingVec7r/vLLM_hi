# vllm_hi Feature Overview

> This document summarizes all features and performance-optimization tricks of vllm_hi.
> Updated 2026-09-09, against the `vllm_hi/vllm_hi/` source tree (~7400 lines across the `common/` / `coordinator/` / `instance/` / `migration/` / `patches/` packages).

---

## 1. Project Positioning

vllm_hi is a package that extends vLLM **without modifying the vLLM source**, via the `vllm.general_plugins` entry point + monkey-patching, to provide **low-downtime live migration** for inference services:

- Keeps a B instance (standby) next to the A instance (active).
- On migration, transfers A's KV Cache + scheduler state to B, which continues serving; client streaming connections stay alive.
- `import vllm_hi` applies all customizations automatically (idempotent).

---

## 2. Architecture

Three roles + two channels:

```
 client ──(SSE streaming)──> Host (SchedulerServer, :8002)
                               │ routing + migration orchestration
                               ▼
                         A instance (:8000, active) ──migrate──> B instance (:8001, standby)
                               │                                        │
                               └── data plane (KV blobs, pluggable) ────┘
```

- **Control plane**: HTTP / FastAPI. Cross-process request/response contracts are defined in `common/protocol.py` (Pydantic DTOs).
- **Data plane**: large KV blocks travel over a pluggable backend (see §5); the small scheduler state travels over the control-plane HTTP.
  - `tcp_shm`: GPU→CPU(shm)→TCP→CPU(shm)→GPU (default, cross-node / fallback).
  - `cuda_ipc`: GPU→GPU direct (same-node P2P).
  - `rdma`: cross-node GPUDirect (skeleton, `implemented=False`).

**Migration flow** (`coordinator/migration.py`, step names in the `MigrationStep` enum):

```
validate → b_instantiate → full_transfer → incremental ×N (adaptive convergence)
→ pause_a → final_incremental → switch_service → handle_source (destroy/preserve/keep)
```

Progress is exposed to clients via `MigrationResult.current_step` + `step_history` (including `*_done` variants).

---

## 3. Module Overview

```
vllm_hi/
  common/          types / config / utils / contracts
    protocol.py       control-plane wire contract (Pydantic request/response DTOs)
    types.py          enums (Instance/Migration types & status, MigrationStep) + Topology + InstanceInfo
    state.py          VllmHiState (instance state dataclass on app.state.vllm_hi)
    parallel.py       global worker rank & PP layer-index helpers (global_rank / worker_global_rank / layer_start_index)
    rpc.py            _vllm_hi_* RPC contract constants
    utils.py          general utils (generate_id/parse_addr/sampling-params cache/NUMA/topology helpers)
  coordinator/     Host scheduler
    server.py         SchedulerServer (FastAPI: register/list/migrate/status + /v1 proxy)
    registry.py       InstanceRegistry (thread-safe registry)
    router.py         RequestRouter + InflightStream (streaming proxy + in-flight stream reattach)
    migration.py      MigrationCoordinator (migration orchestration)
  instance/        A/B instances
    host_client.py    HostClient (instance ↔ Host)
    registration.py   A-instance auto-registration
    activate_endpoint.py  A-side migration endpoints (collect/pause/resume/destroy/convert_to_b)
    standby.py        B standby startup + on-demand instantiation + data-plane server + resume_stream + util clamp
  patches/         monkey-patch + entry point
    __init__.py       facade (apply + patch_* functions, injection only)
    engine_rpc.py     EngineCore/Worker RPC method implementations (state machine)
    plugin.py         vllm.general_plugins entry point (worker subprocesses also patched)
  migration/       migration core
    backends/         data-plane transfer backends (pluggable)
      base.py           KVTransferBackend abstraction + TransferContext
      registry.py       registry + select_backend auto-selection
      tcp_shm.py        TcpStateTransfer (TCP wire protocol) + TcpShmTransferBackend
      cuda_ipc.py       CudaIpcTransferBackend (GPU direct)
      rdma.py           RdmaTransferBackend (skeleton)
    collector.py      A-side state collection orchestration (scheduler + chunked KV aggregation)
    kv_collector.py   in-worker background KV collector (D2H throttling + shm encoding)
    receiver.py       B-side state restore (scheduler + KV)
    remapper.py       topology remapping (TP head shards / PP layer mapping)
    restore_worker.py in-worker KV restore (module-level, unit-testable)
    resume.py         resume output_processor
    filter.py         request-filter extension point (RequestFilter)
    runtime_config.py centralized config (magic numbers / probe switches)
    state_snapshot.py snapshot dataclasses (full/incremental)
```

---

## 4. Feature List (by module)

### 4.1 `common/` — types / contracts / utils
| File | Responsibility |
|---|---|
| `protocol.py` | control-plane DTOs: register / migration command / migration result / state collect / instantiate / receive / activate |
| `types.py` | enums `InstanceType(A/B)`, `InstanceStatus`, `MigrationType`, `MigrationStatus`, `MigrationStep`; `Topology`(TP/PP/util/visible_gpu_ids), `InstanceInfo` |
| `state.py` | `VllmHiState`: centralizes instance type / Host address / engine_client / host_client / collector / receiver / resumed_queues, etc. |
| `parallel.py` | `global_rank(pp,tp,tp_size)` pure function + `worker_global_rank()` + `layer_start_end/index()` (unified rank convention) |
| `rpc.py` | `_vllm_hi_*` RPC method-name constants (shared by injection and data-plane references) |
| `utils.py` | `generate_id`/`parse_addr`/`now_iso8601`/`deserialize_sampling_params`(bounded cache)/`get_device_cpu_affinity`/`bind_process_to_device_numa`/topology & IP helpers |

### 4.2 `coordinator/` — Host scheduler
| File | Responsibility |
|---|---|
| `server.py` | `SchedulerServer`: `/vllm_hi/register|instances|migrate|migration/{id}/status|status` + `/v1/chat/completions|completions|models` proxy; async migration + progress callback into `step_history` |
| `registry.py` | `InstanceRegistry`: `register/deregister/get_by_model/get_by_id/ensure_unique_model_name/update_type/promote_standby_to_active/update_status/update_topology/list_all/count` |
| `router.py` | `RequestRouter` + `InflightStream`: SSE streaming proxy; `switch_inflight` reattaches in-flight streams to B's `resume_stream`; retries once on transport error |
| `migration.py` | `MigrationCoordinator`: orchestrates migration, backend selection, adaptive incremental convergence, service switch and source handling for three `MigrationType`s |

### 4.3 `instance/` — A/B instances
| File | Responsibility |
|---|---|
| `host_client.py` | `HostClient`: registers with Host (with retry), queries instance list |
| `registration.py` | `auto_register_active_instance`: A extracts info from args/vllm_config and auto-registers on startup |
| `activate_endpoint.py` | A-side migration endpoints `collect/pause/resume/destroy/convert_to_b`; `attach_active_endpoints` resolves app via `state.vllm_hi.app` |
| `standby.py` | B standby startup (skips engine instantiation, `_StandbyEngineClient` placeholder), on-demand `instantiate_engine`, lazy data-plane server, `resume_stream`, `_clamp_util_to_free_gpu_memory` |

### 4.4 `migration/` — migration core
| File | Responsibility |
|---|---|
| `collector.py` | `StateCollector`: full/incremental collection, request filtering, **chunked collection + cross-chunk aggregation**, completion-time estimation |
| `kv_collector.py` | `KvCollector`: in-worker background thread + dedicated CUDA stream; `D2HThrottler`/`SmoothD2HThrottler`/`PinnedMemoryPool` |
| `receiver.py` | `StateReceiver`: topology remap + scheduler restore + KV restore (shm write + in-worker GPU copy) |
| `remapper.py` | `TopologyRemapper`: TP expand/shrink (head shards), PP change (layer mapping); emits shard descriptions without copying data |
| `restore_worker.py` | `set_kv_cache`: module-level (unit-testable); copies shm data back to GPU per head-shard/block mapping |
| `resume.py` | `register_resumed_outputs`: registers output_processor and warms up detokenizer for restored requests |
| `filter.py` | `RequestFilter` / `MigrateAllFilter` / `load_request_filter` (request-filter extension point) |
| `runtime_config.py` | `MigrationRuntimeConfig` + `ProbeConfig` (frozen dataclasses, centralized env parsing) |
| `state_snapshot.py` | `BlockTableInfo`/`RequestState`/`KVTensorMeta`/`HeadShard`/`StateSnapshot`/`IncrementalStateUpdate` (data-plane dataclasses) |

### 4.5 `patches/` — runtime patches
| Patch | Effect |
|---|---|
| `patch_cli_args` | adds `--vllm-hi-type/host/data-port` CLI flags |
| `patch_build_app` | `GET /vllm_hi/status` route + `X-VLLM-HI` header on all responses (**pure ASGI middleware**, preserves SSE) |
| `patch_chat_completion_logging` | chat-request logging (model / sampling params / source IP / first 80 chars of last message) |
| `patch_init_app_state` | injects `state.vllm_hi`, attaches migration endpoints, A auto-registration; stashes the original `init_app_state` for B's serving-state reattach |
| `patch_run_server` | B takes the standby path (skips engine instantiation); A additionally does NUMA binding |
| `patch_engine_internals` | injects RPC methods into `EngineCore`/`Worker` (scheduler get/set/apply, KV IPC export/set, background collect dispatch/try_get) |
| `plugin.py` | `vllm.general_plugins` entry point, ensures engine_internals patches apply in worker subprocesses |

---

## 5. Pluggable Data-plane Backends

`migration/backends/base.py` defines the `KVTransferBackend` abstraction (`send_state` / `start_data_server`) with metadata `name`/`gpu_direct`/`cross_node`/`priority`/`implemented`/`available()`. Implemented:

| Backend | `gpu_direct` | Scenario | Notes |
|---|---|---|---|
| `tcp_shm` | no | cross-node / fallback | GPU→CPU(shm)→TCP→CPU(shm)→GPU |
| `cuda_ipc` | yes | same-node P2P | GPU→GPU direct (CUDA IPC handle); only meta over control TCP |
| `rdma` | yes | cross-node GPUDirect | skeleton (`implemented=False`); auto-degraded without RDMA hardware |

**Auto-selection** (`backends/registry.py:select_backend`):
1. `VLLM_HI_TRANSFER_BACKEND` explicit override first (warn + fall back to `tcp_shm` if unavailable/unimplemented);
2. same-node with full P2P connectivity → `cuda_ipc`;
3. cross-node with RDMA available → `rdma`;
4. fallback `tcp_shm`.

The migration coordinator selects the backend by whether A/B are co-located, and passes it through `CollectStateRequest`/`ReceiveStateRequest` so both sides use the same backend.

---

## 6. Migration Types (`MigrationType`)

| Type | Post-migration behavior |
|---|---|
| `destroy_source` | destroy A, B takes over the original model-name route (default) |
| `preserve_source` | A becomes B-class (standby), B takes over the original route |
| `keep_source_active` | A stays active and keeps serving non-migrated requests; B becomes the new A registered under `{name}@{unique suffix}` (unique route by model name) |

---

## 7. Request-filter Extension Point

`migration/filter.py` defines the `RequestFilter` interface (`should_migrate(seq_id, request_state) -> bool`), with `MigrateAllFilter` (migrate everything) as default. Set `VLLM_HI_REQUEST_FILTER=module.Class` to customize (e.g. migrate only long requests); filtered-out requests do not have their KV collected or migrated, and keep running on A.

---

## 8. Performance Optimization Tricks

> Techniques accumulated through repeated tuning, grouped by transfer-link stage.

### 8.1 Collection side (A) — reducing interference with inference
| # | Trick | Location | Rationale / measured gain |
|---|---|---|---|
| T1 | **Collect only used blocks** | `collector.collect_full` | collects only physical blocks referenced by the block table, not all preallocated blocks, greatly reducing D2H and transfer volume |
| T2 | **Chunked collection** | `collector._collect_worker_kv_cache_chunked` | 512 blocks/chunk (`VLLM_HI_KV_COLLECT_CHUNK_BLOCKS`); final incremental (A paused) uses a 1<<20 large chunk to eliminate repeated RPC round-trips and aggregation copies |
| T3 | **Background thread + dedicated CUDA stream** | `kv_collector.KvCollector` | GPU→CPU copy and CPU encode-to-shm no longer block `execute_model`; inference and data movement fully parallel (single-active-task model) |
| T4 | **Smooth D2H throttling (event feedback loop)** | `kv_collector.SmoothD2HThrottler` | derives instantaneous bandwidth from measured `Event.elapsed_time` and yields dynamically against the budget (default 25% of HBM peak), keeping instantaneous bandwidth near budget without starving inference matmul HBM reads |
| T5 | **Pinned memory pool** | `kv_collector.PinnedMemoryPool` | preallocated pinned buffer makes `copy_` truly async (no hidden pageable staging + sync copy-back); falls back to pageable on allocation failure |
| T6 | **Contiguous-range slicing (not fancy-index)** | `kv_collector.slice_blocks` | `t[start:end]` zero-copy view eliminates the `t[block_ids]` gather kernel and GB-scale temporary GPU allocations |
| T7 | **Raise CPU thread count only when A is paused** | `kv_collector.collect_cpu_threads` | only the final incremental (A stopped) boosts to 8–16 threads (single-thread ~1.6GB/s → 8–16 threads ~18GB/s); keeps single thread while serving |
| T8 | **Poll backoff + completion-estimate first sleep** | `collector._collect_worker_kv_cache` | sleeps to near-completion using an empirical rate (20GB/s) before the first poll, compressing RPC interleaving; then 20ms→200ms exponential backoff |
| T9 | **Bounded SamplingParams deserialization cache** | `utils.deserialize_sampling_params` | FIFO cache (256) avoids repeated pickle.loads of the same object; returns shallow copy to avoid per-request mutation pollution |
| T10 | **Explicit `del`, no gc.collect()** | `collector.release_kv_cache_shm` | explicitly releases memoryview/tensor refs so shm.close() doesn't raise BufferError; avoids stop-the-world gc holding the GIL and blocking the event loop |

### 8.2 Encoding / aggregation (A)
| # | Trick | Location | Rationale / measured gain |
|---|---|---|---|
| T11 | **torch.cat aggregation (in thread pool)** | `collector._aggregate_kv_chunks` | ~23GB/s, replacing numpy slice assignment ~1.85GB/s; run in `run_in_executor` (large copy releases GIL) without blocking the event loop |
| T12 | **Contiguous memory buffer + metadata (no pickle)** | `kv_collector._compute_metas` / `collector._assemble_kv_cache` | each layer described by `byte_offset/shape/dtype/nbytes`, zero-copy rebuilt via `torch.frombuffer`, replacing pickle of large tensors |

### 8.3 Restore side (B)
| # | Trick | Location | Rationale / measured gain |
|---|---|---|---|
| T13 | **madvise(MADV_POPULATE_WRITE)** | `receiver._populate_shm_write` | preallocates shm physical pages, eliminating page-fault storms (1.6GB/s → ~10GB/s) |
| T14 | **numpy single-thread chunked memcpy to shm** | `receiver._restore_kv_cache` | ~10GB/s warm; replaces torch multi-thread `copy_` (OpenMP spin contends with A inference); `sleep` between chunks yields CPU |
| T15 | **NUMA binding for shm write** | `receiver._restore_kv_cache` / `utils.get_device_cpu_affinity` | binds the shm-writing process to B's GPU NUMA node, spatially isolating from A, eliminating cross-NUMA memory/bandwidth contention |
| T16 | **Slice head before moving to GPU** | `restore_worker.set_kv_cache` | each rank moves only its own head segment on TP expand/shrink, eliminating redundant movement |
| T17 | **Move only used blocks + chunked temp VRAM** | `restore_worker.set_kv_cache` | moves only used blocks to GPU (prevents full-layer OOM at high util), 512 blocks per slice to bound GPU temporary usage |
| T18 | **Cap restore threads at 4 + inter-layer yield** | `restore_worker.set_kv_cache` | limits torch threads (`VLLM_HI_KV_RESTORE_THREADS`=4), `sleep` between layers to yield CPU/PCIe bandwidth to A |

### 8.4 Transfer (data plane)
| # | Trick | Location | Rationale |
|---|---|---|---|
| T19 | **recv_into + large sendall + SO_SNDBUF** | `backends/tcp_shm.py` | `recv_into` reuses buffer avoiding bytearray reallocation; `SO_SNDBUF`=64MB, `TCP_NODELAY`; `sendall` runs in executor |
| T20 | **ACK handshake** | `backends/tcp_shm.send_state` | A blocks for B's "applied" ACK, fixing the final-incremental vs activate race (fire-and-forget could apply final data after activate) |
| T21 | **asyncio.Lock serialization + data_idle** | `backends/tcp_shm.start_data_server` | serializes full/incremental state rebuild, maintains a `data_idle` event that activate waits on |
| T22 | **CUDA IPC direct + contiguous-range memcpy** | `backends/cuda_ipc.py` / `engine_rpc.worker_set_kv_ipc` | KV moves GPU→GPU via IPC handle (no CPU); block mapping grouped into contiguous "a and b both contiguous" ranges for whole-segment memcpy (scattered gather ~1.5GB/s measured) |
| T23 | **IPC storage cache** | `engine_rpc.worker_set_kv_ipc` | caches opened shared VRAM across rounds, avoiding `cudaErrorInvalidResourceHandle` (each cudaIpcMemHandle can only be opened once per device) |

### 8.5 Scheduling / orchestration
| # | Trick | Location | Rationale |
|---|---|---|---|
| T24 | **Adaptive incremental convergence (stall detection)** | `migration.MigrationCoordinator._run_incremental_rounds` | loops increments until "`VLLM_HI_INCR_STALL_ROUNDS`(=2) consecutive rounds no smaller than historical min" or "`VLLM_HI_MAX_INCR_ROUNDS`(=8) rounds" or "a round ≤1 block", then pauses + final, moving as much data as possible before pause |
| T25 | **Connection: close short connections** | `migration.MigrationCoordinator._post` | avoids reusing keep-alive-stale connections that hang requests |
| T26 | **Request-restore token alignment** | `engine_rpc.resolve_restore_tokens` | allocates all blocks + `computed=num_tokens-1` for decode requests, bypassing the prefix-cache `num_computed_tokens <= num_prompt_tokens` assertion |
| T27 | **Backend auto-selection** | `backends/registry.select_backend` | picks cuda_ipc/rdma/tcp_shm by same/cross-node + availability, with graceful degradation |

### 8.6 Compatibility / environment
| # | Trick | Location | Rationale |
|---|---|---|---|
| T28 | **Clamp util to free VRAM** | `standby._clamp_util_to_free_gpu_memory` | on shared GPUs shrinks util to the minimum free ratio, reserving 8% margin for CUDA-graph warmup + activation buffers, avoiding B-instantiation OOM |
| T29 | **B APIServer NUMA binding** | `standby.run_standby_server` | pins B's data-plane (recv buffer/tmpfs/shm write) to B's NUMA node, spatially isolated from A |
| T30 | `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` | server_demo | engine internal request_id matches external, enabling per-id stream reattach |
| T31 | `VLLM_USE_FLASHINFER_SAMPLER=0` | server_demo | FlashInfer 0.6.17 JIT is incompatible with the local CUDA toolchain, fall back to native sampling |

### 8.7 Experimental apparatus (default off / default-fixed behavior, no semantic change)
| # | Item | Location |
|---|---|---|
| T32 | perf-probe env (`VLLM_HI_PERF_PROBE_*`: `SKIP_D2H`/`SKIP_ENCODE`/`SKIP_TRANSFER`/`SKIP_POLL`; `PINNED`/`FULL_THROTTLE` are "on by default"; `D2H_MODE`=`smooth`/`burst`) | `runtime_config.ProbeConfig` / `kv_collector` / `collector` / `tcp_shm` |

---

## 9. Configuration / Environment Variables

| Env var / CLI | Effect | Default |
|---|---|---|
| `VLLM_HI_TYPE` / `--vllm-hi-type` | instance type A/B | — |
| `VLLM_HI_HOST` / `--vllm-hi-host` | Host scheduler address | — |
| `VLLM_HI_DATA_PORT` / `--vllm-hi-data-port` | data-plane TCP port | 0 |
| `VLLM_HI_TRANSFER_BACKEND` | explicit data-plane backend (`tcp_shm`/`cuda_ipc`/`rdma`) | auto |
| `VLLM_HI_REQUEST_FILTER` | custom request filter class path (`module.Class`) | migrate all |
| `VLLM_HI_MAX_INCR_ROUNDS` | max incremental rounds before the final round | 8 |
| `VLLM_HI_INCR_STALL_ROUNDS` | consecutive non-improving rounds that trigger the final round | 2 |
| `VLLM_HI_KV_COLLECT_CHUNK_BLOCKS` | KV chunked-collection granularity (blocks) | 512 |
| `VLLM_HI_KV_COLLECT_CHUNK_BLOCKS_LARGE` | final-incremental large-chunk granularity (blocks) | 1<<20 |
| `VLLM_HI_SHM_CHUNK_BYTES` | B shm-write chunk size (bytes) | 256MB |
| `VLLM_HI_SHM_YIELD_SECONDS` | B shm-write inter-chunk CPU yield (s) | 0.02 |
| `VLLM_HI_KV_RESTORE_THREADS` | worker KV-restore torch threads | 4 |
| `VLLM_HI_KV_RESTORE_YIELD_SECONDS` | worker KV-restore inter-layer yield (s) | 0.01 |
| `VLLM_HI_D2H_HBM_BUDGET` | D2H throttling HBM budget ratio | 0.25 |
| `VLLM_HI_D2H_BATCH_BLOCKS` | D2H blocks per batch (16 in smooth mode) | 64 |
| `VLLM_HI_PERF_PROBE_SKIP_D2H` | skip D2H copy (experiment) | off |
| `VLLM_HI_PERF_PROBE_SKIP_ENCODE` | skip encode (experiment) | off |
| `VLLM_HI_PERF_PROBE_SKIP_TRANSFER` | skip transfer (experiment) | off |
| `VLLM_HI_PERF_PROBE_SKIP_POLL` | skip completion poll (experiment) | off |
| `VLLM_HI_PERF_PROBE_PINNED` | use pinned pool ("on by default") | on |
| `VLLM_HI_PERF_PROBE_FULL_THROTTLE` | throttle full collection too ("on by default") | on |
| `VLLM_HI_PERF_PROBE_D2H_MODE` | D2H throttling mode `smooth`/`burst` | smooth |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | allow cloudpickle callables over RPC | — |
| `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` | see T30 | — |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | see T31 | — |

---

## 10. Known Limitations / Trade-offs

1. **Default transfer uses CPU relay**: `tcp_shm` is GPU→CPU→shm→TCP→CPU→GPU. Same-node full P2P enables `cuda_ipc` GPU-direct; but **cross GPU architecture (e.g. Blackwell 0-3 / Ada 4-7) has no P2P**, so `cuda_ipc.available()` returns False and it auto-degrades to `tcp_shm`. Cross-node `rdma` is still a skeleton (`implemented=False`, degrades even when explicitly requested).
2. **Incremental convergence vs downtime**: under full load the KV keeps turning over, so incremental volume stops shrinking; the stall detector (T24) mitigates by entering pause+final early, but extreme cases can still degrade to near-full transfer.
3. **SamplingParams cache is bounded**: FIFO cap 256; older entries are evicted across many migrations (only affects deserialization performance, not correctness).
4. **IPC storage reuse across rounds**: `worker._vllm_hi_ipc_storages` caches opened shared VRAM; released by A after migration completes (or naturally on worker exit).
5. **Worker side depends on string RPC contracts**: `_vllm_hi_*` names are centralized in `common/rpc.py`, but still dispatched by string through vLLM's `call_utility_async`/`collective_rpc_async`, so vLLM internal API changes have no compile-time guarantee.
