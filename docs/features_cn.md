# vllm_hi 特性总览

> 本文档汇总 vllm_hi 的全部功能特性与性能优化 trick。
> 更新于 2026-09-09，对应源码树 `vllm_hi/vllm_hi/`（约 7400 行，含 `common/` / `coordinator/` / `instance/` / `migration/` / `patches/` 五个包）。

---

## 1. 项目定位

vllm_hi 是一个**不修改 vLLM 源码**、通过 `vllm.general_plugins` 入口点 + monkey-patch 对 vLLM 进行扩展的包，为推理服务提供**低停机时间的热迁移（live migration）**能力：

- 在 A 实例（活跃服务）旁维护 B 实例（待命）。
- 迁移时把 A 的 KV Cache + scheduler 状态搬到 B，B 接续服务，客户端流式连接不断。
- `import vllm_hi` 即自动应用全部定制（幂等）。

---

## 2. 整体架构

三个角色 + 两类通道：

```
客户端 ──(SSE 流式)──> Host (SchedulerServer, :8002)
                          │ 路由 + 迁移编排
                          ▼
                    A 实例 (:8000, active)  ──迁移──>  B 实例 (:8001, standby)
                          │                                    │
                          └── 数据面 (KV 大块, 可插拔后端) ──────┘
```

- **控制面**：HTTP / FastAPI。跨进程共享的请求/响应契约集中定义在 `common/protocol.py`（Pydantic DTO）。
- **数据面**：KV 大块数据经可插拔后端传输（见 §5），scheduler 状态为小数据仍走控制面 HTTP。
  - `tcp_shm`：GPU→CPU(shm)→TCP→CPU(shm)→GPU（默认，跨节点/兜底）。
  - `cuda_ipc`：GPU→GPU 直连（同节点 P2P）。
  - `rdma`：跨节点 GPUDirect（骨架，`implemented=False`）。

**迁移流程**（`coordinator/migration.py`，step 名见 `MigrationStep` 枚举）：

```
validate → b_instantiate → full_transfer → incremental ×N(自适应收敛)
→ pause_a → final_incremental → switch_service → handle_source(destroy/preserve/keep)
```

进度通过 `MigrationResult.current_step` + `step_history` 暴露给客户端（含 `*_done` 变体）。

---

## 3. 模块结构总览

```
vllm_hi/
  common/          类型/配置/工具/契约
    protocol.py      控制面 wire 契约 (Pydantic 请求/响应 DTO)
    types.py         枚举 (Instance/Migration 类型与状态, MigrationStep) + Topology + InstanceInfo
    state.py         VllmHiState (挂在 app.state.vllm_hi 上的实例状态 dataclass)
    parallel.py      全局 worker rank 与 PP 层号换算 (global_rank / worker_global_rank / layer_start_index)
    rpc.py           _vllm_hi_* RPC 契约常量
    utils.py         通用工具 (generate_id/parse_addr/采样参数缓存/NUMA/拓扑提取)
  coordinator/     Host 调度类
    server.py        SchedulerServer (FastAPI: 注册/列表/迁移/状态 + /v1 透传)
    registry.py      InstanceRegistry (线程安全注册表)
    router.py        RequestRouter + InflightStream (流式透传 + 在途流重接)
    migration.py     MigrationCoordinator (迁移编排)
  instance/        A/B 实例
    host_client.py   HostClient (实例 ↔ Host)
    registration.py  A 类自动注册
    activate_endpoint.py  A 端迁移端点 (collect/pause/resume/destroy/convert_to_b)
    standby.py       B 待命启动 + 按需实例化 + 数据面 server + resume_stream + util 钳制
  patches/         monkey-patch + 入口
    __init__.py      门面 (apply + 各 patch_* 函数, 只做注入)
    engine_rpc.py    EngineCore/Worker RPC 方法实现 (状态机)
    plugin.py        vllm.general_plugins 入口 (worker 子进程也生效)
  migration/       状态迁移核心
    backends/        数据面传输后端 (可插拔)
      base.py          KVTransferBackend 抽象 + TransferContext
      registry.py      注册表 + select_backend 自动选择
      tcp_shm.py       TcpStateTransfer (TCP 线协议) + TcpShmTransferBackend
      cuda_ipc.py      CudaIpcTransferBackend (GPU 直连)
      rdma.py          RdmaTransferBackend (骨架)
    collector.py     A 端状态收集编排 (scheduler + KV 分块聚合)
    kv_collector.py  Worker 内后台 KV 收集执行体 (D2H 限速 + shm 编码)
    receiver.py      B 端状态恢复 (scheduler + KV)
    remapper.py      拓扑重映射 (TP head 分片 / PP 层映射)
    restore_worker.py  B worker 内 KV 恢复 (模块级, 可单测)
    resume.py        续传 output_processor
    filter.py        请求筛选扩展点 (RequestFilter)
    runtime_config.py  集中配置 (魔法数字 / 探针开关)
    state_snapshot.py  快照 dataclass (全量/增量)
```

---

## 4. 功能特性清单（按模块）

### 4.1 `common/` — 类型 / 契约 / 工具
| 文件 | 职责 |
|---|---|
| `protocol.py` | 控制面 DTO：注册 / 迁移命令 / 迁移结果 / 状态收集 / 实例化 / 接收通知 / 激活 |
| `types.py` | 枚举 `InstanceType(A/B)`、`InstanceStatus`、`MigrationType`、`MigrationStatus`、`MigrationStep`；`Topology`(TP/PP/util/visible_gpu_ids)、`InstanceInfo` |
| `state.py` | `VllmHiState`：集中保存实例类型/Host 地址/engine_client/host_client/collector/receiver/resumed_queues 等全部实例状态 |
| `parallel.py` | `global_rank(pp,tp,tp_size)` 纯函数 + `worker_global_rank()` + `layer_start_end/index()`（rank 口径全链路统一） |
| `rpc.py` | `_vllm_hi_*` RPC 方法名字符串常量（EngineCore/Worker 注入与数据面引用共用） |
| `utils.py` | `generate_id`/`parse_addr`/`now_iso8601`/`deserialize_sampling_params`(有界缓存)/`get_device_cpu_affinity`/`bind_process_to_device_numa`/拓扑与 IP 提取 helper |

### 4.2 `coordinator/` — Host 调度类
| 文件 | 职责 |
|---|---|
| `server.py` | `SchedulerServer`：`/vllm_hi/register|instances|migrate|migration/{id}/status|status` + `/v1/chat/completions|completions|models` 透传；迁移异步执行 + 进度回调写入 `step_history` |
| `registry.py` | `InstanceRegistry`：`register/deregister/get_by_model/get_by_id/ensure_unique_model_name/update_type/promote_standby_to_active/update_status/update_topology/list_all/count` |
| `router.py` | `RequestRouter` + `InflightStream`：SSE 流式透传；`switch_inflight` 在切换后把在途流重接到 B 的 `resume_stream`；传输错误重查路由重试一次 |
| `migration.py` | `MigrationCoordinator`：编排完整迁移、数据面后端选择、自适应增量收敛、三种 `MigrationType` 的服务切换与源实例处理 |

### 4.3 `instance/` — A/B 实例
| 文件 | 职责 |
|---|---|
| `host_client.py` | `HostClient`：向 Host 注册（失败重试）、查询实例列表 |
| `registration.py` | `auto_register_active_instance`：A 启动后从 args/vllm_config 提取信息自动注册 |
| `activate_endpoint.py` | A 端迁移端点 `collect/pause/resume/destroy/convert_to_b`；`attach_active_endpoints` 经 `state.vllm_hi.app` 反查 app 注册路由 |
| `standby.py` | B 待命启动（跳过引擎实例化，`_StandbyEngineClient` 占位）、按需 `instantiate_engine`、懒启动数据面 server、`resume_stream` 续传端点、`_clamp_util_to_free_gpu_memory` |

### 4.4 `migration/` — 状态迁移核心
| 文件 | 职责 |
|---|---|
| `collector.py` | `StateCollector`：全量/增量收集、请求筛选、**分块收集 + 跨 chunk 聚合**、完成时长估算 |
| `kv_collector.py` | `KvCollector`：Worker 进程内后台线程 + 独立 CUDA stream 收集；`D2HThrottler`/`SmoothD2HThrottler`/`PinnedMemoryPool` |
| `receiver.py` | `StateReceiver`：拓扑重映射 + scheduler 恢复 + KV 恢复（shm 写 + worker 端 GPU 拷贝） |
| `remapper.py` | `TopologyRemapper`：TP 扩展/收缩（head 分片）、PP 变化（层映射），只产出分片描述不复制数据 |
| `restore_worker.py` | `set_kv_cache`：模块级函数（可单测），按 head 分片/block 映射把 shm 数据搬回 GPU |
| `resume.py` | `register_resumed_outputs`：为恢复请求注册 output_processor 并预热 detokenizer |
| `filter.py` | `RequestFilter` / `MigrateAllFilter` / `load_request_filter`（请求筛选扩展点） |
| `runtime_config.py` | `MigrationRuntimeConfig` + `ProbeConfig`（frozen dataclass，集中读 env） |
| `state_snapshot.py` | `BlockTableInfo`/`RequestState`/`KVTensorMeta`/`HeadShard`/`StateSnapshot`/`IncrementalStateUpdate`（数据面 dataclass） |

### 4.5 `patches/` — 运行时补丁
| 补丁 | 作用 |
|---|---|
| `patch_cli_args` | 增加 `--vllm-hi-type/host/data-port` CLI 参数 |
| `patch_build_app` | `GET /vllm_hi/status` 路由 + 所有响应附加 `X-VLLM-HI` 头（**纯 ASGI 中间件**，不破坏 SSE） |
| `patch_chat_completion_logging` | 聊天请求日志（模型/采样参数/来源 IP/最后一条消息前 80 字符） |
| `patch_init_app_state` | 注入 `state.vllm_hi`、挂迁移端点、A 自动注册；存档原始 `init_app_state` 供 B 补挂 serving 状态用 |
| `patch_run_server` | B 走 standby 路径（跳过引擎实例化）；A 额外做 NUMA 绑定 |
| `patch_engine_internals` | 向 `EngineCore`/`Worker` 注入 RPC 方法（scheduler get/set/apply、KV IPC export/set、后台收集 dispatch/try_get） |
| `plugin.py` | `vllm.general_plugins` 入口，确保 worker 子进程也应用 engine_internals 补丁 |

---

## 5. 数据面传输后端（可插拔）

`migration/backends/base.py` 定义 `KVTransferBackend` 抽象（`send_state` / `start_data_server`），元数据 `name`/`gpu_direct`/`cross_node`/`priority`/`implemented`/`available()`。已实现：

| 后端 | `gpu_direct` | 场景 | 说明 |
|---|---|---|---|
| `tcp_shm` | 否 | 跨节点 / 兜底 | GPU→CPU(shm)→TCP→CPU(shm)→GPU |
| `cuda_ipc` | 是 | 同节点 P2P | GPU→GPU 直传 (CUDA IPC handle)，仅 meta 走控制 TCP |
| `rdma` | 是 | 跨节点 GPUDirect | 骨架（`implemented=False`），无 RDMA 硬件时自动降级 |

**自动选择**（`backends/registry.py:select_backend`）：
1. `VLLM_HI_TRANSFER_BACKEND` 显式指定优先（不可用/未实现则告警降级 `tcp_shm`）；
2. 同节点且 P2P 全连接可用 → `cuda_ipc`；
3. 跨节点且 RDMA 可用 → `rdma`；
4. 兜底 `tcp_shm`。

迁移协调器按 A/B 是否同机选择后端并经 `CollectStateRequest`/`ReceiveStateRequest` 下发，A/B 两侧使用同一后端。

---

## 6. 迁移类型（`MigrationType`）

| 类型 | 迁移后行为 |
|---|---|
| `destroy_source` | 销毁 A，B 接管原模型名路由（默认） |
| `preserve_source` | A 转 B 类（待命），B 接管原模型名路由 |
| `keep_source_active` | A 保持 A 类继续处理未迁移请求；B 变新 A 以 `{原名}@{唯一后缀}` 注册（按模型名路由唯一） |

---

## 7. 请求筛选扩展点

`migration/filter.py` 定义 `RequestFilter` 接口（`should_migrate(seq_id, request_state) -> bool`），默认 `MigrateAllFilter` 全量迁移。通过 `VLLM_HI_REQUEST_FILTER=module.Class` 指定自定义筛选器（例如仅迁移长请求）；被筛选掉的请求不收集 KV、不迁移，保留在 A 继续处理。

---

## 8. 性能优化 trick

> 这些是反复调优沉淀下来的技巧，按「传输链路各环节」归类。

### 8.1 收集端（A）—— 降低对推理的干扰
| # | Trick | 位置 | 原理 / 实测收益 |
|---|---|---|---|
| T1 | **只收集已用块** | `collector.collect_full` | 仅收集 block_table 实际引用的物理块，而非全部预分配块，大幅减少 D2H 与传输量 |
| T2 | **分块收集** | `collector._collect_worker_kv_cache_chunked` | 512 block/chunk（`VLLM_HI_KV_COLLECT_CHUNK_BLOCKS`）；final 增量（A 已暂停）用 1<<20 大块一次收集，消除多轮 RPC 与聚合拷贝 |
| T3 | **后台线程 + 独立 CUDA stream** | `kv_collector.KvCollector` | GPU→CPU 拷贝与 CPU 编码写 shm 不阻塞 `execute_model`，推理与搬运完全并行（单活跃任务模型） |
| T4 | **Smooth D2H 限速（event 反馈闭环）** | `kv_collector.SmoothD2HThrottler` | 依据 `Event.elapsed_time` 实测耗时反推瞬时带宽，与预算（默认 HBM 峰值 25%）比较后动态让渡，瞬时带宽收敛到预算附近，不抢推理 matmul 的 HBM 读带宽 |
| T5 | **Pinned 内存池** | `kv_collector.PinnedMemoryPool` | 预分配 pinned buffer，`copy_` 真异步（无 pageable 隐藏 staging 分配与同步回拷）；分配失败回退 pageable |
| T6 | **连续区间切片（非 fancy-index）** | `kv_collector.slice_blocks` | `t[start:end]` 零拷贝 view，消除 `t[block_ids]` 的 gather kernel 与 GB 级显存临时分配 |
| T7 | **CPU 线程数仅在 A 暂停时提升** | `kv_collector.collect_cpu_threads` | final 增量（A 已停）才 boost 到 8~16 线程（单线程 ~1.6GB/s → 8~16 线程 ~18GB/s），服务期间保持单线程避免争用 |
| T8 | **轮询退避 + 完成时长估算首睡** | `collector._collect_worker_kv_cache` | 首次轮询前按经验速率（20GB/s）睡到接近完成，压缩 RPC 插队次数；后续 20ms→200ms 指数退避 |
| T9 | **SamplingParams 反序列化有界缓存** | `utils.deserialize_sampling_params` | FIFO 缓存（256）避免逐请求 pickle.loads 重复反序列化同一对象；返回浅拷贝防 per-request 修改污染 |
| T10 | **显式 `del` 释放，不依赖 gc.collect()** | `collector.release_kv_cache_shm` | 显式释放 memoryview/tensor 引用使 shm.close() 不抛 BufferError，避免 stop-the-world gc 持有 GIL 阻塞事件循环数秒 |

### 8.2 编码 / 聚合（A）
| # | Trick | 位置 | 原理 / 实测收益 |
|---|---|---|---|
| T11 | **torch.cat 聚合（放线程池）** | `collector._aggregate_kv_chunks` | ~23GB/s，替代 numpy slice 赋值 ~1.85GB/s；在 `run_in_executor` 执行（大块拷贝释放 GIL），不阻塞事件循环 |
| T12 | **连续内存 buffer + 元数据（去 pickle）** | `kv_collector._compute_metas` / `collector._assemble_kv_cache` | 每层张量以 `byte_offset/shape/dtype/nbytes` 描述，`torch.frombuffer` 零拷贝重建，替代 pickle 序列化大张量 |

### 8.3 恢复端（B）
| # | Trick | 位置 | 原理 / 实测收益 |
|---|---|---|---|
| T13 | **madvise(MADV_POPULATE_WRITE)** | `receiver._populate_shm_write` | 预分配 shm 物理页，消除写时 page-fault 风暴（1.6GB/s → ~10GB/s） |
| T14 | **numpy 单线程分块 memcpy 写 shm** | `receiver._restore_kv_cache` | ~10GB/s warm；替代 torch `copy_` 多线程（OpenMP spin 与 A 端推理争 CPU），块间 `sleep` 让渡 CPU |
| T15 | **NUMA 绑定写 shm** | `receiver._restore_kv_cache` / `utils.get_device_cpu_affinity` | 写 shm 进程绑定到 B 端 GPU 所在 NUMA，与 A 端推理空间隔离，消除跨 NUMA 内存/带宽争用 |
| T16 | **先切 head 再搬 GPU** | `restore_worker.set_kv_cache` | TP 扩展/收缩时每 rank 只搬自己负责的 head 段，消除冗余搬移 |
| T17 | **只搬已用块 + 分片控制临时显存** | `restore_worker.set_kv_cache` | 只把已用块搬 GPU（高 util 时防整层搬移 OOM），每片 512 块控制 GPU 临时占用 |
| T18 | **恢复线程数限 4 + 层间让渡** | `restore_worker.set_kv_cache` | 限制 torch 线程数（`VLLM_HI_KV_RESTORE_THREADS`=4），层间 `sleep` 让渡 CPU/PCIe 带宽给 A 端 |

### 8.4 传输（数据面）
| # | Trick | 位置 | 原理 |
|---|---|---|---|
| T19 | **recv_into + 大块 sendall + SO_SNDBUF** | `backends/tcp_shm.py` | `recv_into` 复用 buffer 消除 bytearray 重分配；`SO_SNDBUF`=64MB、`TCP_NODELAY`；`sendall` 在 executor 执行 |
| T20 | **ACK 握手** | `backends/tcp_shm.send_state` | A 阻塞等 B 端「已应用」ACK，修复最终增量与 activate 的竞态（fire-and-forget 时 final 数据可能在 activate 后才应用） |
| T21 | **asyncio.Lock 串行化重建 + data_idle** | `backends/tcp_shm.start_data_server` | 串行化 full/incremental 状态重建，维护 `data_idle` 事件，activate 端点据此等待数据面处理完成 |
| T22 | **CUDA IPC 直连 + 连续区间整段 memcpy** | `backends/cuda_ipc.py` / `engine_rpc.worker_set_kv_ipc` | KV 经 IPC handle GPU→GPU 直传（不经 CPU）；block 映射分组成「a/b 都连续」的区间整段 memcpy（离散 gather 实测 ~1.5GB/s） |
| T23 | **IPC storage 缓存** | `engine_rpc.worker_set_kv_ipc` | 跨轮次缓存已打开的共享显存，避免 `cudaErrorInvalidResourceHandle`（同一 cudaIpcMemHandle 每设备只能 open 一次） |

### 8.5 调度 / 迁移编排
| # | Trick | 位置 | 原理 |
|---|---|---|---|
| T24 | **自适应增量收敛（停滞检测）** | `migration.MigrationCoordinator._run_incremental_rounds` | 多轮增量直到「连续 `VLLM_HI_INCR_STALL_ROUNDS`(=2) 轮数据量不低于历史最低」或「`VLLM_HI_MAX_INCR_ROUNDS`(=8) 轮」或「单轮 <=1 block」即提前 pause+final，把尽量多数据传输放到 pause 前 |
| T25 | **Connection: close 短连接** | `migration.MigrationCoordinator._post` | 避免复用被 keep-alive 关闭的 stale 连接导致请求挂起 |
| T26 | **请求恢复 token 对齐** | `engine_rpc.resolve_restore_tokens` | 恢复 decode 请求时分配全部块 + `computed=num_tokens-1`，绕过 prefix-cache 的 `num_computed_tokens <= num_prompt_tokens` 断言 |
| T27 | **后端自动选择** | `backends/registry.select_backend` | 按同节点/跨节点 + 可用性自动选 cuda_ipc/rdma/tcp_shm，未实现/不可用后端优雅降级 |

### 8.6 兼容性 / 环境
| # | Trick | 位置 | 原理 |
|---|---|---|---|
| T28 | **clamp util 到空闲显存** | `standby._clamp_util_to_free_gpu_memory` | 共享 GPU 时按最小空闲比例收缩 util，预留 8% margin 给 CUDA graphs warmup + 激活缓冲区，避免 B 实例化 OOM |
| T29 | **B 端 APIServer NUMA 绑定** | `standby.run_standby_server` | B 端数据面（recv buffer/tmpfs/写 shm）固定到 B 侧 NUMA，与 A 端推理空间隔离 |
| T30 | `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` | server_demo | engine 内部 request_id 与外部一致，便于按 id 重接续传流 |
| T31 | `VLLM_USE_FLASHINFER_SAMPLER=0` | server_demo | FlashInfer 0.6.17 JIT 与本地 CUDA 工具链不兼容，回退原生采样 |

### 8.7 试验装置（默认关闭/默认修复后行为，不改语义）
| # | 项 | 位置 |
|---|---|---|
| T32 | 性能探针 env（`VLLM_HI_PERF_PROBE_*`：`SKIP_D2H`/`SKIP_ENCODE`/`SKIP_TRANSFER`/`SKIP_POLL`；`PINNED`/`FULL_THROTTLE` 为「默认开启」型；`D2H_MODE`=`smooth`/`burst`） | `runtime_config.ProbeConfig` / `kv_collector` / `collector` / `tcp_shm` |

---

## 9. 配置项 / 环境变量

| 环境变量 / CLI | 作用 | 默认 |
|---|---|---|
| `VLLM_HI_TYPE` / `--vllm-hi-type` | 实例类型 A/B | 无 |
| `VLLM_HI_HOST` / `--vllm-hi-host` | Host 调度类地址 | 无 |
| `VLLM_HI_DATA_PORT` / `--vllm-hi-data-port` | 数据面 TCP 端口 | 0 |
| `VLLM_HI_TRANSFER_BACKEND` | 显式指定数据面后端（`tcp_shm`/`cuda_ipc`/`rdma`） | 自动 |
| `VLLM_HI_REQUEST_FILTER` | 自定义请求筛选类全名（`module.Class`） | 全量迁移 |
| `VLLM_HI_MAX_INCR_ROUNDS` | 最后一轮前最大增量轮次 | 8 |
| `VLLM_HI_INCR_STALL_ROUNDS` | 连续多少轮增量不低于历史最低即提前进最后一轮 | 2 |
| `VLLM_HI_KV_COLLECT_CHUNK_BLOCKS` | KV 分块收集粒度（块） | 512 |
| `VLLM_HI_KV_COLLECT_CHUNK_BLOCKS_LARGE` | final 增量大块收集粒度（块） | 1<<20 |
| `VLLM_HI_SHM_CHUNK_BYTES` | B 端写 shm 分块大小（字节） | 256MB |
| `VLLM_HI_SHM_YIELD_SECONDS` | B 端写 shm 块间让渡 CPU 时长（秒） | 0.02 |
| `VLLM_HI_KV_RESTORE_THREADS` | worker 恢复 KV 的 torch 线程数 | 4 |
| `VLLM_HI_KV_RESTORE_YIELD_SECONDS` | worker 恢复 KV 层间让渡时长（秒） | 0.01 |
| `VLLM_HI_D2H_HBM_BUDGET` | D2H 限速的 HBM 预算比 | 0.25 |
| `VLLM_HI_D2H_BATCH_BLOCKS` | D2H 每批 block 数（smooth 模式为 16） | 64 |
| `VLLM_HI_PERF_PROBE_SKIP_D2H` | 跳过 D2H 拷贝（试验） | off |
| `VLLM_HI_PERF_PROBE_SKIP_ENCODE` | 跳过编码（试验） | off |
| `VLLM_HI_PERF_PROBE_SKIP_TRANSFER` | 跳过传输（试验） | off |
| `VLLM_HI_PERF_PROBE_SKIP_POLL` | 跳过完成轮询（试验） | off |
| `VLLM_HI_PERF_PROBE_PINNED` | 用 pinned 内存池（「默认开启」型） | on |
| `VLLM_HI_PERF_PROBE_FULL_THROTTLE` | 全量收集也纳入限速（「默认开启」型） | on |
| `VLLM_HI_PERF_PROBE_D2H_MODE` | D2H 限速模式 `smooth`/`burst` | smooth |
| `VLLM_ALLOW_INSECURE_SERIALIZATION=1` | 允许 cloudpickle 经 RPC 传 callable | 无 |
| `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` | 见 T30 | 无 |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | 见 T31 | 无 |

---

## 10. 已知限制 / 权衡

1. **默认传输走 CPU 中转**：`tcp_shm` 是 GPU→CPU→shm→TCP→CPU→GPU。同节点且 P2P 全连接时 `cuda_ipc` 可 GPU 直连；但**跨 GPU 架构（如 Blackwell 0-3 / Ada 4-7）无 P2P**，`cuda_ipc` 的 `available()` 探针会返回 False，自动降级 `tcp_shm`。跨节点 `rdma` 仍为骨架（`implemented=False`，显式指定也会降级）。
2. **增量收敛 vs 停机时长**：满负载下 KV 持续周转时增量数据量不再收敛，靠停滞检测（T24）提前进入 pause+final 缓解，但极端场景仍可能退化为近似全量。
3. **SamplingParams 缓存有界**：FIFO 上限 256，跨多次迁移累积不同采样参数时旧的被淘汰（仅影响反序列化性能，不影响正确性）。
4. **IPC storage 跨轮次复用**：`worker._vllm_hi_ipc_storages` 缓存已打开共享显存，迁移完成后由 A 端释放（或随 worker 退出自然释放）。
5. **worker 侧依赖字符串 RPC 契约**：`_vllm_hi_*` 方法名由 `common/rpc.py` 集中声明，但仍靠 vLLM 的 `call_utility_async`/`collective_rpc_async` 按字符串分发，vLLM 内部接口变更时无编译期保障。
