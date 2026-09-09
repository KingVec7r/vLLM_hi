# vllm_hi

**vllm_hi** 是面向 [vLLM](https://github.com/vllm-project/vllm) 的开箱即用插件，为在线推理服务增加**低停机时间的热迁移（live migration）**能力。

[English](../README.md) · [中文文档](readme_cn.md)

## 核心优势

- 单条长序列热迁移停机时间约 **0.1 秒**。
- 满负载全量热迁移停机时间约 **1.5 秒**。

---

## 概述

vllm_hi 在活跃实例（A）旁维护待命实例（B）。迁移时把 A 的 KV Cache 与 scheduler 状态搬到 B，B 接续服务，客户端流式连接不断。

```
 client ──(SSE 流式)──> Host (SchedulerServer :8002)
                          │  路由 + 迁移编排
                          ▼
                    A 实例 (:8000, active) ──迁移──> B 实例 (:8001, standby)
                          │                              │
                          └─── 数据面 (KV 大块, 可插拔) ──┘
```

`import vllm_hi` 即自动应用全部定制（幂等）。所有定制都通过 `vllm.general_plugins` 入口点 + monkey-patch 在运行时注入。

---

## 背景与应用场景

### 1. 低停机时间带负载实例扩缩容

流量增长时，常需要**横向扩容**实例（例如 TP=1 → TP=2）以获得更高吞吐。朴素的方案是停掉旧实例、冷启动新实例——这会丢弃所有在途请求并迫使客户端重试。

vllm_hi 在旧实例**持续服务的同时**，把 KV Cache + scheduler 状态迁移到更大的实例上。客户端持续流式输出，仅在约 1 秒的切换期感知到瞬时吞吐下降（而非断连）。

### 2. 长序列迁移缓解队头堵塞

单条超长序列（例如输出 8K+ token）会占据大量 KV Cache，挡在众多短请求之前——典型的**队头堵塞（head-of-line blocking）**。被挡在后面的短请求即使只需要极少 KV 也会被拖慢。

借助 vllm_hi，可以把长序列**迁移到专用实例**上，让主实例以低延迟服务短请求，而长序列继续在自己的实例上生成。

---

## 核心特性

- **低停机时间热迁移** —— 迁移 KV Cache + scheduler 状态，切换期间客户端流式连接不断。
- **增量收敛** —— 不做一次性的大 final 传输，而是让 A 在持续解码的同时，反复只收集自上一轮以来**新增或修改**的 block（并标记**被丢弃**的 block）。满足以下任一条件即进入最后一轮（暂停 A 解码 + 最后一次增量 + 切换）：
  1. 增量轮次达到上限（默认 **8** 轮，`VLLM_HI_MAX_INCR_ROUNDS`）；
  2. **连续两轮**增量 block 数量**不低于**历史最低值（KV 已不再收敛）；
  3. 增量 block 数量 **≤ 1**。
- **可插拔数据面后端** —— `tcp_shm`（CPU 中转，跨节点兜底）、`cuda_ipc`（GPU 直连，同节点 P2P）、`rdma`（跨节点 GPUDirect，骨架）。按节点拓扑自动选择，可用 `VLLM_HI_TRANSFER_BACKEND` 覆盖。
- **拓扑重映射** —— 支持 TP/PP 的**扩容与缩容**（如 TP1→TP2、TP2→TP1、PP1→PP2、PP2→PP1），含 head 分片与层区间重映射。
- **三种迁移策略**（`MigrationType`）—— `destroy_source`（默认）、`preserve_source`、`keep_source_active`。
- **请求筛选扩展点** —— 通过 `VLLM_HI_REQUEST_FILTER` 只迁移部分请求（如长请求）。
- **零源码修改** —— 所有补丁均通过 `vllm.general_plugins` 入口点在运行时应用。

---

## 架构

三个角色经控制面（HTTP/FastAPI）与数据面（可插拔 KV 后端）通信：

| 角色 | 进程 | 职责 |
|---|---|---|
| **Host**（`SchedulerServer`） | `:8002` | 实例注册表、请求路由、迁移编排 |
| **A 实例** | `:8000` | 活跃推理；暴露迁移端点（`collect`/`pause`/`resume`/`destroy`/`convert_to_b`） |
| **B 实例** | `:8001` | 待命；按需实例化、接收状态、接续服务 |

控制面 wire 契约定义在 `common/protocol.py`（Pydantic DTO）；scheduler 状态走 HTTP，大块 KV 走可插拔数据面后端。

---

## 安装

### 从 PyPI

```bash
pip install vllm-hi
```

该包会注册到 `vllm.general_plugins` 入口点，使所有 vLLM 进程（process0、engine core、worker）都自动加载补丁。

### 从源码

```bash
git clone https://github.com/KingVec7r/vLLM_hi
cd vLLM_hi/vllm_hi
pip install -e .
```

> 依赖 `vllm>=0.13`（本包针对 vLLM `0.13.x` 内部 API）与 Python `>=3.10`。

---

## 快速开始

vllm_hi 给标准 `vllm serve` 命令增加了三个 CLI 参数：

| 参数 | 说明 |
|---|---|
| `--vllm-hi-type {A,B}` | 实例类型（A = 活跃，B = 待命） |
| `--vllm-hi-host ip:port` | Host 调度类地址 |
| `--vllm-hi-data-port PORT` | 数据面 TCP 端口 |

### 1. 启动 Host 调度类

```bash
python -c "from vllm_hi.coordinator.server import SchedulerServer; SchedulerServer(host='0.0.0.0', port=8002).serve_sync()"
```

### 2. 启动 A 类实例（活跃，TP=1）

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve Qwen3-8B \
  --port 8000 --served-model-name vllm_hi/Qwen3-8B \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \
  --vllm-hi-type A --vllm-hi-host 127.0.0.1:8002 --vllm-hi-data-port 9000
```

### 3. 启动 B 类实例（待命，迁移时扩容到 TP=2）

```bash
CUDA_VISIBLE_DEVICES=2,3 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve Qwen3-8B \
  --port 8001 --served-model-name vllm_hi/Qwen3-8B \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
  --vllm-hi-type B --vllm-hi-host 127.0.0.1:8002 --vllm-hi-data-port 9001
```

两个实例都会自动向 Host 注册。启动三者的便捷脚本见 `test/vllm_hi/server_demo.py`。

---

## 使用

所有交互都通过 **Host**（`http://127.0.0.1:8002`）。

### 发送推理请求（透明代理到 A）

```bash
curl http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "vllm_hi/Qwen3-8B", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

Host 按 `served_model_name` 路由，把请求（含 SSE 流式）透明代理到活跃实例。

### 查询调度类状态

```bash
curl http://127.0.0.1:8002/vllm_hi/status          # Host 状态 + 实例数
curl http://127.0.0.1:8002/vllm_hi/instances      # 完整实例列表
curl http://127.0.0.1:8002/v1/models              # 已服务模型（OpenAI 兼容）
```

### 触发热迁移

```bash
curl -X POST http://127.0.0.1:8002/vllm_hi/migrate \
  -H "Content-Type: application/json" \
  -d '{
        "src_vllm_hi_id": "<A实例id>",
        "dst_vllm_hi_id": "<B实例id>",
        "target_tp": 2,
        "target_pp": 1,
        "target_gpu_memory_utilization": 0.95,
        "migration_type": "destroy_source",
        "tn": 2
      }'
```

`src_vllm_hi_id` / `dst_vllm_hi_id` 来自 `GET /vllm_hi/instances` 返回的 `vllm_hi_id` 字段。响应立即返回一个 `migration_id`。

### 跟踪迁移进度

```bash
curl http://127.0.0.1:8002/vllm_hi/migration/<migration_id>/status
```

响应包含 `current_step`、`step_history`、每轮的 `data_size_bytes`/`block_count` 以及最终的 `new_active_instance`。

---

## 配置

| 环境变量 / CLI | 说明 | 默认 |
|---|---|---|
| `VLLM_HI_TYPE` / `--vllm-hi-type` | 实例类型 `A`/`B` | — |
| `VLLM_HI_HOST` / `--vllm-hi-host` | Host 调度类地址 | — |
| `VLLM_HI_DATA_PORT` / `--vllm-hi-data-port` | 数据面 TCP 端口 | `0` |
| `VLLM_HI_TRANSFER_BACKEND` | 数据面后端 `tcp_shm`/`cuda_ipc`/`rdma` | 自动 |
| `VLLM_HI_REQUEST_FILTER` | 自定义请求筛选 `module.Class` | 全量迁移 |
| `VLLM_HI_MAX_INCR_ROUNDS` | 最后一轮前的最大增量轮次 | `8` |
| `VLLM_HI_INCR_STALL_ROUNDS` | 触发最后一轮的连续不收敛轮数 | `2` |

完整的性能调优开关与探针列表见 [`features.md`](features.md)。

---

## 测试结果

硬件：**3 × NVIDIA RTX PRO 6000 Blackwell**，`gpu_memory_utilization = 0.95`。
模型：[Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)。
负载数据：`test/data/LongWriter/1024.json`，是 [LongWriter-6k](https://www.modelscope.cn/datasets/AI-ModelScope/LongWriter-6k) 的子集（实测 Qwen3-8B 输出最长的 1024 条）。
`concurrency` = `max_seq_num` = 256，`migration_delay_s` = 10。

### 满负载热迁移（`max_tokens = 1024`）

| tag | 源(A) | 目的(B) | 停机时间(s) | 传输轮次 | KV block 数 | 数据量(GB) | 原始数据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | tp1pp1 | tp1pp1 | 1.147 | 9  | 10040 | 22.06 | `test/vllm_hi/migration/1024_test.json` |
| a1_btp | tp1pp1 | tp2pp1 | 1.597 | 9  | 17839 | 39.20 | `test/vllm_hi/migration/1024_test_tp_scale_out.json` |
| a1_bpp | tp1pp1 | tp1pp2 | 0.879 | 10 | 17705 | 38.90 | `test/vllm_hi/migration/1024_test_pp_scale_out.json` |

增量收敛（KV block 数 / GB 每轮）：

| tag | 全量 | 增量1 | 增量2 | 增量3 | 增量4 | 增量5 | 增量6 | 增量7 | 增量8 | 增量9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | 4985 / 10.95 | 1727 / 3.79 | 459 / 1.01 | 691 / 1.52 | 409 / 0.90 | 373 / 0.82 | 432 / 0.95 | 464 / 1.02 | 500 / 1.10 | — |
| a1_btp | 11966 / 26.29 | 2017 / 4.43 | 926 / 2.03 | 512 / 1.12 | 451 / 0.99 | 446 / 0.98 | 481 / 1.06 | 558 / 1.23 | 482 / 1.06 | — |
| a1_bpp | 10626 / 23.35 | 3863 / 8.49 | 975 / 2.14 | 338 / 0.74 | 337 / 0.74 | 349 / 0.77 | 292 / 0.64 | 335 / 0.74 | 284 / 0.62 | 306 / 0.67 |

![满负载迁移](../test/vllm_hi/migration/1024_test_baseline.png)

图片呈现三条迁移曲线与 **Baseline**（无迁移事件）下客户端感知的 token 输出速率随时刻变化。客户端对迁移无感知：对比 `a1_b1` 与 Baseline，热迁移使任务完成总耗时增加约 **7 秒**。更多迁移事件可视化见 `test/vllm_hi/migration/*.png`。

### 单条长序列热迁移（`max_tokens = 8192`）

| tag | 源(A) | 目的(B) | 停机时间(s) | 传输轮次 | KV block 数 | 数据量(GB) | 原始数据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | tp1pp1 | tp1pp1 | 0.084 | 3 | 260 | 0.571 | `test/vllm_hi/migration/single_test.json` |
| a1_btp | tp1pp1 | tp2pp1 | 0.116 | 3 | 337 | 0.740 | `test/vllm_hi/migration/single_test_tp_scale_out.json` |
| a1_bpp | tp1pp1 | tp1pp2 | 0.091 | 3 | 280 | 0.615 | `test/vllm_hi/migration/single_test_pp_scale_out.json` |
| atp_b1 | tp2pp1 | tp1pp1 | 0.114 | 4 | 377 | 0.828 | `test/vllm_hi/migration/single_test_tp_scale_in.json` |
| app_b1 | tp1pp2 | tp1pp1 | 0.087 | 3 | 255 | 0.560 | `test/vllm_hi/migration/single_test_pp_scale_in.json` |

增量收敛（KV block 数 / GB 每轮）：

| tag | 全量 | 增量1 | 增量2 | 增量3 |
| --- | --- | --- | --- | --- |
| a1_b1  | 259 / 0.569 | 1 / 0.002 | 0 / 0.000 | — |
| a1_btp | 335 / 0.736 | 1 / 0.002 | 1 / 0.002 | — |
| a1_bpp | 279 / 0.613 | 1 / 0.002 | 0 / 0.000 | — |
| atp_b1 | 374 / 0.822 | 2 / 0.004 | 1 / 0.002 | 0 / 0.000 |
| app_b1 | 254 / 0.558 | 1 / 0.002 | 0 / 0.000 | — |

![单条长序列迁移](../test/vllm_hi/migration/single_test_baseline.png)

图片呈现 `a1_b1`、`a1_btp`、`a1_bpp` 三条曲线与 Baseline 的客户端感知 token 速率。客户端对迁移停机无感知：对比 Baseline，热迁移没有引入额外推理耗时（单条序列直接在 B 上继续解码）。

---

## 局限性

- **停机时间依赖数据面带宽。** 以上测试均使用 `cuda_ipc` 走 PCIe 通道传输；NVLink 能提升性能上限；`rdma` 后端可支持跨节点迁移（测试机无 RDMA 网卡，RDMA 通路未实现但提供了扩展点）；TCP 通路在满负载下增量轮次数据量难以收敛。
- **数据传输不影响 A 的解码速度，但部分状态传输会阻塞 A 端解码 token 返回客户端的通路**，导致客户端接收延时；阻塞缓解后已解码 token 仍会返回，表现为传输时客户端感知解码速率先降至 Baseline 以下再回弹，不会增加总任务耗时。
- **满负载全量缩容会挂起。** 当目标 GPU 少于源（缩容）时，迁移的 KV block 可能超过 B 端 KV Cache 容量而挂起；B 端 KV 自动卸载到 CPU 尚待实现。

---

## 许可证

[Apache-2.0](LICENSE)
