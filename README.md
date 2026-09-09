# vllm_hi

**vllm_hi** is a drop-in plugin for [vLLM](https://github.com/vllm-project/vllm) that adds **low-downtime live migration** to inference services.

[English](README.md) · [中文文档](docs/readme_cn.md)

## Key advantages

- **~0.1 s** downtime for single long-sequence live migration.
- **~1.5 s** downtime for full-load live migration.

---

## Overview

vllm_hi keeps **standby instance (B)** next to the **active instance (A)**. On migration, it transfers A's KV cache and scheduler state to B, which then takes over the traffic seamlessly — client streaming connections are never dropped.

```
 client ──(SSE streaming)──> Host (SchedulerServer :8002)
                               │  routing + migration orchestration
                               ▼
                         A instance (:8000, active) ──migrate──> B instance (:8001, standby)
                               │                                     │
                               └─── data plane (KV blobs, pluggable) ──┘
```

`import vllm_hi` automatically applies every customization (idempotent). All customization is injected at runtime via the `vllm.general_plugins` entry point and monkey-patching.

---

## Background & Use Cases

### 1. Low-downtime elastic scaling under load

When traffic grows, you often need to **scale out** an instance (e.g. TP=1 → TP=2) to gain more throughput. Naively, this means stopping the old instance and cold-starting a new one — dropping every in-flight request and forcing clients to retry.

vllm_hi migrates the KV cache + scheduler state from the small instance to the larger one **while the old instance keeps serving**. Clients keep streaming, and only experience a transient drop in throughput (not a disconnect) during the ~1 second of switchover.

### 2. Long-sequence migration to relieve head-of-line blocking

A single very long sequence (e.g. 8K+ output tokens) occupies a large share of the KV cache and sits in front of many shorter requests — classic **head-of-line blocking**. The short requests behind it stall even though they need only a fraction of the KV.

With vllm_hi you can **migrate the long sequence onto a dedicated instance**, freeing the main instance to serve short requests with low latency, while the long sequence keeps generating on its own instance.

---

## Key Features

- **Low-downtime live migration** — transfer KV cache + scheduler state; client streaming connections stay alive across the switch.
- **Incremental convergence** — instead of a single huge final transfer, vllm_hi repeatedly collects *only* the blocks that were **added or modified** since the last round (and marks **dropped** blocks), while A keeps decoding. It enters the final round (pause A + last incremental + switch) as soon as any of these hold:
  1. the incremental round count reaches the upper limit (**8** by default, `VLLM_HI_MAX_INCR_ROUNDS`);
  2. **two consecutive rounds** have an incremental block count **no smaller than** the historical minimum (KV has stopped converging);
  3. the incremental block count is **≤ 1**.
- **Pluggable data-plane backends** — `tcp_shm` (CPU relay, cross-node fallback), `cuda_ipc` (GPU-direct, same-node P2P), `rdma` (cross-node GPUDirect, skeleton). Auto-selected by node topology, overridable via `VLLM_HI_TRANSFER_BACKEND`.
- **Topology remapping** — supports TP and PP **scale-out and scale-in** (e.g. TP1→TP2, TP2→TP1, PP1→PP2, PP2→PP1), with head-shard and layer-range remapping.
- **Three migration policies** (`MigrationType`) — `destroy_source` (default), `preserve_source`, `keep_source_active`.
- **Request filter extension point** — migrate only a subset of requests (e.g. long requests) via `VLLM_HI_REQUEST_FILTER`.
- **Zero source modification** — every patch is applied at runtime through the `vllm.general_plugins` entry point.

---

## Architecture

Three roles communicate over a control plane (HTTP/FastAPI) and a data plane (pluggable KV backend):

| Role | Process | Responsibility |
|---|---|---|
| **Host** (`SchedulerServer`) | `:8002` | instance registry, request routing, migration orchestration |
| **A instance** | `:8000` | active inference; exposes migration endpoints (`collect`/`pause`/`resume`/`destroy`/`convert_to_b`) |
| **B instance** | `:8001` | standby; instantiates on demand, receives state, resumes service |

The control-plane wire contract is defined in `common/protocol.py` (Pydantic DTOs); scheduler state travels over HTTP, while the large KV blocks travel over the pluggable data-plane backend.

---

## Installation

### From PyPI

```bash
pip install vllm-hi
```

The package registers itself into the `vllm.general_plugins` entry point, so every vLLM process (process0, engine core, workers) automatically loads the patches.

### From source

```bash
git clone https://github.com/KingVec7r/vLLM_hi
cd vLLM_hi/vllm_hi
pip install -e .
```

> Requires `vllm>=0.13` (the package targets the vLLM `0.13.x` internal APIs) and Python `>=3.10`.

---

## Quick Start

vllm_hi adds three CLI flags to the standard `vllm serve` command:

| Flag | Description |
|---|---|
| `--vllm-hi-type {A,B}` | instance type (A = active, B = standby) |
| `--vllm-hi-host ip:port` | Host scheduler address |
| `--vllm-hi-data-port PORT` | data-plane TCP port |

### 1. Start the Host scheduler

```bash
python -c "from vllm_hi.coordinator.server import SchedulerServer; SchedulerServer(host='0.0.0.0', port=8002).serve_sync()"
```

### 2. Start instance A (active, TP=1)

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve Qwen3-8B \
  --port 8000 --served-model-name vllm_hi/Qwen3-8B \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \
  --vllm-hi-type A --vllm-hi-host 127.0.0.1:8002 --vllm-hi-data-port 9000
```

### 3. Start instance B (standby, will be scaled out to TP=2 on migration)

```bash
CUDA_VISIBLE_DEVICES=2,3 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve Qwen3-8B \
  --port 8001 --served-model-name vllm_hi/Qwen3-8B \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 \
  --vllm-hi-type B --vllm-hi-host 127.0.0.1:8002 --vllm-hi-data-port 9001
```

Both instances register with the Host automatically. A convenience script that launches all three processes is provided at `test/vllm_hi/server_demo.py`.

---

## Usage

All interaction goes through the **Host** at `http://127.0.0.1:8002`.

### Send inference requests (proxied to A transparently)

```bash
curl http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "vllm_hi/Qwen3-8B", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The Host routes by `served_model_name` and proxies the request (including SSE streaming) to the active instance.

### Query the scheduler state

```bash
curl http://127.0.0.1:8002/vllm_hi/status          # Host status + instance count
curl http://127.0.0.1:8002/vllm_hi/instances      # full instance list
curl http://127.0.0.1:8002/v1/models              # served models (OpenAI-compatible)
```

### Trigger a live migration

```bash
curl -X POST http://127.0.0.1:8002/vllm_hi/migrate \
  -H "Content-Type: application/json" \
  -d '{
        "src_vllm_hi_id": "<A-instance-id>",
        "dst_vllm_hi_id": "<B-instance-id>",
        "target_tp": 2,
        "target_pp": 1,
        "target_gpu_memory_utilization": 0.95,
        "migration_type": "destroy_source",
        "tn": 2
      }'
```

`src_vllm_hi_id` / `dst_vllm_hi_id` are the `vllm_hi_id` fields from `GET /vllm_hi/instances`. The response returns a `migration_id` immediately.

### Track migration progress

```bash
curl http://127.0.0.1:8002/vllm_hi/migration/<migration_id>/status
```

The response carries `current_step`, a `step_history`, `data_size_bytes`/`block_count` per round, and the final `new_active_instance`.

---

## Configuration

| Env var / CLI | Description | Default |
|---|---|---|
| `VLLM_HI_TYPE` / `--vllm-hi-type` | instance type `A`/`B` | — |
| `VLLM_HI_HOST` / `--vllm-hi-host` | Host scheduler address | — |
| `VLLM_HI_DATA_PORT` / `--vllm-hi-data-port` | data-plane TCP port | `0` |
| `VLLM_HI_TRANSFER_BACKEND` | data-plane backend `tcp_shm`/`cuda_ipc`/`rdma` | auto |
| `VLLM_HI_REQUEST_FILTER` | custom request filter `module.Class` | migrate all |
| `VLLM_HI_MAX_INCR_ROUNDS` | max incremental rounds before the final round | `8` |
| `VLLM_HI_INCR_STALL_ROUNDS` | consecutive non-improving rounds that trigger the final round | `2` |

A full list of performance-tuning knobs and probe switches is documented in [`features.md`](features.md).

---

## Benchmark Results

Hardware: **3 × NVIDIA RTX PRO 6000 Blackwell 96 GB**, `gpu_memory_utilization = 0.95`.
Model: [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B).
Workload data: `test/data/LongWriter/1024.json`, a subset of [LongWriter-6k](https://www.modelscope.cn/datasets/AI-ModelScope/LongWriter-6k) (the 1024 prompts that produce the longest Qwen3-8B outputs).
`concurrency` = `max_seq_num` = 256, `migration_delay_s` = 10.

### Full-load live migration (`max_tokens = 1024`)

| tag | source (A) | target (B) | downtime (s) | transfer rounds | KV blocks | data size (GB) | raw data |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | tp1pp1 | tp1pp1 | 1.147 | 9  | 10040 | 22.06 | `test/vllm_hi/migration/1024_test.json` |
| a1_btp | tp1pp1 | tp2pp1 | 1.597 | 9  | 17839 | 39.20 | `test/vllm_hi/migration/1024_test_tp_scale_out.json` |
| a1_bpp | tp1pp1 | tp1pp2 | 0.879 | 10 | 17705 | 38.90 | `test/vllm_hi/migration/1024_test_pp_scale_out.json` |

Incremental convergence (KV blocks / GB per round):

| tag | full | incr 1 | incr 2 | incr 3 | incr 4 | incr 5 | incr 6 | incr 7 | incr 8 | incr 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | 4985 / 10.95 | 1727 / 3.79 | 459 / 1.01 | 691 / 1.52 | 409 / 0.90 | 373 / 0.82 | 432 / 0.95 | 464 / 1.02 | 500 / 1.10 | — |
| a1_btp | 11966 / 26.29 | 2017 / 4.43 | 926 / 2.03 | 512 / 1.12 | 451 / 0.99 | 446 / 0.98 | 481 / 1.06 | 558 / 1.23 | 482 / 1.06 | — |
| a1_bpp | 10626 / 23.35 | 3863 / 8.49 | 975 / 2.14 | 338 / 0.74 | 337 / 0.74 | 349 / 0.77 | 292 / 0.64 | 335 / 0.74 | 284 / 0.62 | 306 / 0.67 |

![full-load migration](test/vllm_hi/migration/1024_test_baseline.png)

The figure plots the client-perceived token output rate over time for the three migration curves plus a **Baseline** (no migration event). Clients are unaware of the switchover: comparing `a1_b1` to the baseline, live migration adds only **~7 s** to total task time. More event visualizations are under `test/vllm_hi/migration/*.png`.

### Single long-sequence live migration (`max_tokens = 8192`)

| tag | source (A) | target (B) | downtime (s) | transfer rounds | KV blocks | data size (GB) | raw data |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a1_b1  | tp1pp1 | tp1pp1 | 0.084 | 3 | 260 | 0.571 | `test/vllm_hi/migration/single_test.json` |
| a1_btp | tp1pp1 | tp2pp1 | 0.116 | 3 | 337 | 0.740 | `test/vllm_hi/migration/single_test_tp_scale_out.json` |
| a1_bpp | tp1pp1 | tp1pp2 | 0.091 | 3 | 280 | 0.615 | `test/vllm_hi/migration/single_test_pp_scale_out.json` |
| atp_b1 | tp2pp1 | tp1pp1 | 0.114 | 4 | 377 | 0.828 | `test/vllm_hi/migration/single_test_tp_scale_in.json` |
| app_b1 | tp1pp2 | tp1pp1 | 0.087 | 3 | 255 | 0.560 | `test/vllm_hi/migration/single_test_pp_scale_in.json` |

Incremental convergence (KV blocks / GB per round):

| tag | full | incr 1 | incr 2 | incr 3 |
| --- | --- | --- | --- | --- |
| a1_b1  | 259 / 0.569 | 1 / 0.002 | 0 / 0.000 | — |
| a1_btp | 335 / 0.736 | 1 / 0.002 | 1 / 0.002 | — |
| a1_bpp | 279 / 0.613 | 1 / 0.002 | 0 / 0.000 | — |
| atp_b1 | 374 / 0.822 | 2 / 0.004 | 1 / 0.002 | 0 / 0.000 |
| app_b1 | 254 / 0.558 | 1 / 0.002 | 0 / 0.000 | — |

![single long-sequence migration](test/vllm_hi/migration/single_test_baseline.png)

The figure shows the client-perceived token rate for `a1_b1`, `a1_btp`, `a1_bpp` and the baseline. The client does not notice the switchover: compared to the baseline, live migration adds **no extra inference time** (the single sequence simply keeps decoding on B).

---

## Limitations

- **Downtime depends on the data-plane bandwidth.** The benchmarks above use `cuda_ipc` over PCIe. NVLink raises the performance ceiling; the `rdma` backend enables cross-node migration (the test machine has no RDMA NIC, so the RDMA path is a documented extension point but not implemented); over plain TCP the incremental rounds no longer converge under full load.
- **Data transfer does not slow A's decoding, but some state-transfer steps briefly block the path that returns decoded tokens to clients**, so the client-perceived decode rate dips below the baseline during transfer and then rebounds above it — total task time is unaffected.
- **Full-load scale-in may stall B.** When the target has fewer GPUs than the source (scale-in), the migrated KV blocks can exceed B's KV cache capacity and hang; automatic KV offload to CPU on B is planned but not yet implemented.

---

## License

[Apache-2.0](LICENSE)
