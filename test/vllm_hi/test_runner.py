# SPDX-License-Identifier: Apache-2.0
"""Common runner for hot-migration tests.

Reads test_config.json, launches Host + A + B, runs inference + live migration, and
saves results. Four thin wrapper scripts (api_inference_{tcp,ipc}_{1,2}.py) call
:func:`run_test`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]          # test/
REPO_ROOT = BASE_DIR.parent                              # repo root
CONFIG_FILE = BASE_DIR / "vllm_hi" / "test_config.json"
RESULT_DIR = BASE_DIR / "vllm_hi"                        # performance result directory
INFERENCE_DIR = RESULT_DIR / "results"                   # model inference output directory

DEFAULT_URL = "http://127.0.0.1:8002"

# Maximum model context length (hardcoded 32k): supports long outputs with max_tokens=4096 (prompt + max_tokens must be <= this value).
MAX_MODEL_LEN = 32768

# Single-request test parameters
SINGLE_MAX_TOKENS = 20000       # max generation length 20k
SINGLE_MIGRATION_DELAY_S = 10.0  # trigger migration 10s after sending inference
SINGLE_PROMPT_FILE = BASE_DIR / "vllm_hi" / "single_test_prompt.md"  # single-request test prompt
SINGLE_PROMPT_TOPIC = "The philosophical dilemmas and engineering implementation paths of artificial general intelligence (AGI)"        # value used to fill the topic placeholder
SINGLE_PROMPT_PLACEHOLDER = "[Fill in the complex topic you want to test here, for example: The philosophical dilemmas and engineering implementation paths of artificial general intelligence (AGI)]"


# ---------------------------------------------------------------------------
# config reading
# ---------------------------------------------------------------------------

def read_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# server start / stop
# ---------------------------------------------------------------------------

def start_server(
    config: dict,
    a_cfg: dict | None = None,
    b_cfg: dict | None = None,
) -> subprocess.Popen:
    """Launches server_demo.py (Host + A + B) and returns the background process handle.

    a_cfg / b_cfg override the default A / B topology (used for shrink-class tests where the source A topology is variable).
    """
    a = a_cfg or config["a"]
    b = b_cfg or config["b_test_1"]  # both test B configs share the same GPUs; fall back to b_test_1
    cmd = [
        sys.executable, str(BASE_DIR / "vllm_hi" / "server_demo.py"),
        "--model", config["model"],
        "--served-model-name", config["served_model_name"],
        "--host-port", str(config["host_port"]),
        "--a-gpus", ",".join(map(str, a["gpus"])),
        "--a-tp", str(a.get("tensor_parallel_size", 1)),
        "--a-pp", str(a.get("pipeline_parallel_size", 1)),
        "--a-util", str(a["gpu_memory_utilization"]),
        "--b-gpus", ",".join(map(str, b["gpus"])),
        "--b-tp", str(b.get("tensor_parallel_size", 2)),
        "--b-pp", str(b.get("pipeline_parallel_size", 1)),
        "--b-util", str(b["gpu_memory_utilization"]),
        "--max-model-len", str(MAX_MODEL_LEN),
    ]
    env = os.environ.copy()
    # VLLM_HI_TRANSFER_BACKEND is set in advance by run_test and inherited by the child process.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    return proc


def wait_ready(config: dict, timeout: float = 300.0) -> bool:
    host = f"http://127.0.0.1:{config['host_port']}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(host + "/vllm_hi/instances", timeout=5) as resp:
                insts = json.loads(resp.read())
                if len(insts) >= 2:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Terminates server_demo.py and its child processes, and cleans up leftover GPU processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2)
    for pattern in ("server_demo.py", "VLLM::", "EngineCore"):
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            pass
    # Clean up leftover GPU compute processes
    try:
        pids = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        for pid in pids:
            pid = pid.strip()
            if pid.isdigit():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


# ---------------------------------------------------------------------------
# client (inference + migration + metrics), parameterized from api_inference_1.py
# ---------------------------------------------------------------------------

class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0 = 0.0
        self.t0_unix = 0.0
        self.cum_tokens = 0
        self.completed = 0
        self.total = 0
        self.done = False
        self.last_token_activity = 0.0
        self.snapshots: list[tuple[float, int]] = [(0.0, 0)]
        self.speeds: list[tuple[float, float]] = []
        self.failures: list[dict] = []


class IncrementalCounter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.count = 0

    def add(self, content: str) -> int:
        self.count += 1
        return 1


def build_messages(item: dict) -> list[dict]:
    messages: list[dict] = []
    for turn in item.get("conversations", []):
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages:
        messages = [{"role": "user", "content": ""}]
    return messages


def load_tokenizer(path: str) -> Any:
    from vllm.tokenizers import get_tokenizer
    return get_tokenizer(tokenizer_name=path, trust_remote_code=True)


def run_request(
    shared: Shared, tokenizer: Any, item: dict, url: str, model: str,
    max_tokens: int, timeout: float,
) -> dict:
    req_id = str(item.get("id", "?"))
    payload = {
        "model": model,
        "messages": build_messages(item),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    output = ""
    counter = IncrementalCounter(tokenizer)
    usage = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if not content:
                        continue
                    output += content
                    delta_tokens = counter.add(content)
                    with shared.lock:
                        shared.cum_tokens += delta_tokens
                        shared.last_token_activity = time.perf_counter()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        with shared.lock:
            shared.failures.append({"id": req_id, "error": error})
    finally:
        with shared.lock:
            shared.completed += 1
    elapsed = time.perf_counter() - t0
    completion_tokens = (usage or {}).get("completion_tokens", counter.count)
    return {
        "id": req_id, "output": output,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed, "error": error,
    }


def _tokens_at(snapshots, t):
    if not snapshots or t <= snapshots[0][0]:
        return 0.0
    if t >= snapshots[-1][0]:
        return float(snapshots[-1][1])
    for (e0, c0), (e1, c1) in zip(snapshots, snapshots[1:]):
        if e0 <= t <= e1:
            if e1 == e0:
                return float(c1)
            return c0 + (c1 - c0) * (t - e0) / (e1 - e0)
    return float(snapshots[-1][1])


def monitor_loop(shared: Shared) -> None:
    while True:
        time.sleep(1.0)
        with shared.lock:
            if shared.done:
                break
            elapsed = time.perf_counter() - shared.t0
            cum = shared.cum_tokens
            shared.snapshots.append((elapsed, cum))
            sample_no = len(shared.snapshots) - 1
        window = float(sample_no if sample_no <= 4 else 5)
        start = elapsed - window
        cum_start = _tokens_at(shared.snapshots, start)
        actual_window = elapsed if start < 0 else window
        speed = (cum - cum_start) / actual_window if actual_window > 0 else 0.0
        with shared.lock:
            shared.speeds.append((elapsed, speed))
        sys.stdout.write(
            f"\r[{time.strftime('%H:%M:%S')}] instant speed: {speed:>10.2f} token/s "
            f"| cumulative output: {cum:>9} tokens | completed: {shared.completed}/{shared.total}"
        )
        sys.stdout.flush()


def trigger_migration_concurrent(
    host_url: str, target_tp: int, target_pp: int | None, target_util: float,
    t0: float, t0_unix: float,
    events: list[dict] | None = None, result_holder: dict | None = None,
) -> None:
    if events is None:
        events = []
    if result_holder is None:
        result_holder = {}

    KEY_EVENTS = {
        "migration_trigger", "b_instantiate", "b_instantiate_done",
        "full_transfer_done", "pause_a_done", "switch_service_done", "migration_success",
    }

    def _record(event: str, rel: float | None = None, **extra: Any) -> None:
        if rel is None:
            rel = time.perf_counter() - t0
        entry = {
            "event": event, "second": round(rel, 3),
            "wall": time.strftime("%H:%M:%S"),
            "is_key_event": event in KEY_EVENTS,
        }
        entry.update(extra)
        events.append(entry)
        tag = " ★key" if event in KEY_EVENTS else ""
        print(f"\n[migration event] {event} @ second={rel:.3f}{tag}")

    def _record_from_history(step_history: list[dict], seen_keys: set) -> None:
        for entry in step_history:
            step = entry.get("step", "")
            ts = entry.get("timestamp", 0.0)
            round_idx = entry.get("round_idx")
            key = (step, round_idx)
            if step and key not in seen_keys:
                seen_keys.add(key)
                rel = ts - t0_unix if t0_unix > 0 else (time.perf_counter() - t0)
                extra: dict[str, Any] = {}
                if round_idx is not None:
                    extra["round_idx"] = round_idx
                if entry.get("data_size_bytes") is not None:
                    extra["data_size_bytes"] = entry["data_size_bytes"]
                _record(step, rel=rel, **extra)

    try:
        list_url = f"{host_url}/vllm_hi/instances"
        with urllib.request.urlopen(list_url, timeout=30) as resp:
            instances = json.loads(resp.read().decode("utf-8"))
        print(f"\n[migration] instance list ({len(instances)} instances):")
        for inst in instances:
            print(f"  - id={inst['vllm_hi_id']} type={inst['instance_type']} "
                  f"model={inst['served_model_name']} port={inst['port']}")

        a_inst = next((i for i in instances if i["instance_type"] == "A"), None)
        b_inst = next((i for i in instances if i["instance_type"] == "B"), None)
        if a_inst is None or b_inst is None:
            result_holder["migration_info"] = {"success": False, "error": "A or B instance not found"}
            return

        _record("migration_trigger")
        migrate_url = f"{host_url}/vllm_hi/migrate"
        command = {
            "src_vllm_hi_id": a_inst["vllm_hi_id"],
            "dst_vllm_hi_id": b_inst["vllm_hi_id"],
            "target_tp": target_tp,
            "target_gpu_memory_utilization": target_util,
            "migration_type": "destroy_source",
            "tn": 2,
        }
        if target_pp is not None:
            command["target_pp"] = target_pp

        print(f"[migration] sending migration command: {json.dumps(command, indent=2)}")
        req = urllib.request.Request(
            migrate_url, data=json.dumps(command).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        print(f"[migration] migration submitted: {result}")

        migration_id = result.get("migration_id")
        if not migration_id:
            result_holder["migration_info"] = {"success": False, "error": "no migration_id obtained"}
            return

        mig_start = time.perf_counter()
        seen_keys: set = set()
        while True:
            time.sleep(1.0)
            status_url = f"{host_url}/vllm_hi/migration/{migration_id}/status"
            try:
                with urllib.request.urlopen(status_url, timeout=30) as resp:
                    status = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[migration] failed to query status: {exc}")
                continue
            elapsed = time.perf_counter() - mig_start
            step_history = status.get("step_history", [])
            _record_from_history(step_history, seen_keys)
            print(f"[migration] status ({elapsed:.1f}s): {status.get('status')} "
                  f"step={status.get('current_step', '')}")
            if status.get("status") == "success":
                _record_from_history(step_history, seen_keys)
                _record("migration_success")
                result_holder["migration_info"] = {
                    "success": True, "migration_id": migration_id,
                    "duration": status.get("duration", 0.0), "result": status,
                }
                return
            elif status.get("status") == "failed":
                _record_from_history(step_history, seen_keys)
                result_holder["migration_info"] = {
                    "success": False, "migration_id": migration_id,
                    "error": status.get("error"), "result": status,
                }
                return
            if elapsed > 1800:
                result_holder["migration_info"] = {
                    "success": False, "migration_id": migration_id,
                    "error": "migration timeout (1800s)",
                }
                return
    except Exception as exc:  # noqa: BLE001
        print(f"[migration] exception: {exc}")
        result_holder["migration_info"] = {
            "success": False, "error": f"{type(exc).__name__}: {exc}",
        }


def _compute_bytes_per_block(model_path: str) -> int | None:
    """Computes the byte size of a single KV block from the model's config.json (block_size=16 tokens).

    KV per token per layer: 2(K/V) x num_key_value_heads x head_dim elements,
    multiplied by the dtype byte size and the number of layers, then by the number of
    tokens per block. Returns None if the config cannot be read (the caller skips block
    counting).
    """
    try:
        cfg_path = Path(model_path) / "config.json"
        cfg = json.loads(cfg_path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    num_layers = int(cfg.get("num_hidden_layers", 0))
    num_heads = int(cfg.get("num_attention_heads", 0))
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = cfg.get("head_dim")
    if head_dim is None:
        hidden_size = int(cfg.get("hidden_size", 0))
        head_dim = hidden_size // num_heads if num_heads else 0
    head_dim = int(head_dim or 0)
    dtype = str(cfg.get("torch_dtype", "bfloat16")).replace("torch.", "")
    dtype_bytes = {
        "float16": 2, "half": 2, "bfloat16": 2,
        "float32": 4, "float": 4, "float8_e4m3fn": 1, "float8_e5m2": 1,
        "int8": 1, "uint8": 1,
    }.get(dtype, 2)
    block_tokens = 16  # vLLM default KV cache block size
    if not (num_layers and num_kv_heads and head_dim):
        return None
    return block_tokens * 2 * num_kv_heads * head_dim * dtype_bytes * num_layers


def _extract_transfer_sizes(
    migration_info: dict, bytes_per_block: int | None = None,
) -> dict:
    result = migration_info.get("result", {}) or {}
    step_history = result.get("step_history", []) or []
    full_bytes = None
    increments: list[dict] = []
    for entry in step_history:
        size = entry.get("data_size_bytes")
        if size is None:
            continue
        step = entry.get("step", "")
        if step == "full_transfer_done":
            full_bytes = size
        elif step in ("incremental_transfer_done", "final_incremental_done"):
            increments.append({
                "round": entry.get("round_idx"), "data_size_bytes": size,
            })
    out = {"full_transfer_bytes": full_bytes, "incremental_transfers": increments}
    if bytes_per_block:
        out["bytes_per_block"] = bytes_per_block
        if full_bytes is not None:
            out["full_transfer_blocks"] = round(full_bytes / bytes_per_block)
        for inc in increments:
            inc["block_count"] = round(inc["data_size_bytes"] / bytes_per_block)
    return out


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------

def run_test(
    backend: str,
    test_num: int,
    *,
    single: bool = False,
    a_cfg: dict | None = None,
    b_cfg: dict | None = None,
    name: str | None = None,
) -> int:
    """Runs one test and saves results. backend ∈ {tcp, ipc}, test_num ∈ {1, 2}.

    With single=True it is a single-request test: uses the prompt from
    single_test_prompt.md as the only request, max_tokens=20k, and triggers migration
    10s after sending the inference (used by api_inference_ipc_single_{1,2}.py).

    a_cfg / b_cfg / name override the default source/target topology and result filename (shrink-class tests).
    """
    config = read_config()
    data_path = REPO_ROOT / config["data_file"]
    if not data_path.is_file():
        print(f"test data not found: {data_path}", file=sys.stderr)
        return 1
    all_items = json.loads(data_path.read_text("utf-8"))
    a_cfg = a_cfg or config["a"]
    target = b_cfg or (config["b_test_1"] if test_num == 1 else config["b_test_2"])

    if single:
        prompt = SINGLE_PROMPT_FILE.read_text(encoding="utf-8")
        if SINGLE_PROMPT_PLACEHOLDER in prompt:
            prompt = prompt.replace(SINGLE_PROMPT_PLACEHOLDER, SINGLE_PROMPT_TOPIC)
        items = [{"id": "single_prompt", "conversations": [{"from": "human", "value": prompt}]}]
        max_tokens = SINGLE_MAX_TOKENS
        migration_delay = SINGLE_MIGRATION_DELAY_S
        name = name or f"{backend}_single_{test_num}_result.json"
    else:
        items = all_items
        max_tokens = config["max_tokens"]
        migration_delay = config["migration_delay_s"]
        name = name or f"{backend}_{test_num}_result.json"
    output_file = RESULT_DIR / name

    # 1. Set up the data backend
    # ipc: explicitly select cuda_ipc (GPU direct). A (card 0) / B (cards 2,3) are all
    # Blackwell with P2P available, so it must be explicit; otherwise select_backend in
    # the Host process (which sees all 8 cards) detects cross-architecture
    # (Blackwell<->Ada) no-P2P and falls back to tcp_shm, degrading the ipc test to tcp.
    if backend == "tcp":
        os.environ["VLLM_HI_TRANSFER_BACKEND"] = "tcp_shm"
    else:
        os.environ["VLLM_HI_TRANSFER_BACKEND"] = "cuda_ipc"

    # 2. Start the server
    print(f"[test_runner] backend={backend} test={test_num} starting server...")
    proc = start_server(config, a_cfg, target)
    if not wait_ready(config):
        print("[test_runner] ERROR: server not ready")
        stop_server(proc)
        return 1
    print("[test_runner] server ready")

    try:
        # 3. Run inference + migration
        tokenizer = load_tokenizer(config["model"])
        shared = Shared()
        shared.total = len(items)
        shared.t0 = time.perf_counter()
        shared.t0_unix = time.time()

        monitor = threading.Thread(target=monitor_loop, args=(shared,), daemon=True)
        monitor.start()

        url = DEFAULT_URL.rstrip("/") + "/v1/chat/completions"
        results: list[dict] = []
        migration_events: list[dict] = []
        migration_result_holder: dict = {}

        concurrency = config["concurrency"] or len(items)
        pool = ThreadPoolExecutor(max_workers=concurrency)
        migration_thread: threading.Thread | None = None
        all_sent_time = 0.0
        try:
            futures = [
                pool.submit(
                    run_request, shared, tokenizer, item, url,
                    config["served_model_name"], max_tokens, 1800,
                )
                for item in items
            ]
            all_sent_time = time.perf_counter()

            def _migration_worker(local_all_sent_time: float) -> None:
                delay = migration_delay
                wait = max(0.0, local_all_sent_time + delay - time.perf_counter())
                time.sleep(wait)
                target_pp = target["pipeline_parallel_size"]
                trigger_migration_concurrent(
                    DEFAULT_URL,
                    target_tp=target["tensor_parallel_size"],
                    target_pp=target_pp,
                    target_util=target["gpu_memory_utilization"],
                    t0=shared.t0, t0_unix=shared.t0_unix,
                    events=migration_events,
                    result_holder=migration_result_holder,
                )

            migration_thread = threading.Thread(
                target=_migration_worker, args=(all_sent_time,), daemon=True,
            )
            migration_thread.start()

            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    results.append({
                        "id": "?", "output": "", "completion_tokens": 0,
                        "elapsed": 0.0, "error": f"{type(exc).__name__}: {exc}",
                    })
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            t_end = time.perf_counter()
            with shared.lock:
                shared.done = True
        monitor.join(timeout=3.0)

        if migration_thread is not None and migration_thread.is_alive():
            print("[test_runner] inference finished, migration still in progress, waiting for completion...")
            migration_thread.join(timeout=1500)

        # 4. Aggregate results
        total_time = t_end - shared.t0
        total_tokens = sum(r.get("completion_tokens", 0) for r in results)
        avg_speed = total_tokens / total_time if total_time > 0 else 0.0
        inst = [
            {"second": round(elapsed, 3), "speed_token_per_s": round(sp, 2)}
            for (elapsed, sp) in shared.speeds
        ]
        peak_speed = max((sp for _, sp in shared.speeds), default=0.0)
        migration_info = migration_result_holder.get("migration_info", {})

        KEY_EVENT_NAMES = [
            "migration_trigger", "b_instantiate", "b_instantiate_done",
            "full_transfer_done", "pause_a_done", "switch_service_done", "migration_success",
        ]
        key_migration_events = [
            {"event": e["event"], "second": e["second"], "wall": e.get("wall", "")}
            for e in migration_events if e.get("event") in KEY_EVENT_NAMES
        ]
        with shared.lock:
            last_token_activity_second = round(
                shared.last_token_activity - shared.t0, 3
            ) if shared.last_token_activity > 0 else 0.0
        by_event = {e["event"]: e["second"] for e in key_migration_events}
        trigger_second = by_event.get("migration_trigger")
        b_instantiate_done_second = by_event.get("b_instantiate_done")
        switch_service_done_second = by_event.get("switch_service_done")
        assertions = {
            "migration_trigger_within_load": (
                trigger_second is not None and trigger_second <= last_token_activity_second
            ),
            "b_instantiate_done_within_load": (
                b_instantiate_done_second is not None
                and b_instantiate_done_second <= last_token_activity_second
            ),
            "switch_service_done_within_load": (
                switch_service_done_second is not None
                and switch_service_done_second <= last_token_activity_second
            ),
        }

        info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend,
            "test_num": test_num,
            "config": {
                "url": DEFAULT_URL,
                "model": config["served_model_name"],
                "data_file": config["data_file"],
                "max_tokens": max_tokens,
                "concurrency": concurrency,
                "num_requests": len(items),
                "num_failures": len(shared.failures),
                "migration_delay_s": migration_delay,
                "a_topology": a_cfg,
                "b_topology": target,
            },
            "metrics": {
                "total_time_s": round(total_time, 3),
                "total_output_tokens": total_tokens,
                "avg_output_token_speed_token_per_s": round(avg_speed, 2),
                "peak_output_token_speed_token_per_s": round(peak_speed, 2),
                "instantaneous_output_token_speed_token_per_s": inst,
                "last_token_activity_second": last_token_activity_second,
            },
            "migration_events": migration_events,
            "key_migration_events": key_migration_events,
            "migration": migration_info,
            "transfer_sizes": _extract_transfer_sizes(
                migration_info, _compute_bytes_per_block(config["model"])
            ),
            "assertions": assertions,
            "failures": shared.failures,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\n[test_runner] results saved to {output_file}")

        inference_file = INFERENCE_DIR / name
        inference_file.parent.mkdir(parents=True, exist_ok=True)
        with open(inference_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[test_runner] inference results saved to {inference_file}")
        print(f"  migration duration: {migration_info.get('duration', 'N/A')}s  failures: "
              f"{len(shared.failures)}/{len(items)}  assertions: {list(assertions.values())}")
        return 0
    finally:
        stop_server(proc)


def run_shrink_test(
    backend: str, kind: str, *, single: bool = True,
) -> int:
    """Runs one "topology shrink" hot-migration test.

    kind ∈ {"tp", "pp"}:
    - "tp": A tp=2 pp=1 -> B tp=1 pp=1 (TP shrink)
    - "pp": A tp=1 pp=2 -> B tp=1 pp=1 (PP shrink)

    single=True: a single long inference load (single_test_prompt.md, max_tokens=20k,
    10s trigger migration); single=False: full load (test/data/1024_test.json,
    max_tokens=2048, 30s trigger migration).

    Results are saved to ``{backend}_shrink_{kind}_result.json`` (single) or
    ``{backend}_shrink_{kind}_full_result.json`` (full).
    """
    config = read_config()
    shrink = config.get(f"shrink_{kind}")
    if shrink is None:
        print(f"config missing: shrink_{kind}", file=sys.stderr)
        return 1
    suffix = "" if single else "_full"
    return run_test(
        backend,
        0,
        single=single,
        a_cfg=shrink["a"],
        b_cfg=shrink["b"],
        name=f"{backend}_shrink_{kind}{suffix}_result.json",
    )


__all__ = ["run_test", "run_shrink_test", "read_config"]
