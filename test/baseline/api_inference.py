# test/server_demo.py starts a customized development vllm API backend with the vllm_hi customizations injected.

# Uses test/data/filtered_1024_test.json as test data, sending all (1024) inference requests simultaneously with streaming responses. The following metrics are computed from generation:

# - Total time: wall-clock time from sending the first request to the last request completing, in seconds.

# - Instantaneous output token speed: total tokens returned by the service within 5 seconds, divided by 5, as the instantaneous output token speed, in token/s; computed once per second, with the first 4 sampling windows being 1 2 3 4 seconds respectively and 5 seconds afterward.

# - Peak output token speed: the maximum of the instantaneous output token speeds.

# - Average output token speed: total output tokens across all requests, divided by total time, as the average output token speed, in token/s.

# During the run, the instantaneous speed is printed to the terminal once per second; after completion, statistics are saved to test/baseline/server_test_info.json and model outputs to test/baseline/inference_results.json

"""vllm_hi concurrent load-test client (baseline version).

Used with the vllm API backend started by ``test/server_demo.py``:

- Data: ``test/data/filtered_1024_test.json`` (1024 ShareGPT prompts)
- Concurrency: all requests sent simultaneously, streaming (SSE) responses
- Metrics:
  - Total time: wall-clock time (seconds) from sending the first request to the
    last request completing
  - Instantaneous output token speed: computed once per second; the first 4
    sampling windows are 1/2/3/4 seconds respectively, then fixed at 5 seconds
    (tokens returned within the window / window seconds)
  - Peak output token speed: the maximum of the instantaneous output token speeds
  - Average output token speed: total output tokens across all requests / total time
- Outputs:
  - ``test/baseline/server_test_info.json``: statistics
  - ``test/baseline/inference_results.json``: model output for each request

Token counting note: streaming returns text deltas rather than token ids, so the
model tokenizer is used to incrementally encode each SSE chunk to obtain the number
of new tokens in that chunk; the final token count for each request prefers the
``usage`` field at the tail of the SSE stream (exact value).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]  # .../test/
DATA_FILE = BASE_DIR / "data" / "LongWriter/1024.json"
INFO_FILE = BASE_DIR / "baseline" / "LongWriter1024.json"
RESULTS_FILE = BASE_DIR / "baseline" / "inference_results.json"

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "vllm_hi/Qwen3-8B"
DEFAULT_TOKENIZER = "/data/jr/weights/Qwen3-8B"


class Shared:
    """Runtime state shared across threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.t0 = 0.0  # start: when the first request is sent
        self.cum_tokens = 0  # cumulative output tokens received
        self.completed = 0  # number of completed requests
        self.total = 0  # total number of requests
        self.done = False
        # Pre-seed a (0.0, 0) starting point so _tokens_at can interpolate
        # correctly between 0 and the first sample point.
        self.snapshots: list[tuple[float, int]] = [(0.0, 0)]
        self.speeds: list[tuple[float, float]] = []  # [(elapsed, speed)]
        self.failures: list[dict] = []


class IncrementalCounter:
    """Counts tokens for incremental text (each delta encoded independently and summed).

    In vLLM's SSE streaming responses, each content delta is exactly the decoded text
    of one output token; encoding a single delta independently almost always recovers
    exactly 1 token, so summing per delta matches "re-encoding the whole and taking the
    difference" (measured error 0%), while reducing complexity from O(n^2) to O(n) and
    avoiding saturating the client CPU under high concurrency.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.count = 0

    def add(self, content: str) -> int:
        delta = len(self.tokenizer.encode(content, add_special_tokens=False))
        self.count += delta
        return delta


def build_messages(item: dict) -> list[dict]:
    """Converts one ShareGPT sample into chat messages.

    Drops the trailing assistant reference answer so the model continues from the
    last human prompt.
    """
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
    shared: Shared,
    tokenizer: Any,
    item: dict,
    url: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """Sends one streaming request, counting tokens as they arrive, and returns the result."""
    req_id = str(item.get("id", "?"))
    payload = {
        "model": model,
        "messages": build_messages(item),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
        # Ask the server to return the exact usage.completion_tokens at the tail of the SSE stream
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
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
    except Exception as exc:  # noqa: BLE001 - a single failure must not affect the overall load test
        error = f"{type(exc).__name__}: {exc}"
        with shared.lock:
            shared.failures.append({"id": req_id, "error": error})
    finally:
        with shared.lock:
            shared.completed += 1

    elapsed = time.perf_counter() - t0
    completion_tokens = (usage or {}).get("completion_tokens", counter.count)
    return {
        "id": req_id,
        "output": output,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "error": error,
    }


def _tokens_at(snapshots: list[tuple[float, int]], t: float) -> float:
    """Estimates the cumulative token count at time t via linear interpolation."""
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
    """Records a snapshot once per second and refreshes the printed instantaneous speed.

    Instantaneous speed = tokens returned within the window / window seconds;
    the first 4 sampling windows are 1/2/3/4 seconds, then fixed at 5 seconds.
    """
    while True:
        time.sleep(1.0)
        with shared.lock:
            if shared.done:
                break
            elapsed = time.perf_counter() - shared.t0
            cum = shared.cum_tokens
            completed = shared.completed
            shared.snapshots.append((elapsed, cum))
            sample_no = len(shared.snapshots) - 1  # excluding the pre-seeded (0.0, 0)

        # The first 4 sampling windows are 1/2/3/4 seconds, then fixed at 5 seconds.
        window = float(sample_no if sample_no <= 4 else 5)
        start = elapsed - window
        cum_start = _tokens_at(shared.snapshots, start)

        # When the window extends before t<0, the actual usable time is only elapsed, not the full window
        actual_window = elapsed if start < 0 else window
        speed = (cum - cum_start) / actual_window if actual_window > 0 else 0.0

        with shared.lock:
            shared.speeds.append((elapsed, speed))

        line = (
            f"\r[{time.strftime('%H:%M:%S')}] "
            f"instant speed: {speed:>10.2f} token/s "
            f"| window: {actual_window:.2f}s | cumulative output: {cum:>9} tokens "
            f"| completed: {completed}/{shared.total}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()


def save_outputs(results: list[dict], info: dict) -> None:
    INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="vllm_hi concurrent load-test client (baseline)")
    parser.add_argument("--url", default=DEFAULT_URL, help="vllm API address")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name to request")
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER,
        help="tokenizer path used for token counting",
    )
    parser.add_argument("--data", default=str(DATA_FILE), help="test data JSON")
    # generation token length limit
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="maximum generated tokens per request")
    parser.add_argument(
        "--concurrency", type=int, default=256,
        help="number of concurrent requests; 0 means send all at once",
    )
    parser.add_argument(
        "--timeout", type=float, default=180000, help="per-request socket timeout (seconds)"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only take the first N samples (debugging); 0 means all"
    )
    opts = parser.parse_args()

    data_path = Path(opts.data)
    if not data_path.is_file():
        print(f"[api_inference] test data not found: {data_path}", file=sys.stderr)
        return 1
    items = json.loads(data_path.read_text("utf-8"))
    if opts.limit:
        items = items[:opts.limit]
    concurrency = opts.concurrency or len(items)

    print(f"[api_inference] loading tokenizer: {opts.tokenizer}")
    tokenizer = load_tokenizer(opts.tokenizer)

    shared = Shared()
    shared.total = len(items)
    shared.t0 = time.perf_counter()

    monitor = threading.Thread(target=monitor_loop, args=(shared,), daemon=True)
    monitor.start()

    url = opts.url.rstrip("/") + "/v1/chat/completions"
    results: list[dict] = []
    interrupted = False

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = [
            pool.submit(
                run_request, shared, tokenizer, item, url,
                opts.model, opts.max_tokens, opts.timeout,
            )
            for item in items
        ]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "id": "?",
                    "output": "",
                    "completion_tokens": 0,
                    "elapsed": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    except KeyboardInterrupt:
        interrupted = True
        print("\n[api_inference] interrupted, saving completed statistics...")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        t_end = time.perf_counter()
        with shared.lock:
            shared.done = True
    monitor.join(timeout=3.0)

    total_time = t_end - shared.t0
    total_tokens = sum(r.get("completion_tokens", 0) for r in results)
    avg_speed = total_tokens / total_time if total_time > 0 else 0.0
    inst = [
        {"second": i + 1, "speed_token_per_s": round(sp, 2)}
        for i, (_, sp) in enumerate(shared.speeds)
    ]
    peak_speed = max((sp for _, sp in shared.speeds), default=0.0)

    print()
    print("=" * 64)
    print(f"Total time:           {total_time:>10.2f} s")
    print(f"Total output tokens:  {total_tokens:>10}")
    print(f"Avg output token speed: {avg_speed:>10.2f} token/s")
    print(f"Peak output token speed: {peak_speed:>10.2f} token/s")
    print(f"Instant speed samples: {len(inst):>10}")
    print(f"Failed requests:      {len(shared.failures):>10}")
    print("=" * 64)

    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interrupted": interrupted,
        "config": {
            "url": opts.url,
            "endpoint": url,
            "model": opts.model,
            "tokenizer": opts.tokenizer,
            "data_file": str(data_path),
            "max_tokens": opts.max_tokens,
            "concurrency": concurrency,
            "num_requests": len(items),
            "num_failures": len(shared.failures),
        },
        "metrics": {
            "total_time_s": round(total_time, 3),
            "total_output_tokens": total_tokens,
            "avg_output_token_speed_token_per_s": round(avg_speed, 2),
            "peak_output_token_speed_token_per_s": round(peak_speed, 2),
            "instantaneous_output_token_speed_token_per_s": inst,
        },
        "failures": shared.failures,
    }
    save_outputs(results, info)
    print(f"[api_inference] statistics saved to {INFO_FILE}")
    print(f"[api_inference] inference results saved to {RESULTS_FILE}")

    if interrupted:
        # Exit directly to avoid waiting on still-running request threads (atexit would join them)
        os._exit(1)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    sys.exit(main())
