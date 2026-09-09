#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vllm_hi test server demo.

Launches:
- Host scheduler (port 8002)
- A instance (port 8000, GPU 0, TP=1, util=0.97, Qwen3-8B)
- B instance (port 8001, GPU 1,2, standby)

All three register with the Host.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any


def setup_gpu_env(physical_ids: list[int]) -> dict[str, str]:
    """Sets GPU-related environment variables and returns the env dict to apply."""
    env: dict[str, str] = {}
    if "CUDA_DEVICE_ORDER" not in os.environ:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_ids))
    if "NCCL_P2P_DISABLE" not in os.environ:
        env["NCCL_P2P_DISABLE"] = "1"
    # Allow cloudpickle to serialize callables passed over RPC
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # Disable request_id randomization so the engine's internal request_id matches
    # the external one, letting the Host re-attach and resume streams by request_id
    # during the hot-migration switchover.
    env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    # FlashInfer 0.6.17 JIT-compiled sampling.cu is incompatible with the local CUDA
    # toolchain (__builtin_bit_cast undefined), so the FlashInfer sampler is disabled
    # to fall back to native sampling.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Blackwell (sm_120) requires CUDA >= 12.8 toolchain
    cuda_home = "/home/dancer/.cache/jr/miniconda3/envs/cuda129"
    if os.path.isdir(os.path.join(cuda_home, "bin")):
        env["CUDA_HOME"] = cuda_home
    return env


def run_host(port: int = 8002) -> None:
    """Launches the Host scheduler."""
    from vllm_hi.coordinator.server import SchedulerServer

    server = SchedulerServer(host="0.0.0.0", port=port)
    server.serve_sync()


def run_a_instance(args: Any) -> None:
    """Launches the A instance (single card, TP=1, using GPU 0)."""
    # Set up GPU environment
    # Requirement: A uses GPU 0 (single card, TP=1), and B uses GPUs 1,2 during
    # migration, so A and B are placed on disjoint cards without conflict (GPU 2 has
    # other tasks occupying part of its memory, so B uses a low util).
    a_gpus = [int(x) for x in getattr(args, "a_gpus", "0").split(",")]
    for k, v in setup_gpu_env(a_gpus).items():
        os.environ[k] = v

    # Import vllm_hi (apply patches)
    import vllm_hi  # noqa: F401

    # Build args and launch
    import uvloop
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.api_server.entry import run_server

    parser = FlexibleArgumentParser(description="vllm_hi A instance")
    parser = make_arg_parser(parser)

    cli_args = [
        args.model,
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", str(args.a_tp),
        "--pipeline-parallel-size", str(args.a_pp),
        "--gpu-memory-utilization", str(args.a_util),
        "--max-model-len", str(args.max_model_len),
        "--served-model-name", args.served_model_name,
        "--no-enable-log-requests",
        "--disable-custom-all-reduce",
        "--vllm-hi-type", "A",
        "--vllm-hi-host", f"127.0.0.1:{args.host_port}",
        "--vllm-hi-data-port", "9000",
    ]
    parsed = parser.parse_args(cli_args)
    if getattr(parsed, "model_tag", None) is not None:
        parsed.model = parsed.model_tag
    validate_parsed_serve_args(parsed)

    print(f"[A instance] launching A instance: port=8000 GPU={','.join(map(str, a_gpus))} "
          f"TP={args.a_tp} PP={args.a_pp} util={args.a_util}")
    uvloop.run(run_server(parsed))


def run_b_instance(args: Any) -> None:
    """Launches the B instance (using GPUs 1,2, standby)."""
    b_gpus = [int(x) for x in getattr(args, "b_gpus", "1,2").split(",")]
    for k, v in setup_gpu_env(b_gpus).items():
        os.environ[k] = v

    import vllm_hi  # noqa: F401

    import uvloop
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.api_server.entry import run_server

    parser = FlexibleArgumentParser(description="vllm_hi B instance")
    parser = make_arg_parser(parser)

    cli_args = [
        args.model,
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", "8001",
        "--tensor-parallel-size", str(args.b_tp),
        "--pipeline-parallel-size", str(args.b_pp),
        "--gpu-memory-utilization", str(args.b_util),
        "--max-model-len", str(args.max_model_len),
        "--served-model-name", args.served_model_name,
        "--no-enable-log-requests",
        "--disable-custom-all-reduce",
        "--vllm-hi-type", "B",
        "--vllm-hi-host", f"127.0.0.1:{args.host_port}",
        "--vllm-hi-data-port", "9001",
    ]
    parsed = parser.parse_args(cli_args)
    if getattr(parsed, "model_tag", None) is not None:
        parsed.model = parsed.model_tag
    validate_parsed_serve_args(parsed)

    print(f"[B instance] launching B instance: port=8001 GPU={','.join(map(str, b_gpus))} standby")
    uvloop.run(run_server(parsed))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vllm_hi test server (Host + A + B)"
    )
    parser.add_argument(
        "--model", default="/data/jr/weights/Qwen3-8B",
        help="path to model weights",
    )
    parser.add_argument(
        "--served-model-name", default="vllm_hi/Qwen3-8B",
    )
    parser.add_argument("--host-port", type=int, default=8002)
    parser.add_argument(
        "--a-gpus", default="0",
        help="A instance physical GPU ids (comma-separated). A uses card 0, single card, TP=1",
    )
    parser.add_argument(
        "--a-tp", type=int, default=1,
        help="A instance tensor-parallel-size (default 1)",
    )
    parser.add_argument(
        "--a-pp", type=int, default=1,
        help="A instance pipeline-parallel-size (default 1)",
    )
    parser.add_argument(
        "--a-util", type=float, default=0.97,
        help="A instance gpu-memory-utilization (adjust to actual free memory; A's "
        "KV cache capacity must not exceed B's, otherwise the migrated data exceeds "
        "B's capacity and recovery fails)",
    )
    parser.add_argument(
        "--b-gpus", default="1,2",
        help="B instance physical GPU ids (comma-separated, standby, used by the target topology at instantiation)",
    )
    parser.add_argument(
        "--b-tp", type=int, default=2,
        help="B instance standby-registration tensor-parallel-size (overridden by the target topology at instantiation)",
    )
    parser.add_argument(
        "--b-pp", type=int, default=1,
        help="B instance standby-registration pipeline-parallel-size",
    )
    parser.add_argument(
        "--b-util", type=float, default=0.57,
        help="B instance gpu-memory-utilization",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="maximum model sequence length (prompt + output token limit)",
    )
    parser.add_argument(
        "--mode", choices=["all", "host", "a", "b"], default="all",
        help="launch mode: all=launch all three, host/a/b=launch only the corresponding component",
    )
    args = parser.parse_args()

    if args.mode == "host":
        run_host(args.host_port)
        return
    if args.mode == "a":
        run_a_instance(args)
        return
    if args.mode == "b":
        run_b_instance(args)
        return

    # all: launch all three
    print("[server_demo] launching Host scheduler (port 8002)")
    host_proc = mp.Process(
        target=run_host, args=(args.host_port,), name="vllm_hi-host"
    )
    host_proc.start()

    time.sleep(2.0)  # wait for Host to become ready

    print("[server_demo] launching A instance (port 8000, GPU 0)")
    a_proc = mp.Process(
        target=run_a_instance, args=(args,), name="vllm_hi-a"
    )
    a_proc.start()

    print("[server_demo] launching B instance (port 8001, GPU 1,2)")
    b_proc = mp.Process(
        target=run_b_instance, args=(args,), name="vllm_hi-b"
    )
    b_proc.start()

    def _shutdown(signum, frame):
        print("\n[server_demo] signal received, shutting down all processes...")
        for p in (b_proc, a_proc, host_proc):
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[server_demo] all processes started, press Ctrl+C to exit")
    print("  Host: http://0.0.0.0:8002")
    print("  A:    http://0.0.0.0:8000 (GPU 0, TP=1)")
    print("  B:    http://0.0.0.0:8001 (GPU 1,2, standby)")

    try:
        for p in (host_proc, a_proc, b_proc):
            p.join()
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
