#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vllm_hi server demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def setup_gpu_env(physical_ids: list[int]) -> None:
    """Sets GPU-related environment variables before importing vllm.

    This machine's default CUDA enumeration order does not match the nvidia-smi
    numbering; under ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` they agree, so physical ids
    can be used directly.

    In addition, the P2P channel between cards 4 and 5 on this machine is broken
    (``can_device_access_peer`` is True, but P2P copies corrupt data), which would
    cause NCCL initialization to hang indefinitely, so NCCL P2P is disabled to route
    communication through the host relay path.
    """
    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("[server_demo] set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, physical_ids))
        print(
            "[server_demo] set CUDA_VISIBLE_DEVICES="
            f"{os.environ['CUDA_VISIBLE_DEVICES']} (nvidia-smi physical ids)"
        )
    if "NCCL_P2P_DISABLE" not in os.environ:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        print("[server_demo] set NCCL_P2P_DISABLE=1 (GPU P2P channel broken)")
    # FlashInfer 0.6.17 JIT-compiled sampling.cu is incompatible with the local CUDA
    # toolchain (under Blackwell sm_120, check_cuda_arch reports "requires sm75 or higher"),
    # so the FlashInfer sampler is disabled to fall back to native sampling.
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        print("[server_demo] set VLLM_USE_FLASHINFER_SAMPLER=0 (FlashInfer JIT incompatible)")
    # Blackwell (sm_120) requires CUDA >= 12.8 toolchain
    cuda_home = "/home/dancer/.cache/jr/miniconda3/envs/cuda129"
    if "CUDA_HOME" not in os.environ and os.path.isdir(os.path.join(cuda_home, "bin")):
        os.environ["CUDA_HOME"] = cuda_home
        print("[server_demo] set CUDA_HOME=.../cuda129 (Blackwell sm_120 needs CUDA>=12.8)")


def build_args(demo_opts: argparse.Namespace, extra_vllm_args: list[str]):
    """Builds launch arguments using the vllm official CLI parser."""
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vllm_hi API server demo")
    parser = make_arg_parser(parser)

    cli_args = [
        # positional arg: model
        demo_opts.model,
        "--model",
        demo_opts.model,
        "--host",
        demo_opts.host,
        "--port",
        str(demo_opts.port),
        "--tensor-parallel-size",
        str(demo_opts.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(demo_opts.pipeline_parallel_size),
        "--gpu-memory-utilization",
        str(demo_opts.gpu_memory_utilization),
        "--max-model-len",
        str(demo_opts.max_model_len),
        "--served-model-name",
        demo_opts.served_model_name,
        "--no-enable-log-requests",
        # The P2P channel between cards 4/5 is broken, so the IPC path of custom allreduce is equally unreliable
        "--disable-custom-all-reduce",
    ]
    # Tokenizers for models such as Qwen3 ship with their own chat_template, so none
    # is specified here; --chat-template is only forwarded to vllm when explicitly given.
    # if demo_opts.chat_template:
    #     cli_args += ["--chat-template", demo_opts.chat_template]
    cli_args += list(extra_vllm_args)
    args = parser.parse_args(cli_args)
    # Consistent with the vllm serve command: the positional model arg takes precedence
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)
    return args


def main() -> None:
    demo = argparse.ArgumentParser(
        description="vllm_hi server demo (apply vllm_hi customizations and launch the vllm API)"
    )
    demo.add_argument(
        "--model",
        # default="/data/disk2/jr/weights/Qwen2-0.5B-Instruct",
        default="/data/jr/weights/Qwen3-8B",
        help="path to local model weights",
    )
    demo.add_argument(
        "--gpus", default="3", help="nvidia-smi physical GPU ids (comma-separated)"
    )
    demo.add_argument("--host", default="0.0.0.0")
    demo.add_argument("--port", type=int, default=8000)
    demo.add_argument("--tensor-parallel-size", type=int, default=1)
    demo.add_argument("--pipeline-parallel-size", type=int, default=1)
    demo.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    demo.add_argument("--max-model-len", type=int, default=32768)
    demo.add_argument(
        # "--served-model-name", default="vllm_hi/Qwen2-0.5B-Instruct"
        "--served-model-name", default="vllm_hi/Qwen3-8B"
    )
    # Extra CLI args forwarded to vllm (e.g. --disable-custom-all-reduce)
    demo.add_argument(
        "--vllm-args",
        default="",
        help="extra vllm serve args (string, space-separated)",
    )
    opts = demo.parse_args()
    extra_vllm_args = opts.vllm_args.split() if opts.vllm_args else []

    # ---------- 1. Determine visible GPUs before importing vllm ----------
    physical_ids = [int(x) for x in opts.gpus.split(",")]
    setup_gpu_env(physical_ids)

    # ---------- 2. Import vllm_hi and apply its vllm customizations ----------
    # sys.path[0] of test/server_demo.py is the test/ directory, which contains a
    # same-named test/vllm_hi/ directory (no __init__.py, treated as a namespace
    # package) that would shadow the actually-installed vllm_hi package and prevent
    # the patches from taking effect. Place the real package's parent directory at the
    # front of sys.path to ensure `import vllm_hi` resolves to the real package.
    _vllm_hi_pkg_parent = str(Path(__file__).resolve().parents[1] / "vllm_hi")
    if _vllm_hi_pkg_parent not in sys.path:
        sys.path.insert(0, _vllm_hi_pkg_parent)

    import vllm_hi  # noqa: F401  import applies the patches

    print(
        "[server_demo] vllm_hi imported, customization status: "
        f"{vllm_hi.patches.PATCH_STATUS}"
    )

    # ---------- 3. Build args and launch the vllm API backend ----------
    import uvloop

    from vllm.entrypoints.launchers.api_server.entry import run_server

    args = build_args(opts, extra_vllm_args)
    print(
        "[server_demo] launching vllm API backend: "
        f"model={opts.model} "
        f"tp={opts.tensor_parallel_size} "
        f"pp={opts.pipeline_parallel_size} "
        f"gpu_mem_util={opts.gpu_memory_utilization} "
        f"max_model_len={opts.max_model_len}"
        + (f" extra={extra_vllm_args}" if extra_vllm_args else "")
    )
    uvloop.run(run_server(args))


if __name__ == "__main__":
    sys.exit(main())
