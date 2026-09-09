# SPDX-License-Identifier: Apache-2.0
"""KvCollector: in-worker-process background KV collection executor.

Offloads KV Cache state collection during live migration to a dedicated worker
background thread + a dedicated CUDA stream, so GPU→CPU copy and CPU encode-to-shm
no longer block ``execute_model``; inference compute and state transfer run fully
in parallel. This is the sole (default) KV collection implementation, with no
synchronous fallback path.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Iterator

import torch

from vllm.logger import init_logger

from vllm_hi.common.parallel import layer_start_index, worker_global_rank
from vllm_hi.migration.runtime_config import PROBE_CONFIG

logger = init_logger("vllm.vllm_hi.migration.kv_collector")


def collect_cpu_threads() -> int:
    """Suggested torch CPU thread count for the collection phase (briefly raised only when A is paused).

    In steady state vLLM sets the worker's torch thread count to 1 (avoiding OMP
    spin-wait contention), but the GB-scale GPU→CPU copy and torch.cat encode are
    memory-bandwidth-bound: a single thread reaches only ~1.6GB/s, while 8~16
    threads reach ~18GB/s. Only raise the thread count briefly when A is paused
    (final increment) to shorten downtime; during A's serving period (full/
    incremental) keep a single thread to avoid contention.
    """
    return max(8, min(16, (os.cpu_count() or 16) // 2))


def slice_blocks(block_ids: list[int]) -> list[tuple[int, int]]:
    """Partition ascending block_ids into contiguous ranges, returning ``[(start, end), ...]``.

    Within a range use ``t[start:end]`` to slice directly (dim 0 contiguous, zero-copy view); the caller D2H's each range and aggregates on the CPU side by ``byte_offset``. Compared to fancy-indexing ``t[block_ids]``, this eliminates the GPU gather kernel and GB-scale temporary VRAM allocation that contend for SM/HBM bandwidth.
    """
    n = len(block_ids)
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and block_ids[j] == block_ids[j - 1] + 1:
            j += 1
        ranges.append((block_ids[i], block_ids[j - 1] + 1))
        i = j
    return ranges


# Environment variables and defaults for D2H rate limiting.
_VLLM_HI_D2H_HBM_BUDGET = "VLLM_HI_D2H_HBM_BUDGET"
_VLLM_HI_D2H_BATCH_BLOCKS = "VLLM_HI_D2H_BATCH_BLOCKS"
_DEFAULT_HBM_BUDGET_RATIO = 0.25
_DEFAULT_BATCH_BLOCKS = 64
# Empirical HBM peak-bandwidth fallback (bytes/s) when get_device_properties is unavailable.
_FALLBACK_HBM_PEAK_BW = 2000e9


def _parse_env_float(name: str, default: float) -> float:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val


def _parse_env_int(name: str, default: int) -> int:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val


def _hbm_peak_bandwidth() -> float:
    """Estimate the current GPU's HBM peak bandwidth (bytes/s).

    ``memory_clock_rate`` is in kHz, ``memory_bus_width`` in bits; HBM is DDR so
    multiply by 2. Falls back to an empirical constant when
    ``get_device_properties`` fails.
    """
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        peak = (
            props.memory_clock_rate
            * 1e3
            * (props.memory_bus_width / 8.0)
            * 2.0
        )
        if peak > 0:
            return peak
    except Exception:  # noqa: BLE001
        pass
    return _FALLBACK_HBM_PEAK_BW


class D2HThrottler:
    """D2H chunked throttler: split into fixed-block batches and yield HBM bandwidth between batches.

    Splits the KV Cache D2H copy by ``batch_blocks`` and yields HBM bandwidth to
    inference kernels between batches via the ``yield_interval`` gap, so state
    transfer and inference matmul do not starve each other on HBM read bandwidth.
    Parameters can be overridden via the ``VLLM_HI_D2H_HBM_BUDGET`` /
    ``VLLM_HI_D2H_BATCH_BLOCKS`` environment variables.
    """

    def __init__(
        self,
        hbm_budget_ratio: float | None = None,
        batch_blocks: int | None = None,
        peak_hbm_bw: float | None = None,
    ) -> None:
        ratio = _parse_env_float(
            _VLLM_HI_D2H_HBM_BUDGET,
            hbm_budget_ratio
            if hbm_budget_ratio is not None
            else _DEFAULT_HBM_BUDGET_RATIO,
        )
        blocks = _parse_env_int(
            _VLLM_HI_D2H_BATCH_BLOCKS,
            batch_blocks if batch_blocks is not None else _DEFAULT_BATCH_BLOCKS,
        )
        # Fall back to defaults on invalid parameters.
        if ratio <= 0 or ratio > 1:
            ratio = _DEFAULT_HBM_BUDGET_RATIO
        if blocks <= 0:
            blocks = _DEFAULT_BATCH_BLOCKS
        self._hbm_budget_ratio = ratio
        self._batch_blocks = blocks
        self._peak_hbm_bw = (
            peak_hbm_bw if peak_hbm_bw is not None else _hbm_peak_bandwidth()
        )

    @property
    def hbm_budget_ratio(self) -> float:
        return self._hbm_budget_ratio

    @property
    def batch_blocks(self) -> int:
        return self._batch_blocks

    def batches(
        self, ranges: list[tuple[int, int]]
    ) -> Iterator[list[tuple[int, int]]]:
        """Split a list of contiguous ranges into D2H batches by ``batch_blocks``.

        Each range ``(start, end)`` contains ``end - start`` blocks; they are
        accumulated by block count, so a single range may span multiple batches.
        An empty range list yields an empty iteration.
        """

        batch: list[tuple[int, int]] = []
        count = 0
        for start, end in ranges:
            s = start
            while s < end:
                take = min(self._batch_blocks - count, end - s)
                batch.append((s, s + take))
                count += take
                s += take
                if count >= self._batch_blocks:
                    yield batch
                    batch = []
                    count = 0
        if batch:
            yield batch

    def yield_interval(self, batch_bytes: int) -> float:
        """Return the non-negative batch-gap seconds to yield after this batch's D2H.

        Derived from the HBM budget bandwidth (``hbm_budget_ratio`` × peak
        bandwidth): the yield window is the time it would take to transfer this
        batch at the budget bandwidth, keeping D2H average bandwidth within budget.
        """
        if batch_bytes <= 0:
            return 0.0
        budget_bw = self._hbm_budget_ratio * self._peak_hbm_bw
        if budget_bw <= 0:
            return 0.0
        return max(0.0, batch_bytes / budget_bw)


class SmoothD2HThrottler(D2HThrottler):
    """Sustained low-rate D2H throttler: smooths instantaneous bandwidth via an event feedback loop.

    Difference from ``D2HThrottler``: the latter controls time-averaged bandwidth
    via "peak burst + static sleep", so the burst still seizes HBM read bandwidth
    at peak and starves inference matmul. This class uses finer-grained chunks
    (default 16 blocks/batch) and infers instantaneous bandwidth from the measured
    per-batch D2H time (``torch.cuda.Event.elapsed_time``); comparing against the
    budget bandwidth it returns a dynamic yield gap, converging D2H instantaneous
    bandwidth to near budget and interleaving evenly with inference kernels.
    """

    def __init__(
        self,
        hbm_budget_ratio: float | None = None,
        batch_blocks: int | None = None,
        peak_hbm_bw: float | None = None,
    ) -> None:
        super().__init__(
            hbm_budget_ratio=hbm_budget_ratio,
            batch_blocks=batch_blocks if batch_blocks is not None else 16,
            peak_hbm_bw=peak_hbm_bw,
        )

    def next_gap(self, elapsed_ms: float, batch_bytes: int) -> float:
        """Return the non-negative gap seconds to yield after this batch's D2H (event feedback loop).

        Infer instantaneous bandwidth from the measured ``elapsed_ms``: if the
        actual run is faster than the budget-bandwidth time, yield
        "budget time − actual time" to keep overall average bandwidth at
        ``hbm_budget_ratio`` × peak bandwidth; if already slower than budget, no
        further yield.
        """
        if batch_bytes <= 0:
            return 0.0
        budget_bw = self._hbm_budget_ratio * self._peak_hbm_bw
        if budget_bw <= 0:
            return 0.0
        budget_time = batch_bytes / budget_bw
        actual_time = max(0.0, elapsed_ms) / 1000.0
        return max(0.0, budget_time - actual_time)


class PinnedMemoryPool:
    """Reusable pinned memory pool: preallocates a pinned buffer for D2H to land directly.

    Eliminates the pinned staging buffer PyTorch implicitly allocates when copying
    with ``to("cpu", non_blocking=True)`` into pageable memory and the synchronous
    copy-back: each D2H no longer re-cudaHostAlloc's, and ``copy_`` into a pinned
    target can be truly asynchronous (executed on copy_stream).
    When allocation fails (huge pages / locked-memory limits), it is unavailable
    and the caller falls back to the pageable path.
    """

    def __init__(self, nbytes: int) -> None:
        self._buf: torch.Tensor | None = None
        self._nbytes = nbytes
        if nbytes <= 0:
            return
        try:
            self._buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa: BLE001
            self._buf = None

    @property
    def available(self) -> bool:
        return self._buf is not None

    @property
    def buffer(self) -> torch.Tensor:
        """Return the underlying pinned buffer (``None`` when ``available=False``)."""
        return self._buf

    def acquire(self) -> torch.Tensor | None:
        """Lend out the whole pinned buffer; return ``None`` when unavailable."""
        return self._buf

    def release(self, buf: torch.Tensor | None = None) -> None:
        """Return the buffer (single-buffer model, stateless, kept for API compatibility)."""
        return None


class KvCollector:
    """In-worker-process background KV collection executor.

    Holds a dedicated CUDA stream and a daemon background thread. ``dispatch``
    only does ``record_event`` + enqueue on the inference thread (microseconds);
    the background thread asynchronously performs the D2H copy and CPU encode-to-shm
    on copy_stream, hardware-parallel with inference kernels on the compute stream.

    Uses a "single active task" model: migration collection is serial, at most one
    collection task at a time, and ``try_get_result`` can re-read the most recently
    completed result until the next ``dispatch``.
    """

    def __init__(self, model_runner: Any, maxsize: int = 4) -> None:
        kv_caches = getattr(model_runner, "kv_caches", [])
        if not kv_caches:
            raise ValueError("model_runner.kv_caches is unavailable")
        self._model_runner = model_runner
        self._kv_caches = kv_caches
        device = getattr(model_runner, "device", None)
        self._device = device if device is not None else torch.device("cuda:0")
        self._copy_stream = torch.cuda.Stream(device=self._device)
        self._throttler = D2HThrottler()
        self._smooth_throttler = SmoothD2HThrottler()
        # Pinned memory pool: capacity estimated as batch_blocks blocks × per-block
        # bytes, covering the peak data volume of one chunked D2H; on allocation
        # failure (locked-memory limits) fall back to the pageable path.
        self._pinned_pool = PinnedMemoryPool(
            self._smooth_throttler.batch_blocks * self._block_bytes()
        )
        self._task_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._seq = 0
        self._lock = threading.Lock()
        self._result: dict | None = None
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vllm_hi_kv_collector", daemon=True,
        )
        self._thread.start()

    def dispatch(
        self,
        block_ids: list[int] | None,
        boost_cpu: bool,
        snapshot_event: torch.cuda.Event,
    ) -> str:
        """Enqueue a collection task and return its task_id immediately (microseconds, no inference blocking).

        Blocks when the queue is full (backpressure control) to keep dispatch from
        far outpacing copy speed and causing uncontrolled peak usage.
        """
        with self._lock:
            self._seq += 1
            task_id = f"kv-{self._seq:x}"
            self._result = None
            self._done.clear()
        self._task_queue.put((task_id, block_ids, boost_cpu, snapshot_event))
        return task_id

    def try_get_result(self, task_id: str | None = None) -> dict | None:
        """Non-blocking query of the most recently completed task result; return ``None`` if not done.

        Returns ``{shm_name, shm_size, metas, rank}``, ``{empty, rank, metas}`` or
        ``{error}``. The result can be re-read until the next ``dispatch``.
        """
        if self._done.is_set():
            return self._result
        return None

    def shutdown(self) -> None:
        """Stop the background thread (sentinel task; the daemon thread also exits with the process)."""
        try:
            self._task_queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        # The thread does not inherit the main thread's CUDA context; explicitly
        # set_device to reuse the main thread's context and avoid implicitly
        # creating a new context on device 0 that consumes extra VRAM.
        torch.cuda.set_device(self._device)
        while True:
            task = self._task_queue.get()
            if task is None:
                break
            task_id, block_ids, boost_cpu, snapshot_event = task
            try:
                result = self._collect(
                    task_id, block_ids, boost_cpu, snapshot_event
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[vllm_hi] KV collection task %s failed", task_id
                )
                result = {"error": f"{type(exc).__name__}: {exc}"}
            with self._lock:
                self._result = result
                self._done.set()

    def _block_bytes(self) -> int:
        """Total bytes of a single physical block across all layers (for D2H rate-limit estimation)."""
        total = 0
        for t in self._kv_caches:
            if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] > 0:
                total += (t.numel() // t.shape[0]) * t.element_size()
        return total

    def _layer_block_bytes(self) -> list[int]:
        """Per-layer bytes of a single physical block (same basis as ``_block_bytes``, returned per layer)."""
        result: list[int] = []
        for t in self._kv_caches:
            if isinstance(t, torch.Tensor) and t.dim() > 0 and t.shape[0] > 0:
                result.append((t.numel() // t.shape[0]) * t.element_size())
            else:
                result.append(0)
        return result

    def _num_blocks(self) -> int:
        """Total number of physical blocks in the KV cache (shape[0] is identical across layers; take the first tensor)."""
        for t in self._kv_caches:
            if isinstance(t, torch.Tensor) and t.dim() > 0:
                return int(t.shape[0])
        return 0

    def _compute_metas(
        self,
        kv_caches: list,
        ranges: list[tuple[int, int]],
        layer_block_bytes: list[int],
        start_layer: int,
    ) -> tuple[list[dict], int]:
        """Compute metas and total bytes from the range list and per-layer block bytes (decoupled from D2H).

        The layout is layer-major: one meta per layer, ``byte_offset`` is the
        layer's starting offset in shm, and ``shape[0]`` is the total blocks
        collected for that layer. Non-tensor layers get an empty meta.
        """
        total_blocks = sum(e - s for s, e in ranges)
        metas: list[dict] = []
        total = 0
        for li, t in enumerate(kv_caches):
            if not isinstance(t, torch.Tensor):
                metas.append({
                    "layer_idx": start_layer + li,
                    "shape": [],
                    "dtype": "",
                    "byte_offset": total,
                    "nbytes": 0,
                    "head_dim": 1,
                })
                continue
            lb = layer_block_bytes[li]
            layer_nbytes = total_blocks * lb
            first_shape = list(t.shape)
            first_shape[0] = total_blocks
            metas.append({
                "layer_idx": start_layer + li,
                "shape": first_shape,
                "dtype": str(t.dtype),
                "byte_offset": total,
                "nbytes": layer_nbytes,
                "head_dim": 1,
            })
            total += layer_nbytes
        return metas, total

    def _stream_d2h_encode(
        self,
        kv_caches: list,
        batches: list[list[tuple[int, int]]],
        layer_block_bytes: list[int],
        metas: list[dict],
        shm,
        boost_cpu: bool,
        skip_d2h: bool,
        skip_encode: bool,
        copy_stream,
        throttler,
    ) -> None:
        """Streaming pinned D2H + encode: D2H each batch into the pinned buffer, then encode into shm.

        1. The D2H target is a preallocated pinned buffer, so ``copy_`` is truly
           asynchronous (no hidden pageable staging allocation and synchronous
           copy-back);
        2. After each batch's D2H, use the ``Event.elapsed_time`` feedback loop to
           dynamically yield (smooth mode), keeping instantaneous bandwidth near
           budget rather than "peak burst + static sleep";
        3. Encode while D2H'ing, eliminating the intermediate memory peak of first
           fully D2H'ing into pageable memory and then aggregating.
        """
        pinned = self._pinned_pool.acquire()
        if pinned is None:
            raise RuntimeError("pinned buffer unavailable")
        shm_buf = torch.frombuffer(shm.buf, dtype=torch.uint8)
        layer_offset = [m["byte_offset"] for m in metas]
        n_batches = len(batches)
        use_smooth = isinstance(throttler, SmoothD2HThrottler)

        with torch.cuda.stream(copy_stream):
            for bi, batch in enumerate(batches):
                if not batch:
                    continue
                jobs: list[tuple[int, int, int, int]] = []
                batch_bytes = 0
                for li, t in enumerate(kv_caches):
                    if not isinstance(t, torch.Tensor):
                        continue
                    lb = layer_block_bytes[li]
                    for s, e in batch:
                        nbytes = (e - s) * lb
                        jobs.append((li, s, e, nbytes))
                        batch_bytes += nbytes
                if batch_bytes == 0:
                    continue

                elapsed_ms = 0.0
                if not skip_d2h:
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record(copy_stream)
                    off = 0
                    for (li, s, e, nbytes) in jobs:
                        t = kv_caches[li]
                        src = t[s:e].detach().flatten().view(torch.uint8)
                        pinned[off:off + nbytes].copy_(src, non_blocking=True)
                        off += nbytes
                    end_ev.record(copy_stream)
                    copy_stream.synchronize()
                    elapsed_ms = start_ev.elapsed_time(end_ev)

                if not skip_encode:
                    off = 0
                    for (li, s, e, nbytes) in jobs:
                        dst_off = layer_offset[li]
                        shm_buf[dst_off:dst_off + nbytes].copy_(
                            pinned[off:off + nbytes]
                        )
                        layer_offset[li] += nbytes
                        off += nbytes

                if not skip_d2h and not boost_cpu and bi < n_batches - 1:
                    if use_smooth:
                        gap = throttler.next_gap(elapsed_ms, batch_bytes)
                    else:
                        gap = throttler.yield_interval(batch_bytes)
                    if gap > 0:
                        time.sleep(gap)

        copy_stream.synchronize()

    def _collect(
        self,
        task_id: str,
        block_ids: list[int] | None,
        boost_cpu: bool,
        snapshot_event: torch.cuda.Event | None,
    ) -> dict:
        """Perform slicing + D2H copy + encode-to-shm on the dedicated copy_stream.

        With ``boost_cpu=False`` (during A's serving) it uses "pinned streaming
        D2H + sustained low-rate throttling"; full collection is also subject to
        chunked throttling, keeping D2H instantaneous bandwidth within budget and
        not contending with inference matmul for HBM read bandwidth. With
        ``boost_cpu=True`` (final increment, A paused) it uses the peak pageable
        path, prioritizing shorter downtime.
        """
        import numpy as np
        from multiprocessing import shared_memory

        # Global worker rank (consistent with the enumerate index in collector._assemble_kv_cache and the IPC path).
        rank = worker_global_rank()
        kv_caches = self._kv_caches
        if not kv_caches:
            return {"empty": True, "rank": rank, "task_id": task_id, "metas": []}

        # Performance probe switches (root-cause validation apparatus, off by default / default-fixed behavior).
        skip_d2h = PROBE_CONFIG.skip_d2h
        skip_encode = PROBE_CONFIG.skip_encode
        use_pinned = PROBE_CONFIG.pinned
        full_throttle = PROBE_CONFIG.full_throttle
        d2h_mode = PROBE_CONFIG.d2h_mode

        start_layer = layer_start_index(
            getattr(self._model_runner, "model_config", None),
            getattr(self._model_runner, "parallel_config", None),
        )

        old_threads = None
        if boost_cpu:
            old_threads = torch.get_num_threads()
            torch.set_num_threads(collect_cpu_threads())

        copy_stream = self._copy_stream
        if snapshot_event is not None:
            copy_stream.wait_event(snapshot_event)

        # Range partition: None = full (whole [0, num_blocks]), [] = empty, [(s,e),...] = incremental.
        if block_ids is not None:
            ranges = slice_blocks(block_ids)
        else:
            n_blocks = self._num_blocks()
            ranges = [(0, n_blocks)] if n_blocks > 0 else []

        layer_block_bytes = self._layer_block_bytes()
        block_bytes = self._block_bytes()

        # Choose throttler: smooth (default, event feedback loop) / burst (legacy static sleep, control).
        throttler = (
            self._throttler if d2h_mode == "burst" else self._smooth_throttler
        )

        # Batching: boost_cpu (final increment) is peak with no chunking; otherwise
        # chunked throttling (full collection included, unless
        # VLLM_HI_PERF_PROBE_FULL_THROTTLE=off falls back to legacy behavior).
        if boost_cpu:
            batches = [ranges]
        elif not full_throttle and block_ids is None:
            batches = [ranges]
        else:
            batches = list(throttler.batches(ranges))
        n_batches = len(batches)

        t0 = time.time()

        # Non-boost and pinned available: streaming pinned D2H + encode (truly async + throttled).
        if not boost_cpu and use_pinned and self._pinned_pool.available:
            metas, total = self._compute_metas(
                kv_caches, ranges, layer_block_bytes, start_layer
            )
            if total == 0:
                if old_threads is not None:
                    torch.set_num_threads(old_threads)
                return {"empty": True, "rank": rank, "task_id": task_id,
                        "metas": metas}
            shm = shared_memory.SharedMemory(create=True, size=total)
            try:
                self._stream_d2h_encode(
                    kv_caches, batches, layer_block_bytes, metas, shm,
                    boost_cpu, skip_d2h, skip_encode, copy_stream, throttler,
                )
            finally:
                shm.close()
            t2 = time.time()
            logger.info(
                "[vllm_hi Collector] worker=%d KV collection task=%s "
                "(pinned streaming): D2H+encode %.2fs, blocks=%s, bytes=%d, "
                "batches=%d, hbm_budget=%.2f, batch_blocks=%d, mode=%s",
                rank, task_id, t2 - t0,
                (len(block_ids) if block_ids is not None else "full"), total,
                n_batches, throttler.hbm_budget_ratio, throttler.batch_blocks,
                d2h_mode,
            )
            if old_threads is not None:
                torch.set_num_threads(old_threads)
            return {"rank": rank, "shm_name": shm.name,
                    "shm_size": total, "metas": metas, "task_id": task_id}

        # Fallback path (boost_cpu / pinned disabled or unavailable): legacy logic, pageable D2H.
        layer_tensors: list[list] = [[] for _ in kv_caches]
        for li, t in enumerate(kv_caches):
            if not isinstance(t, torch.Tensor):
                layer_tensors[li].append(t)
        with torch.cuda.stream(copy_stream):
            for bi, batch in enumerate(batches):
                for li, t in enumerate(kv_caches):
                    if not isinstance(t, torch.Tensor):
                        continue
                    if not batch:
                        layer_tensors[li].append(
                            t[0:0].detach().to("cpu", non_blocking=True)
                        )
                    else:
                        for s, e in batch:
                            layer_tensors[li].append(
                                t[s:e].detach().to("cpu", non_blocking=True)
                            )
                if not boost_cpu and bi < n_batches - 1 and block_bytes > 0:
                    batch_blocks_n = (
                        sum(e - s for s, e in batch) if batch else 0
                    )
                    copy_stream.synchronize()
                    time.sleep(
                        throttler.yield_interval(batch_blocks_n * block_bytes)
                    )
        copy_stream.synchronize()
        t1 = time.time()

        metas: list[dict] = []
        total = 0
        layer_contiguous: list[list[torch.Tensor]] = []
        for li, tensors in enumerate(layer_tensors):
            tcs: list[torch.Tensor] = []
            layer_nbytes = 0
            layer_blocks = 0
            first_shape: list[int] | None = None
            first_dtype = ""
            for x in tensors:
                if not isinstance(x, torch.Tensor):
                    continue
                tc = x.contiguous()
                layer_nbytes += tc.numel() * tc.element_size()
                if tc.dim() > 0:
                    layer_blocks += tc.shape[0]
                if first_shape is None:
                    first_shape = list(tc.shape)
                    first_dtype = str(tc.dtype)
                tcs.append(tc)
            if first_shape is None:
                metas.append({
                    "layer_idx": start_layer + li,
                    "shape": [],
                    "dtype": "",
                    "byte_offset": total,
                    "nbytes": 0,
                    "head_dim": 1,
                })
            else:
                first_shape[0] = layer_blocks
                metas.append({
                    "layer_idx": start_layer + li,
                    "shape": first_shape,
                    "dtype": first_dtype,
                    "byte_offset": total,
                    "nbytes": layer_nbytes,
                    "head_dim": 1,
                })
            layer_contiguous.append(tcs)
            total += layer_nbytes

        if total == 0:
            if old_threads is not None:
                torch.set_num_threads(old_threads)
            return {"empty": True, "rank": rank, "task_id": task_id,
                    "metas": metas}

        shm = shared_memory.SharedMemory(create=True, size=total)
        buf = None
        dst = None
        try:
            if boost_cpu:
                flat = [
                    tc.view(torch.uint8).reshape(-1)
                    for tcs in layer_contiguous for tc in tcs
                ]
                dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
                torch.cat(flat, out=dst)
            else:
                buf = np.frombuffer(shm.buf, dtype=np.uint8)
                offset = 0
                for tcs in layer_contiguous:
                    for tc in tcs:
                        nbytes = tc.numel() * tc.element_size()
                        buf[offset:offset + nbytes] = (
                            tc.view(torch.uint8).reshape(-1).numpy()
                        )
                        offset += nbytes
        finally:
            if buf is not None:
                del buf
            if dst is not None:
                del dst
            shm.close()
        t2 = time.time()
        logger.info(
            "[vllm_hi Collector] worker=%d KV collection task=%s: GPU→CPU %.2fs, "
            "buffer encode %.2fs, blocks=%s, bytes=%d, batches=%d, "
            "hbm_budget=%.2f, batch_blocks=%d",
            rank, task_id, t1 - t0, t2 - t1,
            (len(block_ids) if block_ids is not None else "full"), total,
            n_batches, throttler.hbm_budget_ratio, throttler.batch_blocks,
        )
        if old_threads is not None:
            torch.set_num_threads(old_threads)
        return {"rank": rank, "shm_name": shm.name,
                "shm_size": total, "metas": metas, "task_id": task_id}