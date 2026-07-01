"""Kernel-agnostic split timing engine.

Produces three first-class metrics for any ``Runnable`` that follows the
setup-hook convention (solution module exports a top-level ``setup`` symbol):

* ``e2e_ms``        : full wrapper cost — each iter clones all tensor args
                      AND re-runs setup() inside the timed region (cudaEvent,
                      eager dispatch). Models a naive serving call.
* ``kernel_ms``     : cross-library-comparable pure kernel time — setup runs
                      ONCE outside, the ``run()`` callable is captured into a
                      CUDA graph, and replay is timed via cudaEvent.
                      Falls back to eager dispatch if capture fails.
* ``kernel_gpu_ms`` : hardware ground truth via CUPTI activity sum (eager
                      dispatch, ``use_cuda_graph=False``). 0.0 if unavailable.

The default single-metric timing path is unchanged; this module is only used
when split timing is explicitly enabled.
"""

from __future__ import annotations

import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from flashinfer_bench.compile import Runnable

from ._common import _device_lock


@dataclass(frozen=True)
class SplitTimingMetrics:
    """Output of :func:`time_runnable_split_timing`.

    All times are medians in milliseconds. ``kernel_ms`` and ``kernel_gpu_ms``
    are ``0.0`` if the corresponding measurement could not be taken (reason in
    the ``*_status`` field, e.g. ``"fallback_eager:RuntimeError"`` or
    ``"no_cupti:ModuleNotFoundError"``).
    """

    e2e_ms: float
    kernel_ms: float
    kernel_gpu_ms: float
    kernel_ms_status: str
    """``"ok"`` | ``"fallback_eager:<Exception>"`` (cudagraph capture failed)."""
    kernel_gpu_ms_status: str
    """``"ok"`` | ``"no_cupti:<Exception>"`` (cupti-python not importable) |
    ``"cupti_no_samples"`` (CUPTI returned an empty list) |
    ``"cupti_fallback:cuda_events"`` (cupti-python installed but the library
    was unusable — e.g. cu12 container with cupti-python 13.x — and flashinfer
    silently fell back to CUDA events; numbers are still valid but no longer a
    CUPTI activity sum)."""


def time_runnable_split_timing(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    *,
    graph_iters: int = 20,
) -> SplitTimingMetrics:
    """Measure e2e / kernel / kernel_gpu in one call.

    Parameters
    ----------
    runnable : Runnable
        The compiled solution. ``runnable.setup_for_workload(*args)`` is
        invoked at the right point for each metric.
    args : List[Any]
        Positional args in definition order (inputs + outputs for DPS).
    warmup : int
        Warmup iterations before each timed measurement.
    iters : int
        Timed iterations per metric (e2e cudaEvent samples, cudagraph
        replays, CUPTI activity rounds).
    device : str
        CUDA device id (e.g. ``"cuda:0"``).
    graph_iters : int
        Number of ``run()`` calls captured into a single CUDA graph.
        ``kernel_ms`` is reported as ``replay_time / graph_iters``.

    Returns
    -------
    SplitTimingMetrics
        e2e_ms, kernel_ms, kernel_gpu_ms (medians) and status strings.
    """
    # Measurement order: kernel_ms -> kernel_gpu_ms -> e2e.
    #
    # Run the kernel-only measurements first so their values are less likely to
    # be affected by any persistent wrapper state, synchronization, or GPU clock
    # effects introduced by the heavier end-to-end phase.
    lock = _device_lock(device)
    with lock:
        with torch.cuda.device(device):
            kernel_ms, kernel_status = _measure_kernel_cudagraph(
                runnable, args, warmup, graph_iters, iters, device
            )
            _cool_down(device)
            kernel_gpu_ms, kernel_gpu_status = _measure_kernel_gpu_cupti(
                runnable, args, warmup, iters, device
            )
            _cool_down(device)
            e2e_ms = _measure_e2e(runnable, args, warmup, iters, device)
    return SplitTimingMetrics(
        e2e_ms=e2e_ms,
        kernel_ms=kernel_ms,
        kernel_gpu_ms=kernel_gpu_ms,
        kernel_ms_status=kernel_status,
        kernel_gpu_ms_status=kernel_gpu_status,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _maybe_clone(a: Any) -> Any:
    """Deep-clone tensor args; pass-through scalars and non-tensor types."""
    if isinstance(a, torch.Tensor):
        return a.clone()
    return a


def _cool_down(device: str, seconds: float = 0.1) -> None:
    """Brief sync + idle between metric phases.

    Without this the three measurements run back-to-back. A short pause helps
    separate phase-local wrapper work from the following kernel-only timing
    pass, especially on workloads whose setup path synchronizes or launches
    auxiliary kernels.

    100 ms is short enough not to noticeably slow benchmark runs while giving
    the device a brief chance to settle between metric phases.
    """
    torch.cuda.synchronize(device)
    time.sleep(seconds)


def _median_cudaevent(fn: Callable[[], Any], iters: int, device: str) -> float:
    """Time ``fn()`` ``iters`` times with cudaEvent pairs; return median ms."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(iters))


def _measure_e2e(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
) -> float:
    """e2e_ms — clone all tensor args + re-run setup + run inside the timed region.

    Each iter:

    1. Deep-clone every tensor in ``args``
    2. ``runnable.setup_for_workload(*cloned)`` — rebuild plan handles, etc.
       against the clones
    3. ``runnable(*cloned)`` — the hot path

    All three steps are inside the cudaEvent timing region. This is the
    worst-case naive-serving cost.
    """

    def one() -> None:
        cloned = tuple(_maybe_clone(a) for a in args)
        runnable.setup_for_workload(*cloned)
        runnable(*cloned)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize(device)
    return _median_cudaevent(one, iters, device)


def _measure_kernel_cudagraph(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    graph_iters: int,
    replays: int,
    device: str,
) -> Tuple[float, str]:
    """kernel_ms — setup ONCE outside; capture ``run()`` into a CUDA graph;
    cudaEvent over graph replay (divided by ``graph_iters`` per replay).

    Falls back to eager dispatch with status ``fallback_eager:<Exception>`` if
    the kernel cannot be captured (stream-ordered allocators, host syncs in
    ``run``, unsupported ops, etc.).
    """
    runnable.setup_for_workload(*args)

    # Warmup against the original args so lazy initialization and wrapper
    # first-call paths are settled before graph capture.
    for _ in range(warmup):
        runnable(*args)
    torch.cuda.synchronize(device)

    try:
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_iters):
                    runnable(*args)
        torch.cuda.current_stream(device).wait_stream(stream)
    except (RuntimeError, NotImplementedError) as ex:
        # Fallback: eager dispatch, no graph
        median = _median_cudaevent(lambda: runnable(*args), replays, device)
        return median, f"fallback_eager:{type(ex).__name__}"

    # Prime replay (first replay can carry one-off init cost). Replays are
    # launched and timed on the capture stream so cudaEvents bracket the actual
    # graph work instead of only measuring launch overhead on the default stream.
    with torch.cuda.stream(stream):
        graph.replay()
    torch.cuda.synchronize(device)

    with torch.cuda.stream(stream):
        median_graph = _median_cudaevent(graph.replay, replays, device)
    return median_graph / graph_iters, "ok"


def _measure_kernel_gpu_cupti(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
) -> Tuple[float, str]:
    """kernel_gpu_ms — setup ONCE outside; ``bench_gpu_time_with_cupti`` on
    eager dispatch. Uses CUPTI activity sum (pure GPU exec, excludes Python
    inter-kernel gaps).

    Distinct mechanism from ``kernel_ms`` -- eager + CUPTI vs cudagraph +
    cudaEvent -- by design. The two metrics should
    agree within a few µs for graph-capturable kernels; a wider gap is
    diagnostic of capture failure / multi-kernel host-side sync / CUPTI
    span-vs-sum issues.
    """
    runnable.setup_for_workload(*args)
    # Capture warnings so we can detect flashinfer's internal CUPTI-fallback
    # path (it raises a UserWarning saying "CUPTI is not installed" /
    # "needs to be >= X.X.X" and silently switches to CUDA events). Without
    # this we'd report status="ok" while actually returning CUDA-event numbers,
    # which hides the fact that kernel_gpu_ms is no longer a CUPTI activity
    # sum. Surface that via status="cupti_fallback:cuda_events".
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            times = bench_gpu_time_with_cupti(
                fn=runnable,
                dry_run_iters=warmup,
                repeat_iters=iters,
                input_args=tuple(args),
                cold_l2_cache=True,
                use_cuda_graph=False,
            )
        except Exception as ex:
            return 0.0, f"no_cupti:{type(ex).__name__}"
    if not times:
        return 0.0, "cupti_no_samples"
    for w in caught:
        msg = str(w.message).lower()
        if "cupti" in msg and (
            "falling back" in msg or "not installed" in msg or "needs to be" in msg
        ):
            return statistics.median(times), "cupti_fallback:cuda_events"
    return statistics.median(times), "ok"
