"""Kernel-agnostic two-mode timing engine.

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

See ``rfcs/two_mode_kernel_agnostic.md`` for the full design and locked
decisions (Q1-Q6 in §8).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from flashinfer_bench.compile import Runnable

from ._common import _device_lock


@dataclass(frozen=True)
class ThreeMetrics:
    """Output of :func:`time_runnable_two_mode`.

    All times are medians in milliseconds. ``kernel_ms`` and ``kernel_gpu_ms``
    are ``0.0`` if the corresponding measurement could not be taken (reason in
    the ``*_status`` field, e.g. ``"fallback_eager:RuntimeError"`` or
    ``"no_cupti:ModuleNotFoundError"``).
    """

    e2e_ms: float
    kernel_ms: float
    kernel_gpu_ms: float
    kernel_ms_status: str
    kernel_gpu_ms_status: str


def time_runnable_two_mode(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    *,
    graph_iters: int = 20,
) -> ThreeMetrics:
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
    ThreeMetrics
        e2e_ms, kernel_ms, kernel_gpu_ms (medians) and status strings.
    """
    lock = _device_lock(device)
    with lock:
        with torch.cuda.device(device):
            e2e_ms = _measure_e2e(runnable, args, warmup, iters, device)
            kernel_ms, kernel_status = _measure_kernel_cudagraph(
                runnable, args, warmup, graph_iters, iters, device
            )
            kernel_gpu_ms, kernel_gpu_status = _measure_kernel_gpu_cupti(
                runnable, args, warmup, iters, device
            )
    return ThreeMetrics(
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

    # Warmup against the original args so internal state (lazy-init in
    # FlashInfer wrappers, FA3 first-call paths, ...) is settled before capture.
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

    # Prime replay (first replay can carry one-off init cost)
    graph.replay()
    torch.cuda.synchronize(device)

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

    Distinct mechanism from ``kernel_ms`` — eager + CUPTI vs cudagraph +
    cudaEvent — by design (see RFC §8 decision 3). The two metrics should
    agree within a few µs for graph-capturable kernels; a wider gap is
    diagnostic of capture failure / multi-kernel host-side sync / CUPTI
    span-vs-sum issues.
    """
    runnable.setup_for_workload(*args)
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
    return statistics.median(times), "ok"
