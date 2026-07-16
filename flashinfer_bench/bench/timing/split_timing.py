"""Kernel-agnostic split timing engine.

Produces three first-class metrics for any ``Runnable`` that follows the
setup-hook convention (solution module exports a top-level ``setup`` symbol):

* ``e2e_ms``        : full wrapper cost — serialized host wall-clock of
                      setup() + run() per invocation, synchronized every
                      iteration. Models a naive serving call. Input clones
                      happen OUTSIDE the timed window (harness overhead,
                      not wrapper cost).
* ``kernel_ms``     : eager ``run()`` latency measured with CUDA Events —
                      setup runs ONCE outside the timed region.
* ``kernel_gpu_ms`` : hardware ground truth via CUPTI activity sum (eager
                      dispatch, ``use_cuda_graph=False``). ``None`` if
                      unavailable.

The default single-metric timing path is unchanged; this module is only used
when split timing is explicitly enabled.
"""

from __future__ import annotations

import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from flashinfer.testing import bench_gpu_time_with_cuda_event, bench_gpu_time_with_cupti

from flashinfer_bench.compile import Runnable

from ._common import _device_lock


@dataclass(frozen=True)
class SplitTimingMetrics:
    """Output of :func:`time_runnable_split_timing`.

    All times are medians in milliseconds. ``kernel_gpu_ms`` is ``None`` if
    the CUPTI measurement could not be taken (reason in the ``*_status``
    field, e.g. ``"no_cupti:ModuleNotFoundError"``).
    """

    e2e_ms: float
    kernel_ms: float
    kernel_gpu_ms: Optional[float]
    kernel_ms_status: str
    """``"ok"`` when eager CUDA Event measurement completed."""
    kernel_gpu_ms_status: str
    """``"ok"`` | ``"no_cupti:<Exception>"`` (cupti-python not importable) |
    ``"cupti_no_samples"`` (CUPTI returned an empty list) |
    ``"cupti_fallback:cuda_events"`` (cupti-python installed but the library
    was unusable — e.g. cu12 container with cupti-python 13.x — and flashinfer
    silently fell back to CUDA events; numbers are still valid CUDA-event
    timings but no longer a CUPTI activity sum). ``kernel_gpu_ms`` is ``None``
    for the two unavailable statuses so callers can never mistake a sentinel
    for a measurement."""


def time_runnable_split_timing(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    *,
    cold_l2_cache: bool = True,
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
        Timed iterations per metric (host wall-clock e2e samples, eager
        run-only CUDA Event samples, and CUPTI activity rounds).
    device : str
        CUDA device id (e.g. ``"cuda:0"``).
    cold_l2_cache : bool
        Apply the same L2 policy to all three metrics. ``True`` flushes L2
        before each measured invocation; ``False`` preserves warm-cache state.
        For ``e2e_ms`` this describes the start of the full setup + run
        invocation, not the cache state at the internal ``run()`` boundary.

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
            kernel_ms, kernel_status = _measure_kernel_cudaevent(
                runnable, args, warmup, iters, device, cold_l2_cache
            )
            _cool_down(device)
            kernel_gpu_ms, kernel_gpu_status = _measure_kernel_gpu_cupti(
                runnable, args, warmup, iters, device, cold_l2_cache
            )
            _cool_down(device)
            e2e_ms = _measure_e2e(runnable, args, warmup, iters, device, cold_l2_cache)
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


def _make_l2_flusher(device: str) -> Callable[[], None]:
    """Build a flush-L2 callable holding a 2x-L2-sized scratch buffer.

    Same policy as flashinfer's ``bench_gpu_time_*`` helpers (not reusable
    here because the e2e loop is timed on the host side). The buffer lives
    only as long as the returned closure, so its memory goes back to the
    allocator after each measurement instead of staying pinned per device.
    """
    l2_size = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    size = (l2_size or 64 * 1024 * 1024) * 2
    buf = torch.empty(size, dtype=torch.int8, device=device)

    def _flush() -> None:
        buf.zero_()

    return _flush


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


def _measure_e2e(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    cold_l2_cache: bool = True,
) -> float:
    """e2e_ms — serialized host wall-clock of setup + run per invocation.

    Each iteration:

    1. Deep-clone every tensor in ``args`` (OUTSIDE the timed window — the
       clone protects against in-place mutation across iterations; it is
       harness overhead, not wrapper cost)
    2. Optional L2 flush, then synchronize
    3. ``t0`` -> ``runnable.setup_for_workload(*cloned)`` ->
       ``runnable(*cloned)`` -> synchronize -> ``t1``

    Host wall-clock with per-iteration synchronization is deliberate:
    CUDA-event stream timing hides CPU-side wrapper work whenever the CPU can
    run ahead of a busy GPU (iterations pipeline and only synchronize after
    the loop). Serializing captures the per-call latency a naive serving
    caller would actually observe, at the cost of one device sync per
    iteration (µs-scale, negligible against wrapper work).
    """
    flush_l2 = _make_l2_flusher(device) if cold_l2_cache else None

    # Warmup mirrors flashinfer's dry-run behavior: flush before every
    # invocation under the cold-L2 policy.
    for _ in range(warmup):
        cloned = tuple(_maybe_clone(a) for a in args)
        if flush_l2 is not None:
            flush_l2()
        runnable.setup_for_workload(*cloned)
        runnable(*cloned)
    torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        cloned = tuple(_maybe_clone(a) for a in args)
        if flush_l2 is not None:
            flush_l2()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        runnable.setup_for_workload(*cloned)
        runnable(*cloned)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _measure_kernel_cudaevent(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    cold_l2_cache: bool = True,
) -> Tuple[float, str]:
    """kernel_ms — setup once outside, then eager ``run()`` via CUDA Events."""
    runnable.setup_for_workload(*args)
    times = bench_gpu_time_with_cuda_event(
        fn=runnable,
        dry_run_iters=warmup,
        repeat_iters=iters,
        input_args=tuple(args),
        cold_l2_cache=cold_l2_cache,
    )
    return statistics.median(times), "ok"


def _measure_kernel_gpu_cupti(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    cold_l2_cache: bool = True,
) -> Tuple[Optional[float], str]:
    """kernel_gpu_ms — setup ONCE outside; ``bench_gpu_time_with_cupti`` on
    eager dispatch. Uses CUPTI activity sum (pure GPU exec, excludes Python
    inter-kernel gaps). Returns ``None`` (never a 0.0 sentinel) when the
    measurement is unavailable.

    Distinct mechanism from ``kernel_ms``: CUPTI activity sum versus eager
    CUDA Event elapsed time. A wider gap is diagnostic of multi-kernel launch
    gaps or CUPTI span-versus-sum differences.
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
                cold_l2_cache=cold_l2_cache,
                use_cuda_graph=False,
            )
        except Exception as ex:
            return None, f"no_cupti:{type(ex).__name__}"
    if not times:
        return None, "cupti_no_samples"
    for w in caught:
        msg = str(w.message).lower()
        if "cupti" in msg and (
            "falling back" in msg or "not installed" in msg or "needs to be" in msg
        ):
            return statistics.median(times), "cupti_fallback:cuda_events"
    return statistics.median(times), "ok"
