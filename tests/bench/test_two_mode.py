"""Unit tests for the kernel-agnostic two-mode timing engine.

Covers ``ThreeMetrics`` schema, the internal ``_measure_*`` helpers, and the
top-level ``time_runnable_two_mode`` API. CPU-only tests use mocks; GPU-required
tests are guarded with ``pytest.mark.skipif(torch.cuda.device_count() == 0)``.
"""

from __future__ import annotations

import statistics
import warnings
from typing import Any, List
from unittest.mock import patch

import pytest
import torch

from flashinfer_bench.bench.timing import ThreeMetrics, time_runnable_two_mode
from flashinfer_bench.bench.timing.two_mode import (
    _maybe_clone,
    _measure_e2e,
    _measure_kernel_cudagraph,
    _measure_kernel_gpu_cupti,
    _median_cudaevent,
)
from flashinfer_bench.compile import Runnable, RunnableMetadata

# -----------------------------------------------------------------------------
# Helpers — build a small mock Runnable for tests
# -----------------------------------------------------------------------------


def _make_runnable(run_fn, setup_fn=None) -> Runnable:
    return Runnable(
        callable=run_fn,
        metadata=RunnableMetadata(
            build_type="python",
            definition_name="test_def",
            solution_name="test_sol",
            destination_passing_style=False,
        ),
        setup_callable=setup_fn,
    )


# -----------------------------------------------------------------------------
# ThreeMetrics dataclass — schema, frozenness, defaults
# -----------------------------------------------------------------------------


class TestThreeMetrics:
    def test_construct_with_all_fields(self):
        m = ThreeMetrics(
            e2e_ms=1.5,
            kernel_ms=0.3,
            kernel_gpu_ms=0.32,
            kernel_ms_status="ok",
            kernel_gpu_ms_status="ok",
        )
        assert m.e2e_ms == 1.5
        assert m.kernel_ms == 0.3
        assert m.kernel_gpu_ms == 0.32
        assert m.kernel_ms_status == "ok"
        assert m.kernel_gpu_ms_status == "ok"

    def test_frozen(self):
        m = ThreeMetrics(1.0, 0.1, 0.1, "ok", "ok")
        with pytest.raises((AttributeError, TypeError)):
            m.e2e_ms = 2.0


# -----------------------------------------------------------------------------
# _maybe_clone — tensor clones, non-tensor passes through
# -----------------------------------------------------------------------------


class TestMaybeClone:
    def test_tensor_is_cloned(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        c = _maybe_clone(t)
        assert c is not t
        assert torch.equal(c, t)

    def test_tensor_clone_independent(self):
        t = torch.tensor([1.0, 2.0])
        c = _maybe_clone(t)
        c[0] = 99.0
        assert t[0].item() == 1.0  # original unaffected

    @pytest.mark.parametrize("value", [42, 3.14, "string", None, (1, 2), [1, 2, 3]])
    def test_non_tensor_passes_through(self, value):
        assert _maybe_clone(value) is value


# -----------------------------------------------------------------------------
# _measure_kernel_gpu_cupti — CPU-mockable: exception + fallback + ok paths
# -----------------------------------------------------------------------------


class TestMeasureKernelGpuCupti:
    """All branches of _measure_kernel_gpu_cupti — no GPU required, the
    `bench_gpu_time_with_cupti` call is mocked out."""

    def _build_runnable(self):
        # The Runnable's setup_for_workload is called before the timer; mock its
        # callable so we don't accidentally exercise real code.
        called = {"setup": 0, "run": 0}

        def _setup(*a):
            called["setup"] += 1
            return {}

        def _run(*a):
            called["run"] += 1

        return _make_runnable(_run, _setup), called

    def test_returns_no_cupti_when_import_fails(self):
        runnable, called = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            side_effect=ModuleNotFoundError("No module named 'cupti'"),
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms == 0.0
        assert status.startswith("no_cupti:")
        assert "ModuleNotFoundError" in status
        # setup is still called before the timer — verifies the contract
        assert called["setup"] == 1

    def test_returns_no_cupti_when_runtime_error(self):
        runnable, _ = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            side_effect=RuntimeError("Incompatible CUPTI Library"),
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms == 0.0
        assert status == "no_cupti:RuntimeError"

    def test_returns_cupti_no_samples_on_empty(self):
        runnable, _ = self._build_runnable()
        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti", return_value=[]
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert ms == 0.0
        assert status == "cupti_no_samples"

    def test_returns_ok_with_median_when_clean(self):
        runnable, _ = self._build_runnable()
        # Three samples; median is the middle one.
        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            return_value=[0.1, 0.2, 0.3],
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=3, device="cuda:0"
            )
        assert ms == pytest.approx(0.2)
        assert status == "ok"

    def test_detects_silent_cupti_fallback_via_warning(self):
        """Flashinfer's `bench_gpu_time_with_cupti` emits a UserWarning when it
        can't reach a real CUPTI runtime and silently falls back to CUDA events.
        Without the warning-capture logic, we'd return status="ok" — hiding
        that the numbers are CUDA-event timings, not CUPTI activity sums. The
        new code surfaces this as `cupti_fallback:cuda_events`."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn(
                "CUPTI is not installed. Try 'pip install -U cupti-python'. "
                "Falling back to CUDA events for benchmarking.",
                UserWarning,
                stacklevel=1,
            )
            return [0.1, 0.2, 0.3]

        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            ms, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=3, device="cuda:0"
            )
        assert ms == pytest.approx(0.2)
        assert status == "cupti_fallback:cuda_events"

    def test_detects_version_mismatch_fallback(self):
        """Same as above but with the "needs to be >= 13.0.0" wording flashinfer
        uses for ABI-version mismatches."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn(
                "CUPTI needs to be >= 13.0.0. Falling back to CUDA events.",
                UserWarning,
                stacklevel=1,
            )
            return [1.0]

        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            _, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert status == "cupti_fallback:cuda_events"

    def test_unrelated_warning_does_not_trigger_fallback_status(self):
        """A non-CUPTI warning shouldn't be misclassified as a CUPTI fallback."""
        runnable, _ = self._build_runnable()

        def _fake_bench(*args, **kwargs):
            warnings.warn("Some unrelated deprecation notice.", DeprecationWarning, stacklevel=1)
            return [0.5]

        with patch(
            "flashinfer_bench.bench.timing.two_mode.bench_gpu_time_with_cupti",
            side_effect=_fake_bench,
        ):
            _, status = _measure_kernel_gpu_cupti(
                runnable, [42], warmup=1, iters=1, device="cuda:0"
            )
        assert status == "ok"


# -----------------------------------------------------------------------------
# GPU-required tests below
# -----------------------------------------------------------------------------

cuda_available = pytest.mark.skipif(
    torch.cuda.device_count() == 0, reason="CUDA devices not available"
)


@cuda_available
class TestMedianCudaevent:
    def test_returns_positive_finite_ms(self):
        # A trivial GPU op — just need to verify a cudaEvent pair returns >0 ms.
        x = torch.randn(1024, device="cuda")
        ms = _median_cudaevent(lambda: x.sum(), iters=5, device="cuda:0")
        assert ms > 0.0
        assert ms < 100.0  # sanity bound — a single small reduction is fast


@cuda_available
class TestMeasureE2E:
    def test_setup_called_once_per_iter(self):
        """Per RFC §8.5: e2e re-runs setup_for_workload inside the timing
        region. Verify it's called warmup+iters times total."""
        called = {"setup": 0, "run": 0}

        def _setup(t):
            called["setup"] += 1
            return {}

        def _run(t):
            called["run"] += 1
            _ = t.sum()

        runnable = _make_runnable(_run, _setup)
        x = torch.randn(64, device="cuda")
        WARMUP, ITERS = 3, 5
        ms = _measure_e2e(runnable, [x], warmup=WARMUP, iters=ITERS, device="cuda:0")
        # Each iter calls setup once + run once → warmup+iters of each
        assert called["setup"] == WARMUP + ITERS
        assert called["run"] == WARMUP + ITERS
        assert ms > 0.0

    def test_tensors_are_cloned_per_iter(self):
        """e2e clones tensor args each iter. Mutating cloned input inside run()
        should not leak back to the caller's original tensor."""
        original = torch.zeros(8, device="cuda")

        def _setup(t):
            return {}

        def _run(t):
            t.add_(1.0)

        runnable = _make_runnable(_run, _setup)
        _measure_e2e(runnable, [original], warmup=1, iters=2, device="cuda:0")
        # If clones leaked, original would have been incremented 3 times.
        assert torch.equal(original, torch.zeros(8, device="cuda"))


@cuda_available
class TestMeasureKernelCudagraph:
    def test_happy_path_returns_ok(self):
        """A simple capturable kernel should produce a cudagraph + cudaEvent
        median > 0 with status="ok"."""
        # Pre-allocate outputs so capture has stable buffers.
        a = torch.randn(128, device="cuda")
        b = torch.randn(128, device="cuda")
        out = torch.empty(128, device="cuda")

        def _setup(a, b):
            return {}

        def _run(a, b):
            torch.add(a, b, out=out)

        runnable = _make_runnable(_run, _setup)
        ms, status = _measure_kernel_cudagraph(
            runnable, [a, b], warmup=3, graph_iters=5, replays=5, device="cuda:0"
        )
        assert status == "ok"
        assert ms > 0.0

    def test_fallback_when_capture_raises(self):
        """If torch.cuda.graph raises (e.g. due to host-sync in run()), the
        engine should fall back to eager dispatch with status starting
        ``fallback_eager:``."""
        a = torch.randn(64, device="cuda")

        def _setup(t):
            return {}

        def _run(t):
            # Sum is fine on its own; we'll force a graph-capture failure
            # by patching torch.cuda.CUDAGraph to raise.
            _ = t.sum()

        runnable = _make_runnable(_run, _setup)
        original_graph = torch.cuda.CUDAGraph

        def _raising_graph(*args, **kwargs):
            raise RuntimeError("simulated capture failure")

        with patch("torch.cuda.CUDAGraph", _raising_graph):
            ms, status = _measure_kernel_cudagraph(
                runnable, [a], warmup=2, graph_iters=3, replays=3, device="cuda:0"
            )
        assert status.startswith("fallback_eager:")
        assert "RuntimeError" in status
        assert ms > 0.0


@cuda_available
class TestTimeRunnableTwoMode:
    """End-to-end: invoke the public API on a mock Runnable, verify all three
    metrics + both status strings come back populated and finite."""

    def test_three_metrics_populated(self):
        a = torch.randn(256, device="cuda")
        b = torch.randn(256, device="cuda")
        out = torch.empty(256, device="cuda")

        def _setup(a, b):
            return {}

        def _run(a, b):
            torch.mul(a, b, out=out)

        runnable = _make_runnable(_run, _setup)
        m = time_runnable_two_mode(
            runnable, [a, b], warmup=3, iters=5, device="cuda:0", graph_iters=5
        )

        assert isinstance(m, ThreeMetrics)
        assert m.e2e_ms > 0.0
        # kernel_ms may be 0 only when graph-capture *and* eager fallback both
        # fail — extremely unlikely for a simple mul, so assert positive.
        assert m.kernel_ms > 0.0
        # kernel_gpu_ms may be 0 if cupti is unavailable; status string then
        # documents the reason. Either way, status must be one of the known
        # values.
        assert m.kernel_ms_status in ("ok",) or m.kernel_ms_status.startswith("fallback_eager:")
        assert m.kernel_gpu_ms_status in (
            "ok",
            "cupti_no_samples",
            "cupti_fallback:cuda_events",
        ) or m.kernel_gpu_ms_status.startswith("no_cupti:")

    def test_e2e_at_least_kernel(self):
        """e2e_ms >= kernel_ms (modulo run-to-run noise — give it some slack).
        Rationale: e2e includes everything kernel_ms does, plus clone + setup
        per iter."""
        a = torch.randn(512, device="cuda")
        out = torch.empty(512, device="cuda")

        def _setup(t):
            return {}

        def _run(t):
            # add(t,t) supports out=; relu(t, out=…) is missing on some torch builds.
            torch.add(t, t, out=out)

        runnable = _make_runnable(_run, _setup)
        m = time_runnable_two_mode(
            runnable, [a], warmup=5, iters=10, device="cuda:0", graph_iters=5
        )
        # 5x slack to absorb noise on extremely small kernels; the inequality
        # is robust on any non-trivial workload.
        assert m.e2e_ms >= m.kernel_ms * 0.5
