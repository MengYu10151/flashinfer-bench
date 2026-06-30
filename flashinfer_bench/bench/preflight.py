"""Runtime dependency preflight checks for benchmark solutions."""

from __future__ import annotations

from flashinfer_bench.bench.flashinfer_preflight import (
    FlashInferPreflightError,
    ensure_flashinfer_runtime_ready,
)
from flashinfer_bench.data import Solution


class SolutionPreflightError(RuntimeError):
    """Raised when a solution cannot run in the current worker environment."""


def ensure_solution_runtime_ready(solution: Solution, device: str) -> None:
    """Run solution preflight checks before building the runnable.

    The runner owns only this generic dispatch point. Kernel/library-specific
    checks live in separate modules and should be narrow, opt-out friendly, and
    side-effect free for unrelated solutions.
    """

    try:
        ensure_flashinfer_runtime_ready(solution, device)
    except FlashInferPreflightError as exc:
        raise SolutionPreflightError(str(exc)) from exc
