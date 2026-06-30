import importlib.util
import sys
import types

import pytest

from flashinfer_bench.bench.flashinfer_preflight import (
    FlashInferPreflightError,
    ensure_flashinfer_runtime_ready,
    requires_flashinfer_sm100_gemm_aot,
)
from flashinfer_bench.bench.preflight import SolutionPreflightError, ensure_solution_runtime_ready
from flashinfer_bench.data import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedLanguages,
)


def _solution(content: str) -> Solution:
    return Solution(
        name="s",
        definition="d",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
        ),
        sources=[SourceFile(path="main.py", content=content)],
    )


def test_requires_flashinfer_sm100_gemm_aot_for_default_grouped_fp8():
    sol = _solution(
        "import flashinfer\n"
        "def run(a,b,sa,sb,indptr):\n"
        "    return flashinfer.gemm.group_gemm_fp8_nt_groupwise(a,b,sa,sb,indptr)\n"
    )

    assert requires_flashinfer_sm100_gemm_aot(sol) is True


def test_does_not_require_flashinfer_sm100_gemm_aot_for_cutile_backend():
    sol = _solution(
        "import flashinfer\n"
        "def run(a,b,sa,sb,indptr):\n"
        "    return flashinfer.gemm.group_gemm_fp8_nt_groupwise(\n"
        "        a,b,sa,sb,indptr, backend=\"cutile\")\n"
    )

    assert requires_flashinfer_sm100_gemm_aot(sol) is False


def test_preflight_fails_fast_when_aot_module_missing(monkeypatch, tmp_path):
    sol = _solution(
        "import flashinfer\n"
        "def run(a,b,sa,sb,indptr):\n"
        "    return flashinfer.gemm.group_gemm_fp8_nt_groupwise(a,b,sa,sb,indptr)\n"
    )

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "flashinfer_jit_cache":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    flashinfer_mod = types.ModuleType("flashinfer")
    flashinfer_mod.__version__ = "0.6.13"
    jit_mod = types.ModuleType("flashinfer.jit")
    env_mod = types.ModuleType("flashinfer.jit.env")
    env_mod.FLASHINFER_AOT_DIR = tmp_path
    jit_mod.env = env_mod
    tvm_ffi_mod = types.ModuleType("tvm_ffi")
    tvm_ffi_mod.load_module = lambda path: object()
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_mod)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", jit_mod)
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", env_mod)
    monkeypatch.setitem(sys.modules, "tvm_ffi", tvm_ffi_mod)

    with pytest.raises(FlashInferPreflightError, match="flashinfer-jit-cache"):
        ensure_flashinfer_runtime_ready(sol, "cuda:0")


def test_preflight_can_be_skipped(monkeypatch):
    sol = _solution(
        "import flashinfer\n"
        "def run(a,b,sa,sb,indptr):\n"
        "    return flashinfer.gemm.group_gemm_fp8_nt_groupwise(a,b,sa,sb,indptr)\n"
    )

    monkeypatch.setenv("FIB_SKIP_FLASHINFER_PREFLIGHT", "1")

    ensure_flashinfer_runtime_ready(sol, "cuda:0")


def test_generic_preflight_wraps_flashinfer_errors(monkeypatch):
    sol = _solution("def run(a):\n    return a\n")

    def fail(solution, device):
        raise FlashInferPreflightError("missing runtime")

    monkeypatch.setattr("flashinfer_bench.bench.preflight.ensure_flashinfer_runtime_ready", fail)

    with pytest.raises(SolutionPreflightError, match="missing runtime"):
        ensure_solution_runtime_ready(sol, "cuda:0")
