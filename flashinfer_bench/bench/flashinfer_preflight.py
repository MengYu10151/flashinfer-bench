"""Preflight checks for FlashInfer solution runtime dependencies."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

from flashinfer_bench.data import Solution


class FlashInferPreflightError(RuntimeError):
    """Raised when a FlashInfer solution depends on unavailable runtime artifacts."""


_SM100_GEMM_AOT_TOKENS = (
    "group_gemm_fp8_nt_groupwise",
    "group_gemm_mxfp4_nt_groupwise",
)
_CUTILE_BACKEND_RE = re.compile(r"backend\s*=\s*['\"]cutile['\"]")


def _source_text(solution: Solution) -> str:
    return "\n".join(source.content for source in solution.sources)


def requires_flashinfer_sm100_gemm_aot(solution: Solution) -> bool:
    """Return whether a solution is expected to load FlashInfer's SM100 GEMM AOT module.

    The default FlashInfer grouped FP8/MXFP4 public APIs call ``get_gemm_sm100_module()``
    on B200/SM100. If the optional ``flashinfer-jit-cache`` package is missing, that
    path falls back to a very heavy local JIT compile and can exceed benchmark timeouts.
    ``backend="cutile"`` is intentionally excluded because it uses the cuTile path.
    """

    text = _source_text(solution)
    if "flashinfer" not in text:
        return False
    if _CUTILE_BACKEND_RE.search(text):
        return False
    return any(token in text for token in _SM100_GEMM_AOT_TOKENS)


def ensure_flashinfer_runtime_ready(solution: Solution, device: str) -> None:
    """Fail fast when a FlashInfer solution needs unavailable AOT runtime support.

    This check runs inside the benchmark worker, so it observes the same Python
    environment and dynamic library paths as the actual solution execution.
    """

    if os.environ.get("FIB_SKIP_FLASHINFER_PREFLIGHT") == "1":
        return
    if not requires_flashinfer_sm100_gemm_aot(solution):
        return
    if not _is_sm100_device(device):
        return

    _check_sm100_gemm_aot_module_loads()


def _is_sm100_device(device: str) -> bool:
    if not device.startswith("cuda"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(torch.device(device))
    except Exception:
        return False
    return major == 10


def _check_sm100_gemm_aot_module_loads() -> None:
    try:
        import flashinfer
        from flashinfer.jit import env as jit_env
        import tvm_ffi
    except Exception as exc:
        raise FlashInferPreflightError(
            "FlashInfer grouped GEMM requires importable flashinfer, "
            f"flashinfer.jit.env, and tvm_ffi packages: {exc}"
        ) from exc

    if _has_flashinfer_jit_cache():
        import flashinfer_jit_cache

        _check_version_match(
            flashinfer_version=getattr(flashinfer, "__version__", ""),
            jit_cache_version=getattr(flashinfer_jit_cache, "__version__", ""),
        )

    aot_dir = Path(jit_env.FLASHINFER_AOT_DIR)
    so_path = aot_dir / "gemm_sm100" / "gemm_sm100.so"
    if not so_path.exists():
        raise FlashInferPreflightError(
            "FlashInfer grouped GEMM requires the SM100 GEMM AOT module, "
            f"but it was not found at {so_path}. "
            "Install the CUDA-version-matched flashinfer-jit-cache package, e.g. "
            "`pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130` "
            "for CUDA 13.x containers."
        )

    try:
        tvm_ffi.load_module(str(so_path))
    except Exception as exc:
        raise FlashInferPreflightError(
            "FlashInfer SM100 GEMM AOT module exists but failed to load. "
            f"path={so_path}; error={exc}. "
            "This usually means the flashinfer-jit-cache CUDA build does not match "
            "the container runtime. For CUDA 13.x containers use the cu130 wheel, "
            "not cu129."
        ) from exc


def _check_version_match(*, flashinfer_version: str, jit_cache_version: str) -> None:
    if not flashinfer_version or flashinfer_version == "0.0.0+unknown":
        return
    if jit_cache_version.startswith(flashinfer_version):
        return
    raise FlashInferPreflightError(
        "flashinfer-jit-cache version does not match flashinfer-python. "
        f"flashinfer={flashinfer_version}, flashinfer_jit_cache={jit_cache_version}."
    )


def _has_flashinfer_jit_cache() -> bool:
    return importlib.util.find_spec("flashinfer_jit_cache") is not None
