"""Two-mode timing demo: FlashInfer paged-prefill attention.

Demonstrates the kernel-agnostic ``time_runnable_two_mode`` API on an
attention kernel where the wrapper ``.plan()`` call is non-trivial
(metadata building, scheduler setup, workspace partitioning).

Setup hook does ``.plan()`` once; ``run()`` calls ``.run(q, kv_cache)``
on every iter. Expected: ``e2e_ms`` >> ``kernel_ms`` (since e2e clones
inputs + re-runs ``.plan()`` per iter) and ``kernel_ms`` ≈ ``kernel_gpu_ms``
(the FlashInfer attention path is graph-capturable).

Run::

    python examples/two_mode_attention.py
"""

from __future__ import annotations

import flashinfer
import torch

from flashinfer_bench.bench.timing import time_runnable_two_mode
from flashinfer_bench.compile import Runnable, RunnableMetadata


def build_attention_runnable(
    workspace_buffer: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    causal: bool = True,
) -> Runnable:
    """Wrap a FlashInfer paged-prefill wrapper as a two-phase Runnable.

    The wrapper instance is captured in a closure so the same ``.plan()``-d
    state survives between ``setup`` and ``run``. ``setup`` returns an empty
    dict because the kernel state lives in the wrapper itself, not in kwargs.
    """
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

    def setup(q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        prefill_wrapper.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
        )
        return {}

    def run(q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        return prefill_wrapper.run(q, kv_cache)

    metadata = RunnableMetadata(
        build_type="python",
        definition_name="paged_prefill_attention",
        solution_name="flashinfer_two_mode_demo",
        destination_passing_style=False,
    )
    return Runnable(callable=run, metadata=metadata, setup_callable=setup)


def main() -> None:
    device = "cuda:0"
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size, max_num_pages = 1, 128
    batch_size, nnz_qo = 7, 100

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    runnable = build_attention_runnable(
        workspace_buffer, num_qo_heads, num_kv_heads, head_dim, page_size
    )

    q = torch.randn(nnz_qo, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    qo_indptr = torch.tensor([0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor(
        [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
    )

    args = [q, kv_cache, qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len]
    metrics = time_runnable_two_mode(
        runnable, args, warmup=5, iters=20, device=device, graph_iters=20
    )

    print("FlashInfer paged-prefill attention (two-mode timing)")
    print(f"  e2e_ms        = {metrics.e2e_ms:.4f}  (clone + plan + run per iter)")
    print(f"  kernel_ms     = {metrics.kernel_ms:.4f}  [{metrics.kernel_ms_status}]")
    print(f"  kernel_gpu_ms = {metrics.kernel_gpu_ms:.4f}  [{metrics.kernel_gpu_ms_status}]")


if __name__ == "__main__":
    main()
