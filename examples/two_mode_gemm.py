"""Two-mode timing demo: GEMM (matmul) with a torch.compile warmup.

Demonstrates the kernel-agnostic ``time_runnable_two_mode`` API on a
GEMM kernel. The setup hook runs ``torch.compile`` once (which carries a
heavy first-call cost on certain backends), then ``run()`` invokes the
compiled callable on every iter.

Expected behavior:

* ``e2e_ms``        — moderate (clone + recompile-cache-hit + run); the
                      compile cost itself amortizes via PyTorch's bytecode
                      cache so the *first* iteration is the only slow one.
* ``kernel_ms``     — pure cuBLAS GEMM time via CUDA graph replay.
* ``kernel_gpu_ms`` — same kernel observed through CUPTI activity sum.

The two kernel metrics should agree within a few µs; if they disagree
that's diagnostic (CUDA-graph capture issue or CUPTI span-vs-sum gap).

Run::

    python examples/two_mode_gemm.py
"""

from __future__ import annotations

import torch

from flashinfer_bench.bench.timing import time_runnable_two_mode
from flashinfer_bench.compile import Runnable, RunnableMetadata


def build_gemm_runnable() -> Runnable:
    """Wrap ``torch.matmul`` as a two-phase Runnable.

    ``setup`` warms a ``torch.compile``'d closure and returns it as kernel
    state. ``run`` receives the compiled fn via ``**state`` kwargs and uses
    it for the matmul. This pattern is useful when a kernel has a heavy
    one-time compile or cache-prime cost that should not pollute the
    kernel-only measurements.
    """

    def setup(a: torch.Tensor, b: torch.Tensor):
        compiled = torch.compile(torch.matmul, dynamic=False)
        # Prime the compile cache so the first run() call is not a cold path.
        _ = compiled(a, b)
        torch.cuda.synchronize(a.device)
        return {"compiled": compiled}

    def run(a: torch.Tensor, b: torch.Tensor, *, compiled):
        return compiled(a, b)

    metadata = RunnableMetadata(
        build_type="python",
        definition_name="gemm",
        solution_name="torch_matmul_compiled_two_mode_demo",
        destination_passing_style=False,
    )
    return Runnable(callable=run, metadata=metadata, setup_callable=setup)


def main() -> None:
    device = "cuda:0"
    m, k, n = 4096, 4096, 4096
    dtype = torch.float16

    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    runnable = build_gemm_runnable()
    args = [a, b]

    metrics = time_runnable_two_mode(
        runnable, args, warmup=5, iters=20, device=device, graph_iters=20
    )

    flops = 2.0 * m * k * n
    tflops_kernel = flops / (metrics.kernel_ms * 1e9) if metrics.kernel_ms > 0 else 0.0

    print(f"GEMM {m}x{k}x{n} {dtype} (two-mode timing)")
    print(f"  e2e_ms        = {metrics.e2e_ms:.4f}  (clone + setup + run per iter)")
    print(f"  kernel_ms     = {metrics.kernel_ms:.4f}  [{metrics.kernel_ms_status}]")
    print(f"  kernel_gpu_ms = {metrics.kernel_gpu_ms:.4f}  [{metrics.kernel_gpu_ms_status}]")
    print(f"  kernel TFLOPS = {tflops_kernel:.2f}")


if __name__ == "__main__":
    main()
