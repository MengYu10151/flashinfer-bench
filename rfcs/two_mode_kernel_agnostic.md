# RFC: Kernel-Agnostic Two-Mode Timing for flashinfer-bench

**Author**: yuny  
**Branch**: `feat/two-mode-kernel-agnostic`  
**Stacked on**: menyu's `feat/setup-hook` (cherry-picked as commit 1)  
**Status**: Draft — pending review

---

## 1. Motivation

The existing `kernel_bench/two_mode_timer*.py` reference timers (committed in PR #1
on branch `feat/two-mode-timing`) are **attention-specific**:

- `two_mode_timer.py` hard-codes `BatchMLAPagedAttentionWrapper.plan(...)` (R14)
- `two_mode_timer_r8.py` hard-codes the FA3 paged-prefill arg layout (R8)
- Both live outside `flashinfer_bench/` and are not wired into the trace evaluator

Now that we want the same three first-class metrics — `e2e_ms`, `kernel_ms`,
`kernel_gpu_ms` — for **GEMM**, **grouped-GEMM**, **MoE**, **sampling** and any
future op type, we need to push two-mode timing into the kernel-agnostic
benchmarking layer rather than maintain one timer per family.

The good news: menyu's `feat/setup-hook` already gives us the abstraction we
need — a per-workload `setup()` callable that is invoked once outside the timed
region and whose returned dict is splatted as kwargs into every `run()` call.
This is exactly the **plan / run split** we need to support kernel mode timing
for any solution.

## 2. Non-goals

- Modify the trace JSON schema in a breaking way. New optional fields only.
- Change existing single-metric timing path. Two-mode is **opt-in via config**.
- Touch flashinfer-trace (`definitions/`, `solutions/`, `workloads/`) at all.
- Replace `bench_gpu_time_with_cupti` as the source of `kernel_gpu_ms`. We
  reuse it; we only add the `e2e_ms` + cudagraph-based `kernel_ms` paths around it.

## 3. Design

### 3.1 Three first-class metrics

| Metric | Setup | Per-iter work | Mechanism | Audience |
|--------|-------|---------------|-----------|----------|
| `e2e_ms` | inside timed region | `setup()` + `run()`, **clone all inputs each iter** | cudaEvent (eager dispatch) | Serving / model owners — full Python+CPU+GPU cost |
| `kernel_ms` | once outside | `run()` only, no clone, captured in CUDA graph | cudaEvent on graph replay | Library benchmarks — cross-library-comparable pure work |
| `kernel_gpu_ms` | once outside | `run()` only, no clone | CUPTI activity sum (hardware counter) | Ground truth for sanity / regression detection |

All three operate on the **same `Runnable`** that the existing evaluator already
builds. The only differentiator is what happens around the call.

### 3.2 Kernel-agnostic by construction

The current attention-specific timers carry their own `build()` / `plan()` /
`run()` paths. The kernel-agnostic engine instead consumes:

- `runnable: Runnable` — already has setup (`_setup_callable`) wired by the
  builder if the solution exports a top-level `setup` symbol
- `args: List[Any]` — positional args in the order the evaluator already passes
- `cfg: ResolvedEvalConfig` — for `warmup_runs`, `iterations`

A solution that opts into two-mode follows the **exact same authoring
convention as menyu's setup-hook**:

```python
# solutions/<author>/<op>/<workload>/<name>/main.py
def setup(*args):
    # Build derived state once per workload — wrappers, plan handles,
    # workspace tensors, CSR indptr, expert ids, etc.
    return {"key": value, ...}   # dict, splatted as kwargs into run()

def run(*args, *, key, ...):     # kwonly params reserved for setup-injected state
    # The hot path. No data clones, no plan rebuilds. Just the kernel work.
    ...
```

This is exactly the contract the colleague's grouped-GEMM baseline already
uses (`flashinfer-trace/solutions/baseline/grouped_gemm/.../main.py` —
`setup()` computes `m_indptr`, `run()` consumes it as kwarg). It already works
unchanged for attention (a wrapper `setup()` returns `{"wrapper": w}`).

### 3.3 New module: `flashinfer_bench/bench/timing/two_mode.py`

```python
# flashinfer_bench/bench/timing/two_mode.py (new)
@dataclass(frozen=True)
class ThreeMetrics:
    e2e_ms: float
    kernel_ms: float
    kernel_gpu_ms: float
    kernel_gpu_status: str           # "ok" / "no_cupti" / "fallback_cudagraph" / ...
    kernel_ms_status: str            # "ok" / "no_cudagraph_fallback_eager" / ...

def time_runnable_two_mode(
    runnable: Runnable,
    args: List[Any],
    warmup: int,
    iters: int,
    device: str,
    *,
    graph_iters: int = 20,
    e2e_iters: Optional[int] = None,   # defaults to iters
) -> ThreeMetrics:
    """Measure e2e/kernel/kernel_gpu in one call."""
```

Internals (sketch):

```python
def time_runnable_two_mode(runnable, args, warmup, iters, device, **kw):
    e2e_ms        = _measure_e2e(runnable, args, warmup, iters, device)
    kernel_ms,    \
      kern_status = _measure_kernel_cudagraph(runnable, args, warmup, kw["graph_iters"], iters, device)
    kgpu_ms,      \
      kgpu_status = _measure_kernel_gpu_cupti(runnable, args, warmup, iters, device)
    return ThreeMetrics(e2e_ms, kernel_ms, kgpu_ms, kgpu_status, kern_status)
```

Each `_measure_*` is kernel-agnostic: they all just call `runnable.setup_for_workload(*args)`
(when permitted) and `runnable(*args)` — no knowledge of attention / gemm / moe.

#### 3.3.1 `_measure_e2e`: full-wrapper cost
```python
def _measure_e2e(runnable, args, warmup, iters, device):
    # Each iter clones args AND re-invokes setup() inside the timed region.
    # This mirrors the "fb-style" wall-clock cost a real serving call would pay.
    def one():
        cloned = _deep_clone_tensors(args)
        runnable.setup_for_workload(*cloned)
        runnable(*cloned)
    for _ in range(warmup): one()
    torch.cuda.synchronize(device)
    return _median_cudaevent(one, iters)
```

#### 3.3.2 `_measure_kernel_cudagraph`: cross-library-comparable kernel time
```python
def _measure_kernel_cudagraph(runnable, args, warmup, graph_iters, iters, device):
    runnable.setup_for_workload(*args)        # once outside
    for _ in range(warmup): runnable(*args)
    torch.cuda.synchronize(device)
    try:
        graph = _capture_into_cudagraph(lambda: runnable(*args), graph_iters)
        return _median_cudaevent(lambda: graph.replay(), iters) / graph_iters, "ok"
    except (RuntimeError, NotImplementedError) as ex:
        # Fallback: eager dispatch (a few µs higher; logged via status)
        return _median_cudaevent(lambda: runnable(*args), iters), f"fallback_eager:{type(ex).__name__}"
```

#### 3.3.3 `_measure_kernel_gpu_cupti`: hardware ground truth
```python
def _measure_kernel_gpu_cupti(runnable, args, warmup, iters, device):
    runnable.setup_for_workload(*args)
    try:
        ms_list = bench_gpu_time_with_cupti(fn=runnable, dry_run_iters=warmup,
                                            repeat_iters=iters, input_args=tuple(args),
                                            cold_l2_cache=True, use_cuda_graph=False)
        return statistics.median(ms_list), "ok"
    except Exception as ex:
        return 0.0, f"no_cupti:{type(ex).__name__}"
```

Note: we deliberately use the **sum** of per-kernel activity time, not the
span (= `max_end - min_start`). For multi-kernel `run()`s the span erroneously
includes Python eager-dispatch gaps between successive kernel launches; sum
gives the actual GPU exec time. `bench_gpu_time_with_cupti` already does this.

### 3.4 Existing `timing.py` → `timing/` package

To avoid breaking imports, move existing `bench/timing.py` to
`bench/timing/__init__.py` (re-exports `time_runnable`) and add
`bench/timing/two_mode.py` for the new function. Public surface unchanged:

```python
from flashinfer_bench.bench.timing import time_runnable                  # legacy single-metric
from flashinfer_bench.bench.timing import time_runnable_two_mode         # new three-metric
```

### 3.5 Evaluator opt-in

Single new field on `ResolvedEvalConfig`:

```python
class ResolvedEvalConfig(...):
    two_mode: bool = False
    """Opt-in: collect e2e_ms + kernel_ms + kernel_gpu_ms instead of a single latency_ms."""
    graph_iters: int = 20
    """When two_mode=True: number of run() calls captured per CUDA graph."""
```

In `default.py::eval_performance` the branch is one block:

```python
if cfg.two_mode:
    metrics = time_runnable_two_mode(sol_runnable, args, cfg.warmup_runs,
                                     cfg.iterations, device, graph_iters=cfg.graph_iters)
    sol_three_metrics.append(metrics)        # collect across trials
else:
    ms = time_runnable(sol_runnable, args, cfg.warmup_runs, cfg.iterations, device)
    sol_latencies.append(ms)
```

After the loop, `two_mode` mode populates the new fields on `Performance`:

```python
performance = Performance(
    latency_ms       = mean(e2e_ms_list),     # serving-shaped latency (= e2e)
    kernel_ms        = mean(kernel_ms_list),  # NEW, cross-library-comparable
    kernel_gpu_ms    = mean(kgpu_ms_list),    # NEW, CUPTI hardware sum
    reference_latency_ms = ref_mean_latency_ms,
    speedup_factor   = ref_mean_latency_ms / mean(e2e_ms_list),
)
```

### 3.6 `Performance` schema extension

```python
class Performance(BaseModelWithDocstrings):
    latency_ms: float = Field(default=0.0, ge=0.0)
    reference_latency_ms: float = Field(default=0.0, ge=0.0)
    speedup_factor: float = Field(default=0.0, ge=0.0)
    # --- NEW (optional; backward-compatible with existing trace JSONs) ---
    kernel_ms: Optional[float] = Field(default=None, ge=0.0)
    """Cross-library-comparable pure kernel time (cudagraph cudaEvent). None when two_mode=False."""
    kernel_gpu_ms: Optional[float] = Field(default=None, ge=0.0)
    """Hardware ground-truth kernel exec time (CUPTI sum). None when two_mode=False."""
```

Existing trace files that lack these fields parse correctly (pydantic
`Optional[float] = None`). Existing readers that only inspect `latency_ms`
continue to work.

## 4. Migration of `kernel_bench/two_mode_timer*.py`

The standalone scripts on `feat/two-mode-timing` become **integration tests +
demos** that drive the kernel-agnostic engine through the existing
`Runnable` + `setup()` convention:

- A new `tests/bench/test_two_mode.py` exercises both attention (R14, R8) and
  one GEMM family using the same engine.
- The `kernel_bench/` scripts are kept as runnable smoke tests for the timer
  itself but stop owning timing logic — they call `time_runnable_two_mode`.

## 5. Backward compatibility

| Concern | Behavior |
|---------|----------|
| Solution has no `setup` symbol | `setup_for_workload()` is a no-op. `e2e_ms` ≈ `kernel_ms` (no plan to amortize). Still useful as a baseline. |
| Builder is not Python (Torch / Triton / tvm-ffi) | menyu's setup-hook patch wires Python only. For other builders, `_setup_callable` is `None` and the two-mode engine degenerates as above. Adding setup support for other builders is a follow-up. |
| CUPTI unavailable | `kernel_gpu_ms = 0.0`, `kernel_gpu_status = "no_cupti:..."`. `e2e_ms` and `kernel_ms` still measured. |
| Solution incompatible with CUDA graph capture (stream-ordered allocator, etc.) | Caught; `kernel_ms` falls back to eager dispatch with `status = "fallback_eager:..."`. |
| Old trace JSON (no `kernel_ms`/`kernel_gpu_ms`) | Loads fine (Optional fields). |
| `cfg.two_mode = False` (default) | Existing `time_runnable` path unchanged. |

## 6. Verification plan

After implementation we re-run the same five-shape verification matrices already
established for R8 (FA3) and R14 (MLA) on `kernel_bench/vendor_cross_validation.md`:

- **A**: `kernel_ms` (cudagraph) vs vendor-API `bench_gpu_time(use_cuda_graph=True)`.
  Expected diff: 0.4–4.3 µs (single `cuLaunchKernelEx` boundary).
- **B**: `kernel_ms` vs `kernel_gpu_ms` (CUPTI sum). Expected diff: 1–3 µs
  except for known span-vs-sum quirks (R14 decode_b32).
- **C**: Run on a GEMM solution (e.g. `grouped_gemm_fp8_e128_n3072_k4096`) and
  cross-check `kernel_ms` against `triton.testing.do_bench` of the same closure
  for that solution. New verification data point.

Numbers go into a new `rfcs/two_mode_kernel_agnostic_verification.md` once
gathered.

## 7. PR plan

Commits, in order, on `feat/two-mode-kernel-agnostic`:

1. `feat: add per-workload setup() hook for python solutions` *(cherry-pick of
   menyu's commit — auto-drops when her PR lands upstream)*
2. `docs(rfc): kernel-agnostic two-mode timing RFC` *(this document)*
3. `feat(bench): kernel-agnostic two-mode timing module
   (flashinfer_bench/bench/timing/two_mode.py)`
4. `feat(bench): opt-in two-mode in evaluator + ResolvedEvalConfig +
   Performance schema`
5. `test(bench): two-mode tests for attention (R8, R14) and grouped_gemm`
6. `docs(rfc): verification numbers across attention + gemm families`

No changes to `flashinfer-trace/`. No changes to existing solutions or
definitions. All single-metric tests remain green.

## 8. Locked design decisions

1. **Stay with menyu's 2-phase contract** (`setup` + `run`); do *not* extend
   to a 3-phase `pre_process / runtime / post_process` split in this RFC.
   Post-process is a future-RFC topic (would add an optional third symbol +
   one extra metric `kernel_full_ms`; non-blocking for v1).
2. **Public API**: `flashinfer_bench.bench.timing.time_runnable_two_mode`
   (sibling of the existing `time_runnable`). Not re-exported under
   `flashinfer_bench.testing.*` for v1.
3. **`kernel_gpu_ms` uses `bench_gpu_time_with_cupti(use_cuda_graph=False)`**
   — gives an independent signal vs the cudagraph-based `kernel_ms`. The two
   metrics deliberately measure on different mechanisms so a discrepancy is
   diagnostic (per the methodology already validated for R8 / R14).
4. **`graph_iters=20` default**, exposed via `ResolvedEvalConfig.graph_iters`
   for tuning on micro-kernels.
5. **e2e mode reruns `setup_for_workload` per iter** with fresh clones — this
   is the definition of e2e (full wrapper cost). Workspace double-allocation
   is *expected* and represents the worst-case naive serving call. No
   `e2e_reuse_workspace` knob in v1.
6. **`Performance` records the mean only** for v1 (`kernel_ms`,
   `kernel_gpu_ms`). Per-trial vectors are out of scope; can be added later as
   `kernel_ms_per_trial: Optional[List[float]]` without breaking compat.
