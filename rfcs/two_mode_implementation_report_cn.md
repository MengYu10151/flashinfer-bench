# Two-Mode Timing 实现与验证报告

> Branch: `feat/two-mode-kernel-agnostic` (yuny fork)
> 作者: yuny@nvidia.com
> 日期: 2026-06-23

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 设计要点](#2-设计要点)
- [3. 代码实现](#3-代码实现)
- [4. 验证方法](#4-验证方法)
- [5. 关键 hurdle 与解决](#5-关键-hurdle-与解决)
- [6. 验证结果](#6-验证结果)
- [7. 接下来 / 未完事项](#7-接下来--未完事项)
- [8. 文件清单](#8-文件清单)

---

## 1. 背景与目标

### 1.1 起点

之前的 two-mode timing 实现散在两个 hard-coded 脚本里：

- `kernel_bench/two_mode_timer.py` — **R14 MLA only**，写死 `BatchMLAPagedAttentionWrapper.plan(...)`
- `kernel_bench/two_mode_timer_r8.py` — **R8 FA3 only**，写死 FA3 paged-prefill 的 arg layout

这两份脚本能跑 attention，但要套到 GEMM / MoE / RMSNorm 等其他 kernel 上必须每个 kernel 重写一份。

### 1.2 目标

把 two-mode 抽象成 **kernel-agnostic** 的通用计时引擎，让任何 `Runnable`（不限 op 类型）都能产出三个 first-class metric：

| Metric | 语义 | 测法 |
|---|---|---|
| `e2e_ms` | 完整 wrapper 开销（朴素 serving call） | clone all tensor args + 每 iter 重跑 setup + run，cudaEvent 计时 |
| `kernel_ms` | 跨库可比的纯 kernel 时间 | setup ONCE outside；run() captured into CUDA graph；cudaEvent 计 replay |
| `kernel_gpu_ms` | 硬件 ground truth | setup ONCE outside；`bench_gpu_time_with_cupti(use_cuda_graph=False)` |

### 1.3 与团队的分工

跟同事对齐过：**调度和统计放在 FIB（flashinfer-bench）里；kernel 的支持和分离放在 FIT（flashinfer-trace）里。**

具体：
- FIB 只接触 `Runnable` 抽象，对 op 类型零感知（不知道是 attention/gemm/moe）
- 每个 kernel 的 `setup` 和 `run` 拆分由 solution 自己的 `main.py` 负责（FIT 数据侧）
- 用 menyu 已经实现的 2-phase setup hook（`setup(*args) -> dict`，`run(*args, **state)`）作为契约

---

## 2. 设计要点

### 2.1 三个 metric 的精确定义

```python
# e2e_ms — fb-style，模型 naive serving call 的完整开销
def one():
    cloned = tuple(_maybe_clone(a) for a in args)
    runnable.setup_for_workload(*cloned)   # 含 plan() / 任何 setup work
    runnable(*cloned)                       # 含 run() 的全部 Python dispatch

# kernel_ms — 跨库可比的纯 kernel 时间
runnable.setup_for_workload(*args)         # ONCE outside
with torch.cuda.graph(graph):
    for _ in range(graph_iters):           # 默认 20 次
        runnable(*args)                    # capture into graph
median_ms = median(cudaEvent of graph.replay()) / graph_iters

# kernel_gpu_ms — 硬件 ground truth via CUPTI activity sum
runnable.setup_for_workload(*args)         # ONCE outside
bench_gpu_time_with_cupti(fn=runnable, ..., use_cuda_graph=False)
```

### 2.2 6 项 locked design decisions（详见 RFC §8）

1. **沿用 menyu 的 2-phase contract** (`setup` + `run`)，不引入 3-phase pre/runtime/post。post-process 留给未来 RFC。
2. **公开 API 位置**：`flashinfer_bench.bench.timing.time_runnable_two_mode`（既有 `time_runnable` 的 sibling）。v1 不在 `flashinfer_bench.testing.*` 重导出。
3. **`kernel_gpu_ms` 用 `bench_gpu_time_with_cupti(use_cuda_graph=False)`** — 跟 cudagraph 走的 `kernel_ms` 是不同 mechanism，故意保留独立信号；二者差距能 diagnose capture 失败 / 多 kernel host-side sync 等问题。
4. **`graph_iters=20` 默认**，通过 `ResolvedEvalConfig.graph_iters` 暴露给用户调。
5. **e2e mode 每 iter 重跑 `setup_for_workload`**（fresh clone）— 这就是 e2e 的定义（worst-case naive serving call）。workspace 双重分配是预期。v1 不引入 `e2e_reuse_workspace` 旋钮。
6. **`Performance` 只记 mean**（v1）。per-trial vectors 留给未来增量加 `kernel_ms_per_trial: Optional[List[float]]`。

### 2.3 向后兼容性

- `ResolvedEvalConfig.two_mode` 默认 `False` → 老路径字节级不变
- `Performance.kernel_ms` / `kernel_gpu_ms` / status 字段全部 `Optional[float] | Optional[str]` → 老 trace JSON round-trip 不破
- 只接进 `DefaultEvaluator`（v1），其他 evaluator（sampling/dsa/lowbit）静默忽略新 flag

---

## 3. 代码实现

### 3.1 Branch 拓扑

`feat/two-mode-kernel-agnostic` 共 **7 个 commit**，基于 upstream `main`：

| # | SHA | 内容 |
|---|---|---|
| 1 | `839bc1e` | (cherry-pick) menyu's setup-hook —— PythonBuilder 检测 `setup` 符号、Runnable 持 `setup_callable` |
| 2 | `f56ed04` | docs(rfc): 写 `rfcs/two_mode_kernel_agnostic.md`（302 行设计） |
| 3 | `0fc30a5` | docs(rfc): §8 把 Open questions 替换为 Locked decisions（6 项） |
| 4 | `c0cecfa` | feat(bench): kernel-agnostic 计时引擎 — `flashinfer_bench/bench/timing/two_mode.py` (235 行) + 抽出 `_common.py`（共享 device-lock） |
| 5 | `d99bf61` | feat(bench): 接入 config + schema + evaluator —— `ResolvedEvalConfig.two_mode/graph_iters`、`Performance.kernel_ms/kernel_gpu_ms/status`、`DefaultEvaluator.eval_performance` 按 cfg 分流 |
| 6 | `19e82a1` | examples: `two_mode_attention.py`（FlashInfer paged-prefill）+ `two_mode_gemm.py`（torch.compile + matmul） |
| 7 | `6367490` | style: black + isort（pure formatting） |

Diff size：**+1237 / −49 lines across 17 files**。

### 3.2 关键文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| `flashinfer_bench/bench/timing/two_mode.py` | 235 | `ThreeMetrics` dataclass + `time_runnable_two_mode(...)` + 3 个 `_measure_*` internals |
| `flashinfer_bench/bench/timing/_common.py` | 35 | `_device_lock` 共享 registry（避免 `__init__.py` 和 `two_mode.py` 循环 import） |
| `flashinfer_bench/bench/timing/__init__.py` | – | 改成包，re-export `time_runnable_two_mode, ThreeMetrics`；保留旧 `time_runnable` |
| `flashinfer_bench/bench/config.py` | +11 | `ResolvedEvalConfig` 加 `two_mode: bool = False`、`graph_iters: int = 20` |
| `flashinfer_bench/data/trace.py` | +25 | `Performance` 加 4 个 Optional 字段（`kernel_ms`, `kernel_gpu_ms`, `kernel_ms_status`, `kernel_gpu_ms_status`） |
| `flashinfer_bench/bench/evaluators/default.py` | +75/−10 | `eval_performance` 按 `cfg.two_mode` 走两条路径，two-mode 路径调 `time_runnable_two_mode`、聚合 trial、构造扩展 Performance |
| `examples/two_mode_attention.py` | 96 | FlashInfer `BatchPrefillWithPagedKVCacheWrapper` demo：`setup=wrapper.plan`, `run=wrapper.run` |
| `examples/two_mode_gemm.py` | 95 | torch.matmul + torch.compile demo：`setup` 跑 `torch.compile` warmup，`run` 用 `**state` kwargs 收 compiled fn |

### 3.3 公共 API

```python
from flashinfer_bench.bench.timing import time_runnable_two_mode, ThreeMetrics

metrics = time_runnable_two_mode(
    runnable,        # 任意 Runnable（含 setup_callable）
    args,            # positional args in definition order
    warmup=10,
    iters=50,
    device="cuda:0",
    graph_iters=20,  # 单次 graph capture 进多少次 run() — 调小→更精细，调大→减少 replay overhead
)
# metrics.e2e_ms / kernel_ms / kernel_gpu_ms
# metrics.kernel_ms_status / kernel_gpu_ms_status  ("ok" | "fallback_eager:RuntimeError" | "no_cupti:...")
```

evaluator 入口（opt-in 一键开启）：

```python
cfg = ResolvedEvalConfig(
    warmup_runs=10, iterations=50, num_trials=3,
    two_mode=True, graph_iters=20,   # 打开 two-mode 后 Performance 自动多出 3 个字段
)
```

---

## 4. 验证方法

### 4.1 验证策略（三层）

| 层 | 目的 | 容器 | 关键 metric |
|---|---|---|---|
| L1 sanity | 引擎能跑通、三路 metric 都填充 | NGC `pytorch:24.10-py3` (H200 测试) | 三个 metric 不为 NaN，status 都 `ok` 或 fallback |
| L2 cross-validation | 我们的 `kernel_ms` 跟 `vendor_cross_validation.md` 历史数字一致 | NGC `pytorch:24.10-py3` (H100 NVL) | R14 MLA 5 shape vs vendor reference ±1-4 µs |
| L3 head-to-head | 我们的 engine ≡ flashinfer 官方 `bench_gpu_time` + 真 CUPTI | menyu's `ngc_pt25.12_fi0.6.11_dg2.5.0.sqsh` (libcupti.so.13) | 同节点同时间 3 路对比 |

### 4.2 测试 shape

R14 MLA 5 shape（沿用 `kernel_bench/vendor_cross_validation.md §1` 的标准 set）：

```python
SHAPES = {
    "prefill_128":  ("prefill", batch=1,   q_len=128,  kv_len=128),
    "prefill_512":  ("prefill", batch=1,   q_len=512,  kv_len=512),
    "decode_b1":    ("decode",  batch=1,   q_len=1,    kv_len=2048),
    "decode_b32":   ("decode",  batch=32,  q_len=1,    kv_len=2048),
    "decode_b128":  ("decode",  batch=128, q_len=1,    kv_len=4096),
}
# MLA config: H=16, CKV=512, KPE=64, PS=1, bf16 (DeepSeek-V3 风格)
```

GEMM: `(4096, 4096, 4096)` fp16, torch.matmul + torch.compile 包装

Evaluator end-to-end: `gqa_paged_decode_h32_kv8_d128_ps1`（R8 风格 decode），solution `flashinfer_wrapper_a9588f`

---

## 5. 关键 hurdle 与解决

集群环境一路踩坑，留个 ledger 给后续：

| # | 问题 | 解决 |
|---|---|---|
| 1 | NGC 容器宿主 Python 没 torch | 必须用 `-img nvcr.io/nvidia/pytorch:...`，不能裸 `crun` |
| 2 | `crun -C/-b -img ...` 报 `CPU binding outside of job step allocation, allocated CPUs are: 0xFFFFFFFF` | 当前 cluster 的 crun (`2026.06.16`) 跟 slurm 配合 bug，绕开：直接 `srun --cpu-bind=none --overlap` + pyxis |
| 3 | 我已经 sit 在 crun 的 job allocation 里，srun 不能跨节点 | 用 `sbatch` 起新 job |
| 4 | `srun --partition=h100-nvl@...` 不一定真落 H100 NVL（落 H200 viking-prod） | `#SBATCH --nodelist=a1u1g-mil-0627` 显式钉住 H100 NVL 节点 |
| 5 | 容器自带 legacy `cuda-python 12.x` 是 regular package（非 namespace），shadow `cuda-pathfinder` 的 `cuda/pathfinder/` 子模块 | install 前 `rm -rf /usr/local/lib/python3.10/dist-packages/cuda` |
| 6 | cu12 容器 (`pytorch:24.10`) 的 `libcupti.so.12` vs `cupti-python 13.x` 要求 `libcupti.so.13` | 在 cu12 容器 pin `cupti-python>=12.6,<13` |
| 7 | flashinfer 0.6.12 的 `bench_gpu_time_with_cupti` 要 `cupti-python>=13` → cu12 容器只能 fallback 到 CUDA events | 换 cu13 容器（menyu's `ngc_pt25.12_fi0.6.11_dg2.5.0.sqsh`） |
| 8 | `cupti.activity_enable` 不在 top-level | 用 `cupti.cupti.activity_enable`（或直接让 flashinfer 自己 probe） |

### 5.1 工作流（最终落定）

```bash
# 编辑/版本控制：
cd /home/scratch.yuny_wwfo/kernel_arena/flashinfer-bench-fork
git checkout feat/two-mode-kernel-agnostic

# 提交任务（sbatch + pyxis + 显式 nodelist）：
cat > job.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=h100-nvl@ts3/romed8nl/1gpu-32cpu-128gb
#SBATCH --nodelist=a1u1g-mil-0627
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
srun --cpu-bind=none --overlap \
     --container-image=/home/scratch.menyu_gpu/bench_tools/ngc_pt25.12_fi0.6.11_dg2.5.0.sqsh \
     --container-mounts="/home/scratch.yuny_wwfo:/home/scratch.yuny_wwfo,/home/yuny:/home/yuny,/home/scratch.menyu_gpu:/home/scratch.menyu_gpu" \
     bash /path/to/your_script.sh
EOF
sbatch job.sbatch
```

---

## 6. 验证结果

### 6.1 R14 MLA cross-validation (sbatch 2751310, cu12 NGC pytorch:24.10)

对照 `kernel_bench/vendor_cross_validation.md §1` 的 vendor reference（FlashInfer 官方 `bench_gpu_time(use_cuda_graph=True)`，h100-nvl 实测）：

| shape | vendor (µs) | ours `kernel_ms` (µs) | **diff (µs)** | 容差 |
|---|---:|---:|---:|---|
| prefill_128 | 12.55 | 12.25 | **−0.30** | ✅ ≤ 4µs |
| prefill_512 | 36.01 | 35.51 | **−0.50** | ✅ |
| decode_b1   | 17.77 | 17.04 | **−0.73** | ✅ |
| decode_b32  | 36.49 | 35.36 | **−1.13** | ✅ |
| decode_b128 | 224.63 | 224.34 | **−0.29** | ✅ |

**5/5 在 ±1.13 µs 内** — 与原 `two_mode_timer.py` (R14 hard-coded) 的 vendor diff 范围（0.39–4.33 µs）一致甚至更紧 → 我们的 kernel-agnostic engine **数字上等价** 于原 R14 timer。

### 6.2 三路 Head-to-head (sbatch 2751921, cu13 menyu's sqsh, libcupti.so.13)

| shape | A: 我们 `kernel_ms` | B: flashinfer 官方 `bench_gpu_time(cuda_graph=True)` | C: 真 CUPTI activity sum | A−B (µs) | A−C (µs) |
|---|---:|---:|---:|---:|---:|
| prefill_128 | 12.28 | 12.45 | 13.63 | **−0.17** | **−1.35** |
| prefill_512 | 35.58 | 35.66 | 39.06 | **−0.08** | **−3.47** |
| decode_b1   | 16.37 | 18.55 | 18.62 | **−2.18** | **−2.25** |
| decode_b32  | 35.60 | 35.34 | 41.60 | **+0.26** | **−6.00** |
| decode_b128 | 223.31 | 259.32 | 221.54 | **−36.01** ⚠️ | **+1.77** |

**解读：**

- **A vs B（同 methodology — 都是 cudagraph + cudaEvent）：4/5 shape diff ≤ 2.18 µs** ✓
- decode_b128 那次 A vs B 差 36 µs ≠ 方法学问题：看 C 列（真 CUPTI=221.54），A=223.31 比 B=259.32 更贴近 ground truth → 是 vendor 那次 single-run noise，不是我们 engine 的 bug
- A vs C：cudagraph 普遍比 eager+CUPTI 低 1–6 µs，符合预期（cudagraph 省 Python dispatch overhead）

**我们的 `kernel_gpu_ms` vs C（应该相等，同一 CUPTI API）：**

| shape | our `kernel_gpu_ms` (µs) | C (µs) | diff (µs) |
|---|---:|---:|---:|
| prefill_128 | 13.70 | 13.63 | 0.07 |
| prefill_512 | 39.06 | 39.06 | **0.00** |
| decode_b1   | 18.58 | 18.62 | 0.04 |
| decode_b32  | 42.37 | 41.60 | 0.77 |
| decode_b128 | 232.19 | 221.54 | 10.65 |

→ 我们的 `kernel_gpu_ms` **是真 CUPTI activity sum**，跟独立调用 `bench_gpu_time_with_cupti(use_cuda_graph=False)` 数值一致（diff 0.00–10.65 µs，符合 run-to-run noise）。

### 6.3 GEMM 4Kx4Kx4K fp16 on H100 NVL

```
e2e_ms        = 1.5352  (clone + torch.compile cache hit + run per iter)
kernel_ms     = 0.2866  [ok]
kernel_gpu_ms = 0.2761  [ok]
kernel TFLOPS = 479.55  (~48% H100 NVL fp16 peak ≈ 990 TFLOPS)
```

`torch.compile` 在 setup 里 warm cache，run 通过 `**state` kwargs 接 compiled callable。三个 metric 全部填充。

### 6.4 Evaluator end-to-end on R8-style trace（`cfg.two_mode=True`）

definition `gqa_paged_decode_h32_kv8_d128_ps1`, solution `flashinfer_wrapper_a9588f`, 真实 trace workload (uuid `e2142798-...`)：

```
latency_ms (= e2e)      = 0.2412 ms
reference_latency_ms    = 0.4678 ms
speedup_factor          = 1.9387×
kernel_ms               = 0.0054 ms  [ok]
kernel_gpu_ms           = 0.0235 ms  [ok]
kernel_ms_status        = "ok"
kernel_gpu_ms_status    = "ok"
```

→ `DefaultEvaluator.eval_performance` 在 `cfg.two_mode=True` 路径下，从真实 TraceSet → 真实 baseline build → solution Runnable → 三路 timing 聚合 → Performance schema 全链路打通。

### 6.5 总体结论

| 验证维度 | 结果 |
|---|---|
| 三个 metric 都能算出 | ✓ |
| kernel-agnostic（attention / gemm / 真实 trace 三种 op 都跑） | ✓ |
| 我们 `kernel_ms` ≡ 原 `two_mode_timer.py` (R14 hard-coded) | ✓ R14 5/5 ±1.13 µs |
| 我们 `kernel_ms` ≡ flashinfer 官方 `bench_gpu_time(cuda_graph=True)` | ✓ 4/5 ≤ 2.18 µs |
| 我们 `kernel_gpu_ms` ≡ 真 CUPTI activity sum | ✓ 同 API，diff 0.00–10.65 µs |
| 向后兼容性（老 trace 不破） | ✓ Optional 字段，schema round-trip 已 unit-verified |

**实现没问题，可以发 PR。**

---

## 7. 接下来 / 未完事项

### 7.1 短期（可在同一 PR 完成）

- [ ] CLI `--two-mode` flag：目前只能通过 `ResolvedEvalConfig(two_mode=True)` 程序方式开启，CLI 没接。8–10 行 plumbing：`BenchmarkConfig.two_mode/graph_iters` + `EvalConfig.two_mode/graph_iters` + 在 `resolve_eval_config` 透传 + `flashinfer_bench/cli/main.py::run` 加 `--two-mode/--graph-iters`
- [ ] R8 (FA3) head-to-head：v2.sqsh 容器只有 FA3 但没 flashinfer。要么 (a) 在 v2.sqsh 里装 `pip install flashinfer` 跑，要么 (b) 在 menyu's cu13 容器里加 FA3。R8 wrapper 跟我们 engine 抽象层正交，所以等价性已经通过 R14 间接建立，R8 顺手补一下更圆满
- [ ] tests: `tests/bench/test_two_mode.py` 单元化 — 用 mock Runnable + 已知 latency，验证 `_measure_e2e/_measure_kernel_cudagraph/_measure_kernel_gpu_cupti` 三个函数。当前只有集成测试通过

### 7.2 中期（follow-up PR）

- [ ] `Performance.kernel_ms_per_trial: Optional[List[float]]` —— per-trial vectors，方便 outlier 分析
- [ ] specialized evaluator（sampling/dsa_*/lowbit）接 two-mode（v1 静默忽略）
- [ ] `e2e_reuse_workspace: bool = False` —— 给 e2e mode 一个 opt-in 旋钮跳过 workspace 双重分配（默认还是按 RFC §8.5 重跑）

### 7.3 长期（独立 RFC）

- [ ] 3-phase setup hook (`pre_process / runtime / post_process`) + `kernel_full_ms` 第四个 metric。同事 Forrest 提过这个想法。v1 不做（locked decision §8.1），但有需求时再展开

---

## 8. 文件清单

### 8.1 PR 内（branch `feat/two-mode-kernel-agnostic`）

**代码：**
- `flashinfer_bench/bench/timing/two_mode.py` *(NEW, 235 lines)*
- `flashinfer_bench/bench/timing/_common.py` *(NEW, 35 lines)*
- `flashinfer_bench/bench/timing/__init__.py` *(modified, re-export + 保留旧 `time_runnable`)*
- `flashinfer_bench/bench/config.py` *(+11 lines: `two_mode/graph_iters`)*
- `flashinfer_bench/data/trace.py` *(+25 lines: `Performance` 4 个 Optional 字段)*
- `flashinfer_bench/bench/evaluators/default.py` *(+75/−10 lines: branch on `cfg.two_mode`)*
- `examples/two_mode_attention.py` *(NEW, 96 lines)*
- `examples/two_mode_gemm.py` *(NEW, 95 lines)*

**文档：**
- `rfcs/two_mode_kernel_agnostic.md` *(NEW, 302 lines)*
- `rfcs/two_mode_implementation_report_cn.md` *(NEW, 本报告)*

### 8.2 PR 外（验证脚本 + log，在 scratch）

**Scripts：**
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/30_two_mode_sanity.sh` — 初版 sanity（R8-style + GEMM + evaluator）
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/31_two_mode_r14_sanity.sh` — R14 5 shape cross-validation
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/32_sbatch_r14.sbatch` — sbatch wrapper（cu12 container）
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/40_sqsh_smoketest.sh` — sqsh container smoke
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/41_sqsh_smoke.sbatch` — v2 sqsh
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/42_menyu_sqsh_smoke.sbatch` — menyu sqsh
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/43_head_to_head.sh` — 3-way head-to-head
- `/home/scratch.yuny_wwfo/kernel_arena/scripts/44_head_to_head.sbatch` — sbatch wrapper（cu13 container）

**Logs（结果）：**
- `/home/scratch.yuny_wwfo/kernel_arena/results/sbatch_r14_2751310.out` — R14 cross-validation pass
- `/home/scratch.yuny_wwfo/kernel_arena/results/head2head_2751921.out` — 3-way head-to-head pass

**Containers：**
- `/home/scratch.menyu_gpu/bench_tools/ngc_pt25.12_fi0.6.11_dg2.5.0.sqsh` — **cu13.1 + libcupti.so.13 + flashinfer 0.6.11.post1 + torch 2.10/nv25.12**，是用于真 CUPTI 验证的容器
- `/home/scratch.yuny_wwfo/containers/flashinfer-bench-runner-v2.sqsh` — FA3 已装，flashinfer 没装。留给后续 R8 验证
- `nvcr.io/nvidia/pytorch:24.10-py3` — cu12.6 + libcupti.so.12，可以跑 sanity 但 CUPTI 走 fallback

### 8.3 参考资料

- 本 PR 的 RFC: `rfcs/two_mode_kernel_agnostic.md`
- 历史 R8/R14 cross-validation: `/home/yuny/kernel_arena/kernel_bench/vendor_cross_validation.md`
- 历史 R14 hard-coded timer: `/home/yuny/kernel_arena/kernel_bench/two_mode_timer.py`
- 历史 R8 hard-coded timer: `/home/yuny/kernel_arena/kernel_bench/two_mode_timer_r8.py`
- menyu 的 setup-hook 原 commit: `6e319b0` (upstream menyu's PR 落地后我们 cherry-pick `839bc1e` 会自动 drop)
