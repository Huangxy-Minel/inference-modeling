# inference-modeling

Roofline-based inference performance + capacity simulator for large language
models. Models the **decode-phase** cost (FLOPs / bytes / arithmetic intensity
/ MFU / TPOT) and the **memory capacity** envelope (HBM-only baseline +
pluggable optimizations like DRAM Pooling) for modern MoE architectures
(DeepSeek **V3 / V3.2 / V4-Pro / V4-Flash**, **GLM-5** and **GLM-5.2**
(asymmetric MLA + DSA, GLM-5.2 adds IndexShare)) on recent accelerators
(**A100 / H100 / H200 / GB200 / GB300 / B200 / H20 / Ascend 910C**).

The simulator is fully analytical (no GPU calls, no RNG): re-running any
sweep produces byte-identical reports between runs.

## What this simulator answers

1. **Per-step performance** — given (model, GPU, parallel config, workload),
   compute the per-stage and total `(FLOPs, Bytes, AI, attainable TFLOPs,
   MFU, wall-clock, TPOT)` from the Roofline model.

2. **Capacity-bounded batch size** — given HBM capacity, derive the maximum
   `batch_per_GPU` after subtracting model weights and per-session KV cache
   for a given `session_length` (1 K – 4 M tokens).

3. **DRAM-Pool optimization trade-offs** — assess two scenarios:
   - **Sparse-on-demand** (Scenario 1): full KV in DRAM, on-miss layer-stride
     fetch. Sweep over `hit_rate ∈ [0, 1]`, DRAM bandwidth (PCIe Gen5 / RDMA
     / NVLink-C2C), and indexer residency (HBM vs DRAM).
   - **Shared-prefix prefetch** (Scenario 2): single shared prefix in DRAM,
     unique-KV in HBM with **automatic layer-stripe spill** when HBM
     overflows. Sweep over `prefix_share_frac` and DRAM bandwidth.

4. **Sweep matrices** — markdown / HTML reports with three-color heat maps
   for quick visual reading.

## Project layout

```
inference-modeling/
├── simulator/                          ← Python package (importable as `simulator`)
│   ├── core.py                         ← GPU spec, Roofline, PerfReport, estimate_perf
│   ├── capacity.py                     ← MemoryProfile, CapacityReport, analyze_capacity
│   ├── models/
│   │   ├── base.py                     ← ModelCostEstimator ABC + MODEL_REGISTRY
│   │   │                                  (+ per-GPU calibrated fixed HBM overhead)
│   │   ├── deepseek_v3.py              ← V3 (dense MLA) + V3.2 (MLA + DSA)
│   │   ├── deepseek_v4.py              ← V4-Pro / V4-Flash (CSA + HCA + SWA, FP4 indexer)
│   │   ├── glm_v5.py                   ← GLM-5 (asymmetric MLA-256 + DSA + MoE)
│   │   └── glm_v52.py                  ← GLM-5.2 (GLM-5 + IndexShare + NoPE head_dim=192)
│   ├── optimizations/
│   │   ├── base.py                     ← CapacityOptimization ABC
│   │   └── dram_pooling.py             ← DRAM Pooling (Scenario 1 + Scenario 2)
│   └── configs/                        ← deployment YAML templates
├── scripts/                            ← CLI sweep entry points
│   ├── sweep_attn_matrix.py            ← attn metrics vs (ctx × batch)
│   ├── sweep_ffn_matrix.py             ← FFN metrics vs (EP × batch)
│   ├── sweep_e2e_matrix.py             ← end-to-end perf vs (cluster × batch)
│   └── sweep_dram_analysis.py          ← DRAM Pool: session × batch × {base, opt}
└── reports/                            ← (gitignored) generated sweep outputs
    ├── metrics_matrix/                 ← one folder per stage / GPU / model / ctx
    ├── dram_s1/                        ← Scenario 1 sweeps (hit-rate × bw)
    └── dram_s2/                        ← Scenario 2 sweeps (prefix-share)
```

## Installation

The simulator only needs `numpy` (required) and `pyyaml` (for `--config`).
A one-click installer is provided:

```bash
cd scripts/inference-modeling

./install.sh                # creates ./myenv venv and installs deps (default)
./install.sh --user         # install into the active python's user site
./install.sh --system       # install into the active python (no venv)
./install.sh --venv .venv   # use a custom venv path instead of ./myenv
```

After a venv install, activate it and you're ready:

```bash
source myenv/bin/activate           # bash / zsh
python3 -m simulator --gpu H100 --model deepseek-v3.2 \
    --tp 4 --ep 32 --dp 8 --batch 64 --ctx 131072
```

Manual install (if you'd rather skip the script):

```bash
python3 -m pip install -r requirements.txt
```

Python 3.9+ is required; the script will refuse older interpreters.

## Quick start

```bash
cd scripts/inference-modeling

# 1) attention sweep (ctx × batch); TP/EP/DP fixed
python3 -m scripts.sweep_attn_matrix \
    --gpu H100 --model deepseek-v3.2 \
    --tp 4 --ep 32 --dp 8 \
    --out reports/metrics_matrix/attn_matrix_v32_h100.md

# 2) FFN sweep (EP × batch); ctx fixed (FFN cost is N-invariant for decode)
python3 -m scripts.sweep_ffn_matrix \
    --gpu H100 --model deepseek-v3.2 \
    --tp 4 --ep 8,16,32,64,128 \
    --out reports/metrics_matrix/ffn_matrix_v32_h100.md

# 3) end-to-end matrix (cluster size × batch) at given ctx
python3 -m scripts.sweep_e2e_matrix \
    --gpu H100 --model deepseek-v4-pro --ctx 1048576 \
    --out reports/metrics_matrix/e2e_matrix_v4pro_h100_ctx1m.md

# 4) DRAM Pool — Scenario 1 (sparse on-demand), bw sweep at hit_rate=0.6
python3 -m scripts.sweep_dram_analysis \
    --gpu H100 --model deepseek-v3.2 \
    --tp 4 --ep 32 --dp 8 \
    --scenario sparse_on_demand \
    --hit-rate 0.6 --bw-sweep 50,100,200,400,800 \
    --out reports/dram_s1/dram_s1_v32_h100.md

# 5) DRAM Pool — Scenario 1, hit-rate sweep at bw=400 GB/s
python3 -m scripts.sweep_dram_analysis \
    --gpu H100 --model deepseek-v3.2 \
    --scenario sparse_on_demand \
    --dram-interconnect-bandwidth-gbs 400 \
    --hit-rate-sweep 0,0.2,0.4,0.6,0.8,0.99 \
    --out reports/dram_s1/dram_s1_v32_h100.md

# 6) DRAM Pool — Scenario 2 (shared-prefix), prefix-share sweep
python3 -m scripts.sweep_dram_analysis \
    --gpu H100 --model deepseek-v3.2 \
    --scenario shared_prefix \
    --dram-interconnect-bandwidth-gbs 400 \
    --prefix-share-sweep 0,0.2,0.4,0.6,0.8,1.0 \
    --out reports/dram_s2/dram_s2_v32_h100.md
```

`--bw-sweep`, `--hit-rate-sweep`, `--prefix-share-sweep` are mutually
exclusive (each one sets the row dimension). When using a sweep flag, the
output filename is auto-suffixed: e.g. `dram_s1_v32_h100.md` →
`dram_s1_v32_h100_hit60.md` for `hit_rate=0.6` row, etc.

`reports/` is in `.gitignore` so generated reports are not committed. To
share a snapshot, package them into an archive (see "Packaging reports"
below).

### Capacity knobs (Scenario 1 sizing accuracy)

The HBM-only / DRAM-pool batch sizing follows one forward equation:

```
hbm_avail (KV budget) = HBM_capacity − weight − (frac × HBM + fixed)
max_batch             = floor(hbm_avail / session_kv)   (or hot/cold split)
```

Three knobs make this match real serving stacks:

* **`--hot-slots N`** (Scenario 1): HBM keeps a hot window of `N`
  tokens/layer/session (across all layers); the cold tail lives in DRAM.
  Default = model top-k (`dsa_topk` / `csa_topk`), floored at top-k,
  capped at `S`. Real stacks reserve more than top-k (e.g. `4096` for a
  `2048` top-k) to lower the miss rate. Cell prints `DRAM used: X / pool`.
* **`--kv-dtype {bf16,fp8,mixed_bf16_fp8}`**: KV-cache latent precision for
  capacity sizing. `bf16` = 2 B/elem latent; `fp8`/`mixed_bf16_fp8` =
  1 B/elem latent + BF16 RoPE. Affects MLA-latent models (V3.2 / GLM-5 /
  GLM-5.2); no-op for V4 (own compressed KV).
* **`--hbm-overhead-fixed-gb G`**: FIXED HBM overhead (CUDA graph +
  DeepEP/NCCL + framework runtime reserve that never reaches the KV pool),
  on top of `--hbm-overhead-frac` (default 5%). **Default = unset →** use
  the model's per-GPU **calibrated** value
  (`ModelCostEstimator.overhead_fixed_gb_by_gpu`); falls back to 0 (with a
  warning) if the model/GPU is not calibrated. An explicit value wins.

#### Calibrating `fixed` overhead from an HBM profiling run

Precedence: **explicit CLI > model calibration > 0**. To calibrate a
(model, GPU) pair from a real HBM report (e.g. SGLang's `HBM Usage
Report`), back it out so the modelled KV budget matches the measured KV
pool:

```
fixed = HBM_capacity × (1 − frac) − weight_model − KV_pool_measured
```

Then store it in the model's `overhead_fixed_gb_by_gpu`, e.g. in
`glm_v5.py`:

```python
class GLM5Estimator(...):
    overhead_fixed_gb_by_gpu = {"H20": 19.8, "H100": 18.3}
```

Now any run of that model on that GPU auto-uses the calibrated value; the
report header shows `... GiB fixed [glm-5@H20 calibrated]`. Uncalibrated
(model, GPU) pairs show `[uncalibrated]` and use 0 unless `--hbm-overhead-
fixed-gb` is passed. Validation anchor: predicted
`KV_pool / session_kv` should match the measured admitted decode batch
(e.g. GLM-5.1-FP8 on 32×H800: 21.0 GiB / 0.393 GiB = 53 == measured 53).

### Packaging reports

`reports/` is gitignored, so to version/share a snapshot, package it into
an archive committed at the repo (not under `reports/`):

```bash
tar -czf dram_s1_reports.tar.gz -C reports dram_s1
git add dram_s1_reports.tar.gz && git commit -m "reports: dram_s1 snapshot"
```

## Inline / one-off cases (no sweep scripts)

For a quick "what does this single (model, GPU, batch, ctx) configuration
look like" question, skip the sweep scripts entirely. There are three
entry points, in increasing order of convenience:

### Option 1 — `python3 -m simulator` with inline flags

The simulator package is directly runnable; no script needed:

```bash
python3 -m simulator \
    --gpu H100 --model deepseek-v3.2 \
    --tp 4 --ep 32 --dp 8 \
    --batch 64 --ctx 131072
```

prints a per-stage Perf report (FLOPs / Bytes / AI / attainable TFLOPs /
MFU / mem-vs-compute bound / wall-clock per stage). All flags:

```
--gpu {A100,H100,H200,GB200,Ascend910C}   GPU preset
--model <registered model name>           e.g. deepseek-v3.2 / deepseek-v4-pro
--phase {prefill,decode}                  default: decode
--tp / --ep / --pp / --dp                 parallel layout (DP×TP must equal EP)
--batch                                   per-DP batch size (not cluster-wide)
--ctx                                     context length N
--new-tokens                              decode default 1; prefill auto-sets to ctx
```

### Option 2 — YAML deployment config (`--config`)

For repeatable / reviewable configs, edit a YAML in
`simulator/configs/` (templates already there for V3.2 prefill / decode
on GB200) and pass it via `--config`:

```bash
# pyyaml is already installed by ./install.sh; otherwise:
#   python3 -m pip install pyyaml
python3 -m simulator --config simulator/configs/deepseek_v3.2_decode_gb200.yaml
```

YAML schema:

```yaml
gpu: GB200                # preset name, OR inline {name, peak_tflops, bandwidth_gbs}
model: deepseek-v3.2

# Optional per-deployment overrides into model's default_model_config
# (e.g. tweak DSA top-k, KV dtype, expert count, ...)
model_overrides:
  dsa_topk: 2048

parallel:
  tp: 4
  ep: 32
  pp: 1
  dp: 1

workload:
  phase: decode             # or "prefill"
  batch_size: 64
  context_length: 9216
  new_tokens: 1
  dtype_param: fp8
  dtype_kv: mixed_bf16_fp8
```

`--config` overrides any inline flags. To programmatically run a YAML
without the CLI, use `simulator.run_from_yaml(path)`.

### Option 3 — Python REPL / `python3 -c` (most flexible)

For capacity / DRAM-Pool questions (which `python3 -m simulator` doesn't
cover, since `-m simulator` is perf-only), drop into Python:

```bash
# (a) Per-step perf — equivalent to python3 -m simulator but lets you keep going
python3 -c "
from simulator import (GPU_PRESETS, ParallelConfig, WorkloadConfig, Phase,
                       estimate_perf, format_report)
gpu      = GPU_PRESETS['H100']
parallel = ParallelConfig(tp=4, ep=32, dp=8)
workload = WorkloadConfig(phase=Phase.DECODE, batch_size=64, context_length=128*1024)

perf = estimate_perf(gpu, 'deepseek-v3.2', parallel, workload)
print(format_report(perf))
print(f'TPOT      = {perf.tpot_seconds*1000:.2f} ms')
print(f'MFU       = {perf.total.mfu*100:.1f}%')
print(f'Cluster TP = {perf.cluster_tput_tps:,.0f} tok/s')
"
```

```bash
# (b) Capacity — how many sessions fit at this ctx?
python3 -c "
from simulator import (GPU_PRESETS, MEMORY_PRESETS, ParallelConfig,
                       analyze_capacity, format_capacity_report)
gpu, mem = GPU_PRESETS['GB200'], MEMORY_PRESETS['GB200']
parallel = ParallelConfig(tp=4, ep=32, dp=8)

cap = analyze_capacity(gpu, mem, 'deepseek-v4-pro', parallel,
                       session_length=1024*1024)   # 1M ctx
print(format_capacity_report(cap))
print(f'baseline bs (HBM only) = {cap.max_batch_per_gpu}')
print(f'session_kv per GPU      = {cap.session_kv_bytes/2**30:.3f} GiB')
"
```

```bash
# (c) DRAM Pool — single (bs, hit_rate, bw) point
python3 -c "
from simulator import (GPU_PRESETS, MEMORY_PRESETS, ParallelConfig,
                       analyze_capacity,
                       DramPoolingConfig, DramPoolingOptimization)
gpu, mem = GPU_PRESETS['H100'], MEMORY_PRESETS['H100']
parallel = ParallelConfig(tp=4, ep=32, dp=8)
baseline = analyze_capacity(gpu, mem, 'deepseek-v3.2', parallel,
                            session_length=1024*1024)

cfg = DramPoolingConfig(
    dram_capacity_gb=12*1024,
    dram_interconnect_bandwidth_gbs=400.0,
    mode='sparse_on_demand',
    kv_cache_hit_rate=0.6,
    indexer_in_dram=False,
)
rep = DramPoolingOptimization(cfg).apply(baseline, gpu=gpu, mem=mem,
                                          parallel=parallel,
                                          batch_size_override=64)
print(f'baseline bs = {baseline.max_batch_per_gpu}')
print(f'opt bs      = {rep.max_batch_per_gpu}  (bound by {rep.bs_bound_by})')
print(f'penalty     = {rep.penalty_seconds*1000:.2f} ms')
print(f'speedup     = {(rep.cluster_tput_tps/baseline.cluster_tput_tps if baseline.cluster_tput_tps else 0):.2f}x')
print(f'notes       = {rep.notes}')
"
```

If a one-liner gets unwieldy, drop the same body into `python3 -i` or
a Jupyter cell — the API is identical.

## Programmatic API

### Per-step performance

```python
from simulator import (
    GPU_PRESETS, ParallelConfig, WorkloadConfig, Phase,
    estimate_perf, format_report,
)

gpu      = GPU_PRESETS["GB200"]
parallel = ParallelConfig(tp=4, ep=32, dp=8)               # decode switching parallel
workload = WorkloadConfig(phase=Phase.DECODE, batch_size=128, context_length=9216)

perf = estimate_perf(gpu, "deepseek-v4-pro", parallel, workload)
print(format_report(perf))

print(f"TPOT       = {perf.tpot_seconds * 1000:.2f} ms")
print(f"MFU        = {perf.total.mfu * 100:.1f}%")
print(f"Cluster TP = {perf.cluster_tput_tps:,.0f} tok/s")
```

### Capacity + DRAM Pool

```python
from simulator import (
    MEMORY_PRESETS, analyze_capacity,
    DramPoolingConfig, DramPoolingOptimization,
)

mem      = MEMORY_PRESETS["H100"]
baseline = analyze_capacity(gpu, mem, "deepseek-v3.2", parallel,
                            session_length=1024 * 1024)        # 1M ctx

# Scenario 1: sparse on-demand
opt1 = DramPoolingOptimization(DramPoolingConfig(
    dram_capacity_gb                = 12 * 1024,                # 12 TiB / GPU
    dram_interconnect_bandwidth_gbs = 400.0,
    mode                            = "sparse_on_demand",
    kv_cache_hit_rate               = 0.6,                      # per-layer hit
    indexer_in_dram                 = False,                    # indexer stays in HBM
))
report1 = opt1.apply(baseline, gpu=gpu, mem=mem, parallel=parallel,
                     batch_size_override=64)

# Scenario 2: shared prefix (auto-spill enabled)
opt2 = DramPoolingOptimization(DramPoolingConfig(
    dram_capacity_gb                = 12 * 1024,
    dram_interconnect_bandwidth_gbs = 400.0,
    mode                            = "shared_prefix",
    prefix_share_frac               = 0.6,                      # 60 % shared
))
report2 = opt2.apply(baseline, gpu=gpu, mem=mem, parallel=parallel)

print(f"baseline bs       = {baseline.max_batch_per_gpu}")
print(f"S1 idx-HBM bs     = {report1.max_batch_per_gpu}  (bound by {report1.bs_bound_by})")
print(f"S2 shared-prefix  = {report2.max_batch_per_gpu}, k_spilled = {report2.spilled_layers_count}")
print(f"S2 penalty        = {report2.penalty_seconds * 1000:.2f} ms")
```

## Key concepts

### Roofline + per-stage time

For each stage (attn / FFN), `evaluate_stage` returns a `StageReport` with:

```
ai                  = flops / bytes                        # FLOPs / byte
attainable_tflops   = min(peak_tflops, bandwidth × ai)
mfu                 = attainable_tflops / peak_tflops      # ∈ [0, 1]
time_seconds        = bytes / bandwidth   if memory-bound  # i.e. ai < ridge
                    = flops / peak        if compute-bound
```

Total step time = `attn.time_seconds + ffn.time_seconds` (serial). For
decode, TPOT = total step time (one token per step). Total MFU is
recomputed from `total_flops / total_time / peak`, so it is not a simple
weighted average of per-stage MFU.

### Switching parallel (decode-only constraint)

`ParallelConfig.__post_init__` enforces `DP × TP == EP`, modeling the V3 /
V3.2 / V4 deployment where the **same physical GPUs** run TP × DP for
attention and **reshape** to EP for FFN. Hence:

```
world_size = EP × PP   (= DP × TP × PP under the constraint)
```

Violating configs raise `ValueError` immediately with a fix hint.

### DRAM Pool — Scenario 1: sparse on-demand

Three capacity constraints; `bs_max` is the min of all three:

```
A. Total KV capacity:    bs × session_kv ≤ HBM_avail + DRAM
B. Indexer in HBM:       bs × indexer_kv ≤ HBM_avail   (if --indexer-in-dram=False)
C. (S2 only — see below)
```

Penalty = main-fetch + (indexer-prefetch if `indexer_in_dram=True`):

```
miss_layers   = num_layers × (1 − hit_rate)
cold_per_miss = cold_layer_per_session × n_missing_sessions
penalty_main  = miss_layers × cold_per_miss / dram_bw

# Indexer-in-DRAM path (auto roofline overlap, no user knob):
t_layer       = (attn.time + ffn.time) / num_layers
prefetch_per  = (bs × indexer_kv) / num_layers / dram_bw
overlap       = min(1, t_layer / prefetch_per)
penalty_idx   = (bs × indexer_kv / dram_bw) × (1 − overlap)
```

The two `tokens_per_miss_layer` knobs:
- `None` (default) → **model-native top-k lower bound**. V3.2 returns
  `top-k × v_token` per missed layer per session.
- `int` → measured override (e.g. 8192 for "fetch the page containing each
  top-k token" mid-granularity).
- `"page"` → **legacy upper bound** (= entire layer KV per session).
  Emits a `DeprecationWarning`: this assumes 75× page-fetch amplification
  vs sparse top-k, which has not been empirically calibrated.

The `n_missing_sessions` knob defaults to `None` → resolved at runtime to
`new_batch` (pessimistic batch-wide miss upper bound; "every session
misses simultaneously"). Set explicitly to `1` for the optimistic lower
bound.

### DRAM Pool — Scenario 2: shared-prefix prefetch (with layer-spill)

Single copy of the shared prefix lives in DRAM; the unique part of each
session lives in HBM. When HBM cannot hold the full per-batch unique KV,
the **tail k layers** (out of L) of unique KV automatically spill to DRAM
and are layer-stride prefetched alongside the shared prefix:

```
α = prefix_share_frac
shared_prefix_bytes = α × session_kv                   (single copy)
PLU                 = (1 − α) × session_kv / num_layers (per-layer unique per session)

k(bs) = ceil((bs × (1 − α) × S_kv − HBM_avail) / (bs × PLU))   ∈ [0, L]
bs_max = (HBM_avail + (DRAM − α × S_kv)) / ((1 − α) × S_kv)    (page-aligned)
```

Layer-split roofline overlap with two paths:

```
t_layer            = (attn.time + ffn.time) / L
t_pref_light       = (α × S_kv / L) / bw                 # prefix only
t_pref_heavy       = (α × S_kv / L + bs × PLU) / bw      # prefix + that layer's unique
penalty_total      = (L − k) × t_pref_light × (1 − overlap_light)
                   +   k     × t_pref_heavy × (1 − overlap_heavy)
```

`bs_bound_by` distinguishes regimes: `HBM-cap` (no spill needed),
`DRAM-spill-cap` (spill kicks in), `DRAM-prefix-cap` (prefix alone exceeds
DRAM — infeasible), `model-unbounded` (α = 1 corner), `user-override`.

## Built-in presets

### GPUs (`GPU_PRESETS`)

| GPU         | peak FP8 TFLOPs | HBM bandwidth | HBM capacity |
| ----------- | --------------: | ------------: | -----------: |
| A100        |             312 |    2,039 GB/s |        80 GB |
| H100        |           1,979 |    3,350 GB/s |        80 GB |
| H200        |           1,979 |    4,800 GB/s |       141 GB |
| GB200       |           5,000 |    8,000 GB/s |       192 GB |
| Ascend910C  |             486 |    4,000 GB/s |        96 GB |

(A100 row is FP16-dense peak; the rest are FP8-dense per spec sheet.
Ascend 910C uses FP16 dense as the "FP8-equivalent" reference.)

### Models (`MODEL_REGISTRY`)

| Model                 | Notes                                                       |
| --------------------- | ----------------------------------------------------------- |
| `deepseek-v3`         | MLA, dense attention, no DSA                                |
| `deepseek-v3.2`       | MLA + DSA (top-k = 2048), FP8 indexer K-cache               |
| `deepseek-v4-pro`     | 61L hybrid CSA / HCA, FP4 indexer, MoE 1+384 / top-6        |
| `deepseek-v4-flash`   | 43L hybrid CSA / HCA / pure-SWA, MoE 1+256 / top-6          |

Verified V4 paper §2.3 (CSA/HCA weight shapes, FP4 indexer, RoPE 64,
SWA rolling buffer); V4-Pro/V4-Flash KV-cache @ 1M ctx ≈ 10 % / 7 % of
V3.2 (matches paper Figure 1).

## Determinism & sanity workflow

The simulator is fully analytical: re-running any sweep produces
byte-identical markdown. Use `md5` to verify after refactors:

```bash
md5 reports/dram_s1/dram_s1_v32_h100_hit60.md > /tmp/before.txt
# ... make some refactor that should not change behaviour ...
python3 -m scripts.sweep_dram_analysis \
    --gpu H100 --model deepseek-v3.2 --scenario sparse_on_demand \
    --hit-rate 0.6 --bw-sweep 50,100,200,400,800 \
    --out reports/dram_s1/dram_s1_v32_h100.md
md5 reports/dram_s1/dram_s1_v32_h100_hit60.md > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt   # should be empty for a refactor-only change
```

## Adding a new GPU / model preset

GPU: edit two files —
1. `simulator/core.py::GPU_PRESETS` (add `peak_tflops`, `bandwidth_gbs`).
2. `simulator/capacity.py::MEMORY_PRESETS` (add `hbm_capacity_gb`).

The CLI `--gpu` choices are auto-derived from `GPU_PRESETS.keys()`, so no
script edits are required.

Model: subclass `ModelCostEstimator` in `simulator/models/<name>.py`,
implement `compute_attn_cost`, `compute_ffn_cost`, `session_kv_bytes`,
`indexer_kv_bytes_per_session`, `cold_layer_kv_bytes_per_session`,
`weight_bytes_per_gpu`, decorate with `@register_model("<name>")`, and
import the module in `simulator/models/__init__.py`.

## License

(Add license here when ready.)
