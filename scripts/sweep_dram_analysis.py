#!/usr/bin/env python3
"""DRAM extension analysis: HBM-only baseline vs DRAM Pooling.

Two layout modes (chosen via CLI):

    Mode B — DEFAULT, "session × batch_size × {baseline, optimized}":
        rows = session_length, cols = batch_size in --batch-sizes
            (default 4,8,16,32,64,128,256,512). Each cell holds TWO
            sub-blocks (baseline and DRAM-pooling optimized) so the same
            requested bs can be compared head-to-head:

            +---------+---------------------------+
            | sess    |    bs=64                  |
            +---------+---------------------------+
            |  1M     | base bs=4 (req=64, HBM)   |
            | (max=4) |   MFU 11% TPOT 3.9 ms     |
            |         |   1.2 M tok/s             |
            |         | opt  bs=64                |
            |         |   penalty 23 ms           |
            |         |   MFU 60% TPOT 29 ms      |
            |         |   4.5 M tok/s (3.7x)      |
            +---------+---------------------------+

        baseline DEGRADES to min(req_bs, hbm_max_batch) when HBM is too
        small for the requested bs. The actual bs used is shown explicitly
        ("bs=4 (req=64, HBM-cap)") so the comparison is honest.

    Mode A — "session sweep × {baseline, optimized}" (legacy):
        rows = session_length, cols = (baseline, optimized).
        Triggered with --batch-sizes auto.

Penalty model (per decode step, sparse on-demand scenario):

    penalty_main = num_layers * (1 - hit_rate)
                 * cold_layer_kv_per_session * n_missing_sessions
                 / dram_interconnect_bandwidth

    # Indexer-in-DRAM penalty (idx-DRAM sub-block only); overlap is
    # auto-derived from a roofline model — layer i+1's indexer prefetch
    # overlaps with layer i's full (attn + ffn) execution time, since
    # K_indexer[i+1] is independent of layer i's outputs.
    t_layer = (mfu.attn.time + mfu.ffn.time) / num_layers
    t_prefetch = bs * indexer_kv_per_session / num_layers / dram_bw
    overlap = min(1, t_layer / t_prefetch)
    penalty_indexer = bs * indexer_kv_per_session / dram_bw * (1 - overlap)

    penalty = penalty_main + penalty_indexer

See `simulator.optimizations.dram_pooling` module docstring for full discussion.

Usage (Mode B, default):
    python3 -m scripts.sweep_dram_analysis \\
        --gpu GB200 --model deepseek-v3.2 \\
        --tp 4 --ep 320 --dp 80 \\
        --dram-capacity-gb 512 --dram-interconnect-bandwidth-gbs 50 \\
        --hit-rate 0.9 \\
        --out dram_analysis.md

Usage (Mode A, legacy):
    python3 -m scripts.sweep_dram_analysis \\
        --hit-rate 0.99 --batch-sizes auto --out dram_analysis_legacy.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from simulator import (
    GPU_PRESETS,
    MEMORY_PRESETS,
    DramPoolingConfig,
    DramPoolingOptimization,
    MemoryProfile,
    ParallelConfig,
    Phase,
    WorkloadConfig,
    analyze_capacity,
    estimate_perf,
)
from simulator.capacity import CapacityReport
from simulator.core import Dtype
from simulator.models import MODEL_REGISTRY
from simulator.optimizations import OptimizedCapacityReport


def _fixed_overhead_str(args, gpu) -> str:
    """Human-readable resolved FIXED overhead for report headers, matching
    analyze_capacity's precedence: explicit CLI > model calibration > 0."""
    if args.hbm_overhead_fixed_gb is not None:
        return f"{float(args.hbm_overhead_fixed_gb):.1f} GiB fixed [explicit]"
    est = MODEL_REGISTRY.get(args.model)
    v = est.resolve_overhead_fixed_gb(gpu.name) if est is not None else None
    if v is not None:
        return f"{float(v):.1f} GiB fixed [{args.model}@{gpu.name} calibrated]"
    return "0.0 GiB fixed [uncalibrated]"


# ============================================================================
#  Color helpers (copied from sweep_e2e_matrix.py for self-containedness).
# ============================================================================

_COLOR_RED    = "red"
_COLOR_YELLOW = "orange"
_COLOR_GREEN  = "green"


def _html_color(text: str, color: str) -> str:
    return f'<span style="color:{color}">{text}</span>'


def color_higher_is_better(value: float, lo: float, hi: float, fmt: str) -> str:
    text = format(value, fmt)
    if value < lo:
        c = _COLOR_RED
    elif value < hi:
        c = _COLOR_YELLOW
    else:
        c = _COLOR_GREEN
    return _html_color(text, c)


def color_lower_is_better(value: float, lo: float, hi: float, fmt: str) -> str:
    text = format(value, fmt)
    if value <= lo:
        c = _COLOR_GREEN
    elif value <= hi:
        c = _COLOR_YELLOW
    else:
        c = _COLOR_RED
    return _html_color(text, c)


def color_speedup(value: float) -> str:
    text = f"{value:.2f}x"
    if value < 1.0:
        c = _COLOR_RED
    elif value < 1.5:
        c = _COLOR_YELLOW
    else:
        c = _COLOR_GREEN
    return _html_color(text, c)


# ============================================================================
#  Style block.
# ============================================================================

_STYLE_BLOCK = """\
<style>
.cap-tbl { border-collapse: collapse; min-width: 100%; font-size: 13px; }
.cap-tbl th, .cap-tbl td {
  white-space: nowrap;
  padding: 4px 8px;
  border: 1px solid #555;
  vertical-align: top;
  text-align: left;
}
.cap-tbl thead th { background: #2b2b2b; color: #eee; }
.cap-tbl thead tr.lvl1 th { text-align: center; }
.cap-tbl thead tr.lvl2 th { font-weight: 600; }
.cap-tbl tbody td.row-label { font-weight: 600; background: #232323; color: #eee; }
.cap-tbl td.b-base { background: #181f25; }
.cap-tbl td.b-opt  { background: #1a2218; }
.cap-tbl td.infeasible { background: #2a1820; color: #aaa; font-style: italic; }
.cap-scroll { overflow-x: auto; max-width: 100%; }
</style>
"""

# Scenario 2 has shorter cell text than Scenario 1 (no hot-cache /
# bs-bound annotations etc.), so the global `min-width: 100%` ends up
# stretching the table over-wide and giving each cell more horizontal
# space than its content needs. We override `min-width` to `auto` for
# Scenario 2 so the table sizes itself to its content. Scenario 1
# reports remain unaffected because they embed `_STYLE_BLOCK` as-is.
_STYLE_BLOCK_S2_OVERRIDE = """\
<style>
.cap-tbl { min-width: auto !important; width: auto !important; }
</style>
"""


# ============================================================================
#  Argument parsing
# ============================================================================


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_batch_sizes(s: str) -> Optional[List[int]]:
    """`auto` (case-insensitive) or empty -> None (Mode A).
    Otherwise comma-separated ints (Mode B).
    """
    if s is None:
        return None
    if not s.strip() or s.strip().lower() == "auto":
        return None
    return parse_int_list(s)


def session_label(s: int) -> str:
    """Human-readable label for a session length: 128K, 1M, 4M, ..."""
    if s >= 1024 * 1024:
        v = s / (1024 * 1024)
        return f"{v:g}M".replace(".0M", "M")
    if s >= 1024:
        v = s / 1024
        return f"{v:g}K".replace(".0K", "K")
    return str(s)


# Default sweep: 1K -> 4M, doubling each step.
_DEFAULT_SESSIONS = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288,
    1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024,
]


# ============================================================================
#  Cell renderers
# ============================================================================


def fmt_baseline_cell(
    rep: CapacityReport,
    *,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
) -> str:
    """Render the BASELINE block (one <td>) for a row (Mode A only)."""
    if not rep.feasible:
        return '<td class="b-base infeasible">infeasible<br/>(KV doesn\'t fit)</td>'

    GiB = 2 ** 30
    weight_pct = rep.weight_fraction * 100
    kv_pct = rep.kv_fraction * 100
    mfu_pct = rep.perf_report.total.mfu * 100 if rep.perf_report else 0.0
    tpot_ms = rep.tpot_seconds * 1000 if rep.tpot_seconds else 0.0
    tput = rep.cluster_tput_tps or 0.0

    return (
        '<td class="b-base">'
        f"bs={rep.max_batch_per_gpu}<br/>"
        f"weight {rep.weight_per_gpu_bytes/GiB:.1f} GiB ({weight_pct:.0f}%)<br/>"
        f"KV/sess {rep.session_kv_bytes/GiB:.2f} GiB ({kv_pct:.0f}% HBM)<br/>"
        f"MFU {color_higher_is_better(mfu_pct, mfu_lo, mfu_hi, '.1f')}%<br/>"
        f"TPOT {color_lower_is_better(tpot_ms, tpot_lo, tpot_hi, '.2f')} ms<br/>"
        f"{tput:,.0f} tok/s"
        "</td>"
    )


def fmt_optimized_cell(
    res: OptimizedCapacityReport,
    *,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
    show_delta_bs: bool = True,
) -> str:
    """Render the DRAM POOLING block (one <td>).

    Shared by Mode A (where show_delta_bs=True, showing the bs delta vs
    baseline) and Mode B (where show_delta_bs=False — the bs is already
    encoded in the column header).
    """
    if not res.feasible:
        return '<td class="b-opt infeasible">infeasible<br/>(DRAM pool too small)</td>'

    mfu_pct = res.perf_report.total.mfu * 100 if res.perf_report else 0.0
    tpot_ms = (res.tpot_seconds or 0.0) * 1000
    tput = res.cluster_tput_tps or 0.0
    speedup = res.speedup_vs_baseline
    speedup_text = color_speedup(speedup) if speedup else "n/a"

    # bs line
    if show_delta_bs:
        bs_delta = res.max_batch_per_gpu - res.baseline.max_batch_per_gpu
        bs_text = (
            f"bs={res.max_batch_per_gpu} (+{bs_delta})"
            if bs_delta >= 0 else
            f"bs={res.max_batch_per_gpu}"
        )
    else:
        bs_text = f"bs={res.max_batch_per_gpu}"
    # Annotate which constraint is binding the chosen bs.
    if res.bs_bound_by and res.bs_bound_by != "user-override":
        bs_text += f' <span style="color:#aaa">[{res.bs_bound_by}]</span>'

    parts = [bs_text]
    parts.append(f"penalty {res.penalty_seconds*1000:.2f} ms")
    parts.append(f"MFU {color_higher_is_better(mfu_pct, mfu_lo, mfu_hi, '.1f')}%")
    parts.append(f"TPOT {color_lower_is_better(tpot_ms, tpot_lo, tpot_hi, '.2f')} ms")
    parts.append(f"{tput:,.0f} tok/s ({speedup_text})")

    # Hot-cache fittability: how many layers' worth of hot KV fit in HBM.
    # Color: < 1.0 = red (hit_rate forced to 0); >= 1.0 = neutral.
    hot_fit = res.hot_cache_layers_fittable
    if hot_fit < 1.0:
        parts.append(
            f'<i style="color:red">hot-cache: {hot_fit:.2f} layers '
            "(hit forced to 0)</i>"
        )
    else:
        parts.append(
            f'<i style="color:#aaa">hot-cache: {hot_fit:.1f} layers fittable</i>'
        )

    # Indexer prefetch overlap (only set when --indexer-in-dram).
    if res.indexer_prefetch_overlap_effective is not None:
        parts.append(
            f'<i style="color:#aaa">prefetch overlap: '
            f'{res.indexer_prefetch_overlap_effective*100:.1f}%</i>'
        )

    # Append clamp/infeasibility notes if any.
    if res.notes:
        # Show only short notes; full notes shown for clamps in tooltip-style.
        short = "; ".join(
            n for n in res.notes
            if "clamp" in n.lower() or "infeasible" in n.lower()
        )
        if short:
            parts.append(f'<i style="color:#aaa">{short}</i>')

    return '<td class="b-opt">' + "<br/>".join(parts) + "</td>"


# ============================================================================
#  Argument parsing
# ============================================================================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Topology / model.
    ap.add_argument("--gpu", default="GB200", choices=list(GPU_PRESETS.keys()))
    ap.add_argument("--model", default="deepseek-v3.2",
                    choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ep", type=int, default=32)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)

    # HBM (defaults from MEMORY_PRESETS based on --gpu).
    ap.add_argument("--hbm-capacity-gb", type=float, default=None,
                    help="override HBM capacity (GiB); default uses MEMORY_PRESETS")
    ap.add_argument("--hbm-overhead-frac", type=float, default=0.05,
                    help="PROPORTIONAL HBM overhead as a fraction of capacity "
                         "(allocator reserve / kernel scratch). Default 0.05.")
    ap.add_argument("--hbm-overhead-fixed-gb", type=float, default=None,
                    help="FIXED HBM overhead in GiB, added on top of "
                         "--hbm-overhead-frac: CUDA-graph capture + "
                         "DeepEP/NCCL comm buffers + framework runtime "
                         "reserve. For DeepEP-based MoE decode this is large "
                         "(~15-20 GiB even on 96 GiB cards). Default: unset → "
                         "use the model's per-GPU CALIBRATED value "
                         "(ModelCostEstimator.overhead_fixed_gb_by_gpu); "
                         "falls back to 0 if the model has no calibration. "
                         "An explicit value always overrides.")
    ap.add_argument("--kv-dtype", default="mixed_bf16_fp8",
                    choices=["bf16", "fp8", "mixed_bf16_fp8"],
                    help="KV-cache dtype for capacity sizing. `bf16` = "
                         "full BF16 latent (2 B/elem); `fp8`/`mixed_bf16_fp8` "
                         "= FP8 latent (1 B/elem) + BF16 RoPE. Default "
                         "mixed_bf16_fp8 (unchanged from prior behaviour).")
    ap.add_argument("--hot-slots", type=int, default=None,
                    help="(Scenario 1 only.) HBM hot-cache window size, in "
                         "tokens/layer/session. Capacity uses a physical "
                         "split: HBM holds min(S, hot_slots) hot main-KV "
                         "tokens/layer across all layers (+ indexer if "
                         "HBM-resident); DRAM holds the cold tail. Reflects "
                         "reserving more slots than top-k (e.g. 4096 for a "
                         "2048 top-k) to cut miss rate. Default = model top-k "
                         "(dsa_topk), the physical minimum; explicit values "
                         "are floored at top-k and capped at S. Affects batch "
                         "sizing only, NOT the hit_rate penalty.")

    # Runtime context length used when calling estimate_perf (vs the session
    # length used for capacity sizing). See WorkloadConfig.context_length
    # semantics in simulator.core.
    ap.add_argument("--context-length", type=int, default=9216,
                    help="N for estimate_perf — the runtime attention length "
                         "used to compute MFU. DECOUPLED from session_length "
                         "(which sizes KV capacity). Default 9216 (typical "
                         "RL-decode runtime N) so MFU rows are comparable "
                         "across session_length. Pass an explicit value to "
                         "evaluate at long-context steady state. Pass "
                         "`--context-length -1` to fall back to session_length "
                         "per row (worst-case decode step).")
    # DEPRECATED alias from the days when this CLI flag was named after the
    # old `runtime_context_length` argument; both map to args.context_length.
    ap.add_argument("--runtime-context-length", type=int, default=None,
                    dest="_legacy_runtime_context_length",
                    help="[DEPRECATED] alias for --context-length")

    # Session sweep.
    ap.add_argument("--sessions", type=parse_int_list, default=_DEFAULT_SESSIONS,
                    help="comma-separated session_lengths to sweep "
                         "(default: 1K..4M, doubling)")

    # Scenario selection — chooses which DRAM pooling physics to model.
    ap.add_argument(
        "--scenario", default="sparse_on_demand",
        choices=["sparse_on_demand", "shared_prefix"],
        help="Which DRAM pooling scenario to render. "
             "'sparse_on_demand' (default): per-layer on-demand fetch "
             "for sparse-attention models (Scenario 1; cell shows base "
             "/ opt-idx-HBM / opt-idx-DRAM). "
             "'shared_prefix': cross-session shared prefix in DRAM with "
             "layer-stride prefetch (Scenario 2; cell shows base / "
             "opt-shared-prefix). The two scenarios use different cell "
             "layouts and different sweep dimensions — see "
             "--prefix-share-sweep for Scenario 2.",
    )

    # DRAM Pooling — capacity & interconnect.
    ap.add_argument("--dram-capacity-gb", type=float, default=12288.0,
                    help="per-GPU DRAM pool capacity (GiB). Default 12288 "
                         "(= 12 TiB / GPU), enough to support bs=256 at 1M "
                         "ctx with V3.2 on 32-GPU GB200 deployment. Reduce "
                         "if modeling a smaller pool.")
    ap.add_argument("--dram-interconnect-bandwidth-gbs", type=float, default=50.0,
                    help="per-GPU DRAM interconnect bandwidth (GB/s, decimal). "
                         "Typical: PCIe Gen5=64, RDMA=25/50/100, NVLink-C2C=450. "
                         "Default 50 (PCIe / RDMA tier). Ignored if "
                         "--bw-sweep is given.")
    ap.add_argument("--bw-sweep", type=str, default=None,
                    help="Comma-separated DRAM interconnect bandwidths (GB/s) "
                         "to sweep, e.g. '50,100,200,400,800'. When given, "
                         "produces one report per value with a `_bwNN` "
                         "suffix appended to --out. Useful for scaleup-fabric "
                         "DRAM placements where the link rate is the main "
                         "design knob. Mutually exclusive with --hit-rate-sweep.")
    ap.add_argument("--hit-rate-sweep", type=str, default=None,
                    help="Comma-separated per-layer hit_rate values in [0,1] "
                         "to sweep, e.g. '0,0.2,0.4,0.6,0.8,0.99'. When given, "
                         "produces one report per value with a `_hitNN` "
                         "suffix appended to --out (NN = pct, e.g. hit99). "
                         "Mutually exclusive with --bw-sweep / "
                         "--prefix-share-sweep. Scenario 1 only.")
    ap.add_argument("--prefix-share-sweep", type=str, default=None,
                    help="(Scenario 2 only.) Comma-separated "
                         "prefix_share_frac values in [0,1] to sweep, "
                         "e.g. '0,0.2,0.4,0.6,0.8,1.0'. When given, "
                         "produces one report per value with a `_pfNN` "
                         "suffix appended to --out (NN = pct, e.g. pf80, "
                         "pf100). Mutually exclusive with --bw-sweep / "
                         "--hit-rate-sweep. Requires "
                         "--scenario shared_prefix.")
    # DEPRECATED alias (mapped onto the new flag).
    ap.add_argument("--dram-bandwidth-gbs", type=float, default=None,
                    help="[DEPRECATED] alias for --dram-interconnect-bandwidth-gbs")

    # DRAM Pooling — sparse on-demand penalty parameters.
    ap.add_argument("--hit-rate", type=float, default=0.0,
                    help="(Scenario 1 only.) per-layer KV cache hit rate "
                         "in [0,1]. "
                         "0=naive (every layer misses, full DRAM stream); "
                         "0.99=production-usable; 1.0=ideal. Default 0.0.")
    ap.add_argument("--n-missing-sessions", type=int, default=None,
                    help="(Scenario 1 only.) How many of `bs` sessions "
                         "miss on a missed layer. "
                         "Default (unset) = `bs` (pessimistic batch-wide "
                         "miss upper bound; the new default since the cold-"
                         "fetch model was switched to top-k). "
                         "Pass 1 explicitly for the optimistic lower bound "
                         "(de-correlated per-session miss timing). Real "
                         "value depends on session-level correlation in "
                         "HBM eviction.")

    # DRAM Pooling — shared-prefix scenario knobs.
    ap.add_argument("--prefix-share-frac", type=float, default=0.5,
                    help="(Scenario 2 only.) Fraction of per-session KV "
                         "that is cross-session-shared prefix, in [0,1]. "
                         "DRAM holds a single shared copy; HBM holds the "
                         "(1-α) per-session unique part. Default 0.5. "
                         "Ignored if --prefix-share-sweep is given.")

    # Mode B (trade-off matrix). Default: enabled with 4..512 (8 cols geom).
    # `auto` switches to Mode A (legacy 2-column baseline-vs-optimized layout).
    ap.add_argument("--batch-sizes", type=str,
                    default="4,8,16,32,64,128,256,512",
                    help="Mode B (default): comma-separated request batch sizes "
                         "to sweep, e.g. '4,8,16,32,64,128,256,512'. "
                         "When the requested bs exceeds HBM capacity, the "
                         "BASELINE block is degraded to hbm_max_batch and "
                         "the actual bs is shown. "
                         "Pass 'auto' to switch to Mode A (legacy 2-col layout, "
                         "optimization picks its own auto-max bs).")

    # Color thresholds.
    ap.add_argument("--mfu-lo", type=float, default=20.0)
    ap.add_argument("--mfu-hi", type=float, default=60.0)
    ap.add_argument("--tpot-lo", type=float, default=10.0)
    ap.add_argument("--tpot-hi", type=float, default=30.0)

    ap.add_argument("--out", default="dram_analysis.md")
    return ap


# ============================================================================
#  Header rendering
# ============================================================================


def render_header(args, gpu, mem, parallel, dram_cfg, mode: str) -> List[str]:
    lines: List[str] = []
    lines.append(_STYLE_BLOCK)
    if mode == "A":
        lines.append("# DRAM extension analysis: session_length × {baseline, DRAM Pooling}")
    else:
        lines.append("# DRAM extension trade-off: session_length × batch_size")
    lines.append("")
    lines.append(f"- **Model**: {args.model}")
    lines.append(f"- **GPU**: {gpu.name} "
                 f"(peak {gpu.peak_tflops:.0f} TF/s, BW {gpu.bandwidth_gbs:.0f} GB/s, "
                 f"HBM {mem.hbm_capacity_gb:.0f} GiB, "
                 f"overhead {mem.weight_overhead_frac*100:.1f}% "
                 f"+ {_fixed_overhead_str(args, gpu)})")
    lines.append(f"- **Parallel**: TP={parallel.tp}, EP={parallel.ep}, "
                 f"PP={parallel.pp}, DP={parallel.dp} "
                 f"(world_size={parallel.world_size} GPUs)")
    lines.append(
        "- **Parallel mode**: switching parallel — "
        "DP×TP=EP enforced; same GPUs run TP×DP for attn, "
        "reshape to EP for FFN (V3/V3.2 decode style)."
    )
    n_miss_disp = (
        "bs (default upper bound)"
        if dram_cfg.n_missing_sessions is None
        else str(dram_cfg.n_missing_sessions)
    )
    hot_slots_disp = (
        f"{dram_cfg.hot_slots} tok/layer (HBM hot window; DRAM holds cold tail)"
        if dram_cfg.hot_slots is not None
        else "model top-k (dsa_topk) — physical min; DRAM holds cold tail"
    )
    lines.append(
        f"- **DRAM Pooling**: capacity={dram_cfg.dram_capacity_gb:.0f} GiB/GPU, "
        f"interconnect BW={dram_cfg.dram_interconnect_bandwidth_gbs:.1f} GB/s, "
        f"hit_rate={dram_cfg.kv_cache_hit_rate:.3f}, "
        f"n_missing_sessions={n_miss_disp}, "
        f"hot_slots={hot_slots_disp}, "
        "indexer residency = {HBM, DRAM} (both shown per cell; "
        "DRAM-resident uses auto roofline prefetch overlap)"
    )
    if args.context_length is None:
        # `-1` sentinel was translated to None in main() before calling
        # render_header. Means: fall back to session_length per row.
        lines.append(
            "- **Runtime context length** (for MFU): "
            "= session_length per row (worst-case decode step; long-N attn "
            "dominates MFU and varies by row)."
        )
    else:
        lines.append(
            f"- **Runtime context length** (for MFU): "
            f"{args.context_length:,} (FIXED across all session_length "
            f"rows, decoupled from session_length).<br/>"
            f"  *MFU is therefore comparable row-to-row; differences come "
            f"only from the available batch size.*"
        )
    lines.append(
        "- **Glossary**: `session_length` = KV-capacity upper bound for one "
        "session (drives `max_batch_per_gpu`). `context_length` = runtime "
        "attention length N for one decode step (drives MFU / TPOT). The "
        "two are independent: a session sized for 128K may be at runtime "
        "N=9,216 (early in the conversation) or N=128K (late, near capacity)."
    )
    lines.append(
        "- **TODO** (modeling): a real workload has a *distribution* of "
        "runtime N over the session lifetime (early steps short, late "
        "steps long); current tool uses a single fixed value. Future "
        "revision should sweep / weight runtime N by that distribution."
    )
    lines.append("")

    # Penalty model recap.
    lines.append("## Penalty model (sparse on-demand DRAM pooling)")
    lines.append("")
    lines.append("```")
    lines.append("miss_layers       = num_layers * (1 - hit_rate)")
    lines.append("cold_layer        = cold_layer_kv_per_session   # model-aware:")
    lines.append("                    #   V3 base: full per-layer KV  (S × v_token)")
    lines.append("                    #   V3.2 DSA: only top_k tokens (top_k × v_token)")
    lines.append("                    #   V4 hybrid: weighted CSA/HCA/SWA average")
    lines.append("cold_total        = cold_layer * n_missing_sessions")
    lines.append("penalty_main      = miss_layers * cold_total / dram_bw")
    lines.append("")
    lines.append("# Indexer-in-DRAM penalty (idx-DRAM cell only):")
    lines.append("# Layer i+1's indexer K can be prefetched during layer i's")
    lines.append("# (attn + ffn) compute window (K_indexer[i+1] is independent")
    lines.append("# of layer i's outputs). Effective overlap is derived from")
    lines.append("# a roofline — no user knob:")
    lines.append("t_layer           = (mfu.attn.time + mfu.ffn.time) / num_layers")
    lines.append("t_prefetch_layer  = bs * indexer_kv_per_session / num_layers / dram_bw")
    lines.append("overlap           = min(1, t_layer / t_prefetch_layer)")
    lines.append("indexer_per_step  = bs * indexer_kv_per_session")
    lines.append("penalty_indexer   = indexer_per_step / dram_bw * (1 - overlap)")
    lines.append("")
    lines.append("penalty           = penalty_main + penalty_indexer  # per token")
    lines.append("```")
    lines.append("")

    # Color legend.
    lines.append("## Color legend")
    lines.append("")
    lines.append(
        f"- **MFU%**: "
        f"{_html_color('< ' + format(args.mfu_lo, '.0f'), _COLOR_RED)} red, "
        f"{_html_color(format(args.mfu_lo, '.0f') + '-' + format(args.mfu_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= ' + format(args.mfu_hi, '.0f'), _COLOR_GREEN)} green "
        "(higher is better)."
    )
    lines.append(
        f"- **TPOT (ms)**: "
        f"{_html_color('<= ' + format(args.tpot_lo, '.0f'), _COLOR_GREEN)} green, "
        f"{_html_color(format(args.tpot_lo, '.0f') + '-' + format(args.tpot_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('> ' + format(args.tpot_hi, '.0f'), _COLOR_RED)} red "
        "(lower is better)."
    )
    lines.append(
        f"- **speedup**: "
        f"{_html_color('< 1.0x', _COLOR_RED)} red, "
        f"{_html_color('1.0-1.5x', _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= 1.5x', _COLOR_GREEN)} green."
    )
    lines.append("")
    return lines


# ============================================================================
#  Mode A: session sweep × {baseline, optimized}
# ============================================================================


def render_mode_a(args, gpu, mem, parallel, optimizer, dram_cfg) -> List[str]:
    lines: List[str] = []
    lines.append("## Cell format")
    lines.append("")
    lines.append("Each row is one `session_length`. Two side-by-side blocks:")
    lines.append("")
    lines.append("- **BASELINE**: `bs` / weight (GiB,%) / session-KV (GiB,%HBM) / "
                 "MFU% / TPOT (ms) / cluster tput (tok/s)")
    lines.append("- **DRAM POOLING**: `bs (delta vs baseline)` / penalty (ms) / "
                 "MFU% / TPOT (ms) / cluster tput (speedup vs baseline)")
    lines.append("")

    lines.append('<div class="cap-scroll">')
    lines.append('<table class="cap-tbl">')
    lines.append('<thead>')
    lines.append('<tr class="lvl1">'
                 '<th rowspan="2">session_length</th>'
                 '<th>BASELINE (HBM-only)</th>'
                 f'<th>DRAM POOLING (+{dram_cfg.dram_capacity_gb:.0f} GiB @ '
                 f'{dram_cfg.dram_interconnect_bandwidth_gbs:.0f} GB/s, '
                 f'hit={dram_cfg.kv_cache_hit_rate:.2f})</th>'
                 '</tr>')
    lines.append('<tr class="lvl2"><th>baseline</th><th>optimized</th></tr>')
    lines.append('</thead>')

    lines.append('<tbody>')
    n_feasible_base = 0
    n_feasible_opt = 0
    for s in args.sessions:
        baseline = analyze_capacity(
            gpu=gpu, mem=mem,
            model_name=args.model,
            parallel=parallel,
            session_length=s,
            context_length=args.context_length,
            dtype_kv=Dtype(args.kv_dtype),
        )
        optimized = optimizer.apply(
            baseline,
            gpu=gpu, mem=mem, parallel=parallel,
        )
        if baseline.feasible:
            n_feasible_base += 1
        if optimized.feasible:
            n_feasible_opt += 1

        row_label = f"<b>{session_label(s)}</b><br/>({s:,})"
        row_html = [f'<td class="row-label">{row_label}</td>']
        row_html.append(fmt_baseline_cell(
            baseline,
            mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
            tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
        ))
        row_html.append(fmt_optimized_cell(
            optimized,
            mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
            tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
            show_delta_bs=True,
        ))
        lines.append('<tr>' + ''.join(row_html) + '</tr>')
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    lines.append("")
    lines.append(f"_Summary: {n_feasible_base}/{len(args.sessions)} session lengths "
                 f"feasible under HBM-only baseline; "
                 f"{n_feasible_opt}/{len(args.sessions)} feasible with DRAM Pooling._")
    return lines


# ============================================================================
#  Mode B: session × batch_size × {baseline, optimized}
# ============================================================================


def _baseline_at_bs(
    *,
    gpu, mem, parallel, args,
    baseline_capacity: CapacityReport,
    req_bs: int,
):
    """Build a (actual_bs, perf_report, tput_tps) tuple for the BASELINE block
    at a requested batch size, degrading to hbm_max_batch when needed.

    Returns:
        (actual_bs, perf_report, cluster_tput_tps, was_degraded)
        or (0, None, 0.0, ...) if the session is not feasible at all
        under HBM (i.e. hbm_max_batch == 0).
    """
    hbm_max = baseline_capacity.max_batch_per_gpu
    if not baseline_capacity.feasible or hbm_max < 1:
        return 0, None, 0.0, False

    actual_bs = min(req_bs, hbm_max)
    was_degraded = actual_bs < req_bs

    if actual_bs == hbm_max and req_bs >= hbm_max:
        # The default `analyze_capacity` already ran estimate_perf at hbm_max.
        # Reuse its perf_report to avoid a redundant call.
        rep = baseline_capacity.perf_report
    else:
        # Re-run estimate_perf at the degraded bs.
        wl = WorkloadConfig(
            phase=Phase.DECODE,
            batch_size=actual_bs,
            context_length=baseline_capacity.workload.context_length,
            new_tokens=baseline_capacity.workload.new_tokens,
            dtype_param=baseline_capacity.workload.dtype_param,
            dtype_kv=baseline_capacity.workload.dtype_kv,
        )
        rep = estimate_perf(
            gpu=gpu,
            model_name=args.model,
            parallel=parallel,
            workload=wl,
        )

    if rep is None or rep.total.time_seconds <= 0:
        return actual_bs, rep, 0.0, was_degraded

    bs_in_flight = actual_bs * max(1, parallel.dp)
    tput = bs_in_flight / rep.total.time_seconds
    return actual_bs, rep, float(tput), was_degraded


def _render_opt_block(
    *,
    opt: OptimizedCapacityReport,
    label: str,
    req_bs: int,
    base_tput: float,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
) -> str:
    """Render one OPTIMIZED sub-block (used twice per cell: indexer-in-HBM
    and indexer-in-DRAM). Returns a single <div>...</div> string.
    """
    if not opt.feasible:
        return (
            '<div style="color:#aaa;font-style:italic">'
            f'<b>{label}</b>: infeasible (DRAM cap)</div>'
        )

    opt_mfu = opt.perf_report.total.mfu * 100 if opt.perf_report else 0.0
    opt_tpot_ms = (opt.tpot_seconds or 0.0) * 1000
    opt_tput = opt.cluster_tput_tps or 0.0

    # speedup = optimized tput / baseline tput at the SAME requested bs
    if base_tput > 0 and opt_tput > 0:
        speedup_val = opt_tput / base_tput
        speedup_text = color_speedup(speedup_val)
    else:
        speedup_text = "n/a"

    bs_text = f"bs={opt.max_batch_per_gpu}"
    if opt.max_batch_per_gpu < req_bs:
        bound_label = opt.bs_bound_by or "DRAM-cap"
        bs_text += (
            f' <span style="color:#aaa">(req={req_bs}, '
            f'{bound_label})</span>'
        )

    # Hot-cache fittability indicator.
    hot_fit = opt.hot_cache_layers_fittable
    if hot_fit < 1.0:
        hot_text = (
            f'<i style="color:red">hot-cache: {hot_fit:.2f} '
            "layers (hit→0)</i>"
        )
    else:
        hot_text = (
            f'<i style="color:#aaa">hot-cache: {hot_fit:.1f} layers</i>'
        )
    # Indexer prefetch overlap (only set when indexer_in_dram=True).
    if opt.indexer_prefetch_overlap_effective is not None:
        hot_text += (
            f'<br/><i style="color:#aaa">prefetch overlap: '
            f'{opt.indexer_prefetch_overlap_effective*100:.1f}%</i>'
        )

    # DRAM pool occupancy (actually used) vs pool capacity.
    dram_used_gib = opt.dram_used_bytes / (1024 ** 3)
    dram_cap_gib = opt.extra_capacity_bytes / (1024 ** 3)
    dram_pct = (100.0 * dram_used_gib / dram_cap_gib) if dram_cap_gib > 0 else 0.0
    dram_text = (
        f'<i style="color:#aaa">DRAM used: {dram_used_gib:,.2f} / '
        f'{dram_cap_gib:,.0f} GiB ({dram_pct:.1f}%)</i>'
    )

    return (
        '<div>'
        f'<b>{label}</b> ' + bs_text + '<br/>'
        f"penalty {opt.penalty_seconds*1000:.2f} ms<br/>"
        f"MFU {color_higher_is_better(opt_mfu, mfu_lo, mfu_hi, '.1f')}%, "
        f"TPOT {color_lower_is_better(opt_tpot_ms, tpot_lo, tpot_hi, '.2f')} ms<br/>"
        f"{opt_tput:,.0f} tok/s ({speedup_text})<br/>"
        f"{dram_text}<br/>"
        f"{hot_text}"
        '</div>'
    )


def fmt_combined_cell(
    *,
    req_bs: int,
    actual_base_bs: int,
    base_rep,
    base_tput: float,
    was_degraded: bool,
    opt_hbm: OptimizedCapacityReport,
    opt_dram: OptimizedCapacityReport,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
) -> str:
    """Render one Mode B cell: 3 stacked sub-blocks separated by <hr/>:

      1. BASELINE             (HBM-only)
      2. OPT, indexer in HBM  (DRAM pooling for main KV; indexer stays HBM)
      3. OPT, indexer in DRAM (DRAM pooling + indexer also moved, with
                               roofline-derived prefetch overlap)

    All three blocks correspond to the SAME requested batch size `req_bs`.
    Any block may be infeasible (rendered as italic gray).
    """
    # Use a thin horizontal rule as the visual separator between sub-blocks.
    sep = (
        '<hr style="margin:6px 0;border:0;border-top:1px dashed #ccc"/>'
    )

    parts: List[str] = []

    # ---------- (1) BASELINE block ----------
    if base_rep is None or actual_base_bs < 1:
        parts.append(
            '<div style="color:#aaa;font-style:italic">'
            f'<b>base</b>: infeasible at bs={req_bs}</div>'
        )
    else:
        bs_text = (
            f"bs={actual_base_bs}"
            + (f' <span style="color:#aaa">(req={req_bs}, HBM-cap)</span>'
               if was_degraded else "")
        )
        mfu_pct = base_rep.total.mfu * 100 if base_rep else 0.0
        tpot_ms = base_rep.total.time_seconds * 1000 if base_rep else 0.0
        parts.append(
            '<div>'
            '<b>base</b> ' + bs_text + '<br/>'
            f"MFU {color_higher_is_better(mfu_pct, mfu_lo, mfu_hi, '.1f')}%, "
            f"TPOT {color_lower_is_better(tpot_ms, tpot_lo, tpot_hi, '.2f')} ms<br/>"
            f"{base_tput:,.0f} tok/s"
            '</div>'
        )

    # ---------- (2) OPT, indexer in HBM ----------
    parts.append(sep)
    parts.append(_render_opt_block(
        opt=opt_hbm, label="opt (idx-HBM)",
        req_bs=req_bs, base_tput=base_tput,
        mfu_lo=mfu_lo, mfu_hi=mfu_hi,
        tpot_lo=tpot_lo, tpot_hi=tpot_hi,
    ))

    # ---------- (3) OPT, indexer in DRAM (with prefetch) ----------
    parts.append(sep)
    parts.append(_render_opt_block(
        opt=opt_dram, label="opt (idx-DRAM)",
        req_bs=req_bs, base_tput=base_tput,
        mfu_lo=mfu_lo, mfu_hi=mfu_hi,
        tpot_lo=tpot_lo, tpot_hi=tpot_hi,
    ))

    return '<td class="b-opt">' + "".join(parts) + "</td>"


def render_mode_b(
    args, gpu, mem, parallel, optimizer_hbm, optimizer_dram, dram_cfg,
    batch_sizes: List[int],
) -> List[str]:
    lines: List[str] = []
    lines.append("## Cell format (Mode B: session × batch_size × {base, opt})")
    lines.append("")
    lines.append(
        "Each cell holds **three** stacked sub-blocks (separated by a "
        "dashed line) for the SAME requested batch size:"
    )
    lines.append("")
    lines.append(
        "- **base**: HBM-only baseline. If the requested bs exceeds HBM "
        "capacity, it is **degraded** to `hbm_max_batch` and the actual bs "
        'is shown as `bs=4 (req=64, HBM-cap)`.'
    )
    lines.append(
        "- **opt (idx-HBM)**: DRAM pooling for **main KV only** — the "
        "indexer K-cache stays HBM-resident. `bs` may be clamped by the "
        "indexer footprint (`indexer-cap`) or DRAM capacity (`DRAM-cap`)."
    )
    lines.append(
        "- **opt (idx-DRAM)**: DRAM pooling **plus** indexer moved to "
        "DRAM. Indexer K is prefetched async at the start of layer i for "
        "layer i+1 (see roofline overlap formula in the header). `prefetch "
        "overlap: X%` shows the fraction of fetch time hidden under the "
        "(attn + ffn) compute window."
    )
    lines.append("")

    lines.append(
        "- **speedup** (per opt block) = optimized cluster tput / baseline "
        "cluster tput at the SAME requested bs."
    )
    lines.append("")

    lines.append('<div class="cap-scroll">')
    lines.append('<table class="cap-tbl">')
    lines.append('<thead>')
    lines.append('<tr class="lvl1">'
                 f'<th rowspan="2">session_length<br/><i>(HBM max bs)</i></th>'
                 f'<th colspan="{len(batch_sizes)}">'
                 f'requested batch_size — base / opt (idx-HBM) / opt (idx-DRAM) '
                 f'(+{dram_cfg.dram_capacity_gb:.0f} GiB @ '
                 f'{dram_cfg.dram_interconnect_bandwidth_gbs:.0f} GB/s, '
                 f'hit={dram_cfg.kv_cache_hit_rate:.2f})'
                 '</th>'
                 '</tr>')
    lines.append('<tr class="lvl2">'
                 + ''.join(f'<th>bs={bs}</th>' for bs in batch_sizes)
                 + '</tr>')
    lines.append('</thead>')

    lines.append('<tbody>')
    for s in args.sessions:
        baseline_cap = analyze_capacity(
            gpu=gpu, mem=mem,
            model_name=args.model,
            parallel=parallel,
            session_length=s,
            context_length=args.context_length,
            dtype_kv=Dtype(args.kv_dtype),
        )
        hbm_max = (
            baseline_cap.max_batch_per_gpu if baseline_cap.feasible else 0
        )
        row_label = (
            f"<b>{session_label(s)}</b><br/>({s:,})"
            f"<br/><i>HBM max bs={hbm_max}</i>"
        )
        row_html = [f'<td class="row-label">{row_label}</td>']

        for bs in batch_sizes:
            actual_base_bs, base_rep, base_tput, was_degraded = (
                _baseline_at_bs(
                    gpu=gpu, mem=mem, parallel=parallel, args=args,
                    baseline_capacity=baseline_cap, req_bs=bs,
                )
            )
            opt_hbm = optimizer_hbm.apply(
                baseline_cap,
                gpu=gpu, mem=mem, parallel=parallel,
                batch_size_override=bs,
            )
            opt_dram = optimizer_dram.apply(
                baseline_cap,
                gpu=gpu, mem=mem, parallel=parallel,
                batch_size_override=bs,
            )
            row_html.append(fmt_combined_cell(
                req_bs=bs,
                actual_base_bs=actual_base_bs,
                base_rep=base_rep,
                base_tput=base_tput,
                was_degraded=was_degraded,
                opt_hbm=opt_hbm,
                opt_dram=opt_dram,
                mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
                tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
            ))
        lines.append('<tr>' + ''.join(row_html) + '</tr>')
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return lines


# ============================================================================
#  Scenario 2: Shared-Prefix Prefetch — header + cell + Mode A/B renderers.
#
#  Layout differences from Scenario 1:
#    * Cell has 2 sub-blocks (base / opt-shared-prefix), not 3.
#    * No idx-HBM vs idx-DRAM split (indexer concept does not apply).
#    * No hit_rate / n_missing_sessions in header; instead α
#      (prefix_share_frac) is the headline knob.
#    * Penalty model recap differs (single-shared-copy roofline).
# ============================================================================


def render_header_s2(
    args, gpu, mem, parallel, dram_cfg, mode: str,
) -> List[str]:
    """Header for Scenario 2 reports. Differs from Scenario 1's
    `render_header` only in the DRAM-pooling line and the penalty-model
    recap; everything else (style block, parallel info, color legend) is
    shared.
    """
    lines: List[str] = []
    lines.append(_STYLE_BLOCK)
    # Scenario 2 cells have shorter text → narrow the table to fit
    # content (override the global `min-width: 100%`).
    lines.append(_STYLE_BLOCK_S2_OVERRIDE)
    if mode == "A":
        lines.append("# Shared-prefix DRAM Pooling: session_length × {baseline, opt}")
    else:
        lines.append("# Shared-prefix DRAM Pooling: session_length × batch_size")
    lines.append("")
    lines.append(f"- **Model**: {args.model}")
    lines.append(f"- **GPU**: {gpu.name} "
                 f"(peak {gpu.peak_tflops:.0f} TF/s, BW {gpu.bandwidth_gbs:.0f} GB/s, "
                 f"HBM {mem.hbm_capacity_gb:.0f} GiB, "
                 f"overhead {mem.weight_overhead_frac*100:.1f}% "
                 f"+ {_fixed_overhead_str(args, gpu)})")
    lines.append(f"- **Parallel**: TP={parallel.tp}, EP={parallel.ep}, "
                 f"PP={parallel.pp}, DP={parallel.dp} "
                 f"(world_size={parallel.world_size} GPUs)")
    lines.append(
        "- **Parallel mode**: switching parallel — "
        "DP×TP=EP enforced; same GPUs run TP×DP for attn, "
        "reshape to EP for FFN (V3/V3.2 / V4-Pro decode style)."
    )
    lines.append(
        f"- **Shared-prefix DRAM Pooling**: capacity="
        f"{dram_cfg.dram_capacity_gb:.0f} GiB/GPU (single shared copy "
        f"only; α × session_kv must fit), interconnect BW="
        f"{dram_cfg.dram_interconnect_bandwidth_gbs:.1f} GB/s, "
        f"**α (prefix_share_frac)={dram_cfg.prefix_share_frac:.3f}** "
        f"(fraction of session_kv that is cross-session-shared prefix)."
    )
    if args.context_length is None:
        lines.append(
            "- **Runtime context length** (for MFU): "
            "= session_length per row (worst-case decode step; long-N attn "
            "dominates MFU and varies by row)."
        )
    else:
        lines.append(
            f"- **Runtime context length** (for MFU): "
            f"{args.context_length:,} (FIXED across all session_length "
            f"rows, decoupled from session_length).<br/>"
            f"  *MFU is therefore comparable row-to-row; differences come "
            f"only from the available batch size.*"
        )
    lines.append(
        "- **Glossary**: `session_length` = KV-capacity upper bound for one "
        "session. `α` = `prefix_share_frac` = fraction of session_kv that "
        "is shared across all batched sessions (system prompt, tool "
        "catalog, RAG context, ...) and lives as a SINGLE copy in DRAM. "
        "The remaining (1-α) per-session unique part stays HBM-resident "
        "and bounds bs."
    )
    lines.append("")

    # Penalty model recap — Scenario 2.
    lines.append("## Penalty model (shared-prefix prefetch + layer-spill)")
    lines.append("")
    lines.append("```")
    lines.append("# Capacity (with automatic tail-layer spill to DRAM)")
    lines.append("# Production fact: prefill streams KV layer-by-layer; KV pool is")
    lines.append("# layer × token, so the natural spill unit is the LAST k layers.")
    lines.append("DRAM-prefix-cap (binary):  α × session_kv          ≤ DRAM_capacity")
    lines.append("HBM holds first L-k layers' unique:  (L-k) × bs × PLU ≤ HBM_avail")
    lines.append("DRAM holds prefix + last k layers' unique:")
    lines.append("    α × session_kv + k × bs × PLU                    ≤ DRAM_capacity")
    lines.append("    where PLU = (1-α) × session_kv / num_layers      # per-layer unique")
    lines.append("⇒ bs_max = (HBM_avail + (DRAM - α × session_kv)) / ((1-α) × session_kv)")
    lines.append("")
    lines.append("# Roofline prefetch overlap — split into LIGHT / HEAVY paths:")
    lines.append("t_layer       = (mfu.attn.time + mfu.ffn.time) / num_layers")
    lines.append("# Light layers (L-k): unique in HBM → only prefix needs DMA")
    lines.append("t_pref_light  = (α × session_kv / L) / bw")
    lines.append("overlap_light = min(1, t_layer / t_pref_light)")
    lines.append("# Heavy layers (k): both prefix AND that layer's bs × PLU from DRAM")
    lines.append("t_pref_heavy  = (α × session_kv / L + bs × PLU) / bw")
    lines.append("overlap_heavy = min(1, t_layer / t_pref_heavy)")
    lines.append("penalty = (L-k) × t_pref_light × (1 - overlap_light)")
    lines.append("        +     k × t_pref_heavy × (1 - overlap_heavy)")
    lines.append("```")
    lines.append("")
    lines.append(
        "Key physical difference vs Scenario 1's indexer-in-DRAM term: "
        "the prefix is a SINGLE shared copy serving the entire batch (light "
        "path, NOT × bs); only the spilled tail-k layers' unique KV is per-"
        "session and × bs (heavy path). Cells expose `k/L` and the heavy-"
        "path overlap when spill is active."
    )
    lines.append("")

    # Color legend (shared with Scenario 1).
    lines.append("## Color legend")
    lines.append("")
    lines.append(
        f"- **MFU%**: "
        f"{_html_color('< ' + format(args.mfu_lo, '.0f'), _COLOR_RED)} red, "
        f"{_html_color(format(args.mfu_lo, '.0f') + '-' + format(args.mfu_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= ' + format(args.mfu_hi, '.0f'), _COLOR_GREEN)} green "
        "(higher is better)."
    )
    lines.append(
        f"- **TPOT (ms)**: "
        f"{_html_color('<= ' + format(args.tpot_lo, '.0f'), _COLOR_GREEN)} green, "
        f"{_html_color(format(args.tpot_lo, '.0f') + '-' + format(args.tpot_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('> ' + format(args.tpot_hi, '.0f'), _COLOR_RED)} red "
        "(lower is better)."
    )
    lines.append(
        f"- **speedup**: "
        f"{_html_color('< 1.0x', _COLOR_RED)} red, "
        f"{_html_color('1.0-1.5x', _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= 1.5x', _COLOR_GREEN)} green."
    )
    lines.append("")
    return lines


def _num_layers_from_report(opt: OptimizedCapacityReport) -> int:
    """Look up `num_layers` for the model in `opt.baseline`.

    Used by Scenario 2 cell rendering to display `k/L` (spilled tail
    layers vs total layers). Returns 0 if the model is missing or has
    no num_layers entry — the caller should treat 0 as "unknown" and
    display `k/?`.
    """
    try:
        from simulator.models import get_model
        model = get_model(opt.baseline.model_name)
        return int(model.merged_config(None).get("num_layers", 0))
    except Exception:
        return 0


def _render_opt_block_s2(
    *,
    opt: OptimizedCapacityReport,
    label: str,
    req_bs: int,
    base_tput: float,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
) -> str:
    """Render the Scenario 2 OPTIMIZED sub-block.

    Differs from Scenario 1's `_render_opt_block`:
      * No `hot-cache layers fittable` line (concept does not apply).
      * `prefetch overlap` (light path) line shown when set.
      * `bs_bound_by` may be "HBM-cap" / "DRAM-prefix-cap" /
        "DRAM-spill-cap" / "model-unbounded" / "user-override".
      * "model-unbounded" gets a clarifying gray note.
      * When unique KV is spilled to DRAM (k>0) we add a dedicated
        line `spill: k/L layers (heavy overlap X%)` because the heavy
        path is the dominant penalty source.
    """
    if not opt.feasible:
        notes_short = "; ".join(opt.notes) if opt.notes else "DRAM cap"
        return (
            '<div style="color:#aaa;font-style:italic">'
            f'<b>{label}</b>: infeasible ({notes_short})</div>'
        )

    opt_mfu = opt.perf_report.total.mfu * 100 if opt.perf_report else 0.0
    opt_tpot_ms = (opt.tpot_seconds or 0.0) * 1000
    opt_tput = opt.cluster_tput_tps or 0.0

    if base_tput > 0 and opt_tput > 0:
        speedup_text = color_speedup(opt_tput / base_tput)
    else:
        speedup_text = "n/a"

    bs_text = f"bs={opt.max_batch_per_gpu}"
    if opt.bs_bound_by == "model-unbounded":
        bs_text += (
            ' <span style="color:#aaa">(α=1, sentinel — '
            "model-unbounded)</span>"
        )
    elif opt.max_batch_per_gpu < req_bs and opt.bs_bound_by:
        bs_text += (
            f' <span style="color:#aaa">(req={req_bs}, '
            f'{opt.bs_bound_by})</span>'
        )

    # Light-path prefetch overlap (prefix only — present whenever α > 0).
    overlap_text = ""
    if opt.indexer_prefetch_overlap_effective is not None:
        overlap_text = (
            f'<i style="color:#aaa">prefix prefetch overlap: '
            f'{opt.indexer_prefetch_overlap_effective*100:.1f}%</i>'
        )

    # Layer-spill diagnostic — only meaningful when k > 0.
    spill_text = ""
    k = opt.spilled_layers_count
    if k is not None and k > 0:
        L = _num_layers_from_report(opt)
        L_text = str(L) if L > 0 else "?"
        heavy_pct = (
            f"{opt.spilled_unique_overlap_effective*100:.1f}%"
            if opt.spilled_unique_overlap_effective is not None
            else "n/a"
        )
        spill_text = (
            f'<i style="color:#fa8">spill: {k}/{L_text} layers '
            f'(heavy overlap {heavy_pct})</i>'
        )

    extra_lines = []
    if overlap_text:
        extra_lines.append(overlap_text)
    if spill_text:
        extra_lines.append(spill_text)
    extra_html = ("<br/>" + "<br/>".join(extra_lines)) if extra_lines else ""

    return (
        '<div>'
        f'<b>{label}</b> ' + bs_text + '<br/>'
        f"penalty {opt.penalty_seconds*1000:.2f} ms<br/>"
        f"MFU {color_higher_is_better(opt_mfu, mfu_lo, mfu_hi, '.1f')}%, "
        f"TPOT {color_lower_is_better(opt_tpot_ms, tpot_lo, tpot_hi, '.2f')} ms<br/>"
        f"{opt_tput:,.0f} tok/s ({speedup_text})"
        + extra_html
        + '</div>'
    )


def fmt_combined_cell_s2(
    *,
    req_bs: int,
    actual_base_bs: int,
    base_rep,
    base_tput: float,
    was_degraded: bool,
    opt_s2: OptimizedCapacityReport,
    mfu_lo: float, mfu_hi: float,
    tpot_lo: float, tpot_hi: float,
) -> str:
    """Render one Scenario 2 cell: 2 stacked sub-blocks separated by <hr/>:
      1. BASELINE                (HBM-only)
      2. OPT (shared-prefix)     (single shared copy in DRAM, prefetch overlap)
    """
    sep = (
        '<hr style="margin:6px 0;border:0;border-top:1px dashed #ccc"/>'
    )
    parts: List[str] = []

    # ---------- (1) BASELINE block (identical to Scenario 1) ----------
    if base_rep is None or actual_base_bs < 1:
        parts.append(
            '<div style="color:#aaa;font-style:italic">'
            f'<b>base</b>: infeasible at bs={req_bs}</div>'
        )
    else:
        bs_text = (
            f"bs={actual_base_bs}"
            + (f' <span style="color:#aaa">(req={req_bs}, HBM-cap)</span>'
               if was_degraded else "")
        )
        mfu_pct = base_rep.total.mfu * 100 if base_rep else 0.0
        tpot_ms = base_rep.total.time_seconds * 1000 if base_rep else 0.0
        parts.append(
            '<div>'
            '<b>base</b> ' + bs_text + '<br/>'
            f"MFU {color_higher_is_better(mfu_pct, mfu_lo, mfu_hi, '.1f')}%, "
            f"TPOT {color_lower_is_better(tpot_ms, tpot_lo, tpot_hi, '.2f')} ms<br/>"
            f"{base_tput:,.0f} tok/s"
            '</div>'
        )

    # ---------- (2) OPT (shared-prefix) ----------
    parts.append(sep)
    parts.append(_render_opt_block_s2(
        opt=opt_s2, label="opt (shared-prefix)",
        req_bs=req_bs, base_tput=base_tput,
        mfu_lo=mfu_lo, mfu_hi=mfu_hi,
        tpot_lo=tpot_lo, tpot_hi=tpot_hi,
    ))

    return '<td class="b-opt">' + "".join(parts) + "</td>"


def render_mode_a_s2(args, gpu, mem, parallel, optimizer, dram_cfg) -> List[str]:
    """Mode A (legacy 2-col layout) for Scenario 2."""
    lines: List[str] = []
    lines.append("## Cell format")
    lines.append("")
    lines.append("Each row is one `session_length`. Two side-by-side blocks:")
    lines.append("")
    lines.append("- **BASELINE**: `bs` / weight (GiB,%) / session-KV (GiB,%HBM) / "
                 "MFU% / TPOT (ms) / cluster tput (tok/s)")
    lines.append("- **DRAM POOLING (shared-prefix)**: `bs (delta vs baseline)` / "
                 "penalty (ms) / MFU% / TPOT (ms) / cluster tput (speedup) / "
                 "prefetch overlap %")
    lines.append("")

    lines.append('<div class="cap-scroll">')
    lines.append('<table class="cap-tbl">')
    lines.append('<thead>')
    lines.append('<tr class="lvl1">'
                 '<th rowspan="2">session_length</th>'
                 '<th>BASELINE (HBM-only)</th>'
                 f'<th>SHARED-PREFIX DRAM POOLING (α='
                 f'{dram_cfg.prefix_share_frac:.2f}, '
                 f'+{dram_cfg.dram_capacity_gb:.0f} GiB @ '
                 f'{dram_cfg.dram_interconnect_bandwidth_gbs:.0f} GB/s)</th>'
                 '</tr>')
    lines.append('<tr class="lvl2"><th>baseline</th><th>opt (shared-prefix)</th></tr>')
    lines.append('</thead>')

    lines.append('<tbody>')
    n_feasible_base = 0
    n_feasible_opt = 0
    for s in args.sessions:
        baseline = analyze_capacity(
            gpu=gpu, mem=mem,
            model_name=args.model,
            parallel=parallel,
            session_length=s,
            context_length=args.context_length,
            dtype_kv=Dtype(args.kv_dtype),
        )
        optimized = optimizer.apply(
            baseline,
            gpu=gpu, mem=mem, parallel=parallel,
        )
        if baseline.feasible:
            n_feasible_base += 1
        if optimized.feasible:
            n_feasible_opt += 1

        row_label = f"<b>{session_label(s)}</b><br/>({s:,})"
        row_html = [f'<td class="row-label">{row_label}</td>']
        row_html.append(fmt_baseline_cell(
            baseline,
            mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
            tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
        ))
        # For Mode A use the existing fmt_optimized_cell — it shows
        # bs delta, penalty, MFU, TPOT, hot-cache (irrelevant in S2 but
        # rendered as `inf layers fittable` which is harmless), and the
        # prefetch overlap line. Acceptable for the legacy layout.
        row_html.append(fmt_optimized_cell(
            optimized,
            mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
            tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
            show_delta_bs=True,
        ))
        lines.append('<tr>' + ''.join(row_html) + '</tr>')
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    lines.append("")
    lines.append(
        f"_Summary: {n_feasible_base}/{len(args.sessions)} session lengths "
        f"feasible under HBM-only baseline; "
        f"{n_feasible_opt}/{len(args.sessions)} feasible with shared-prefix "
        f"DRAM Pooling at α={dram_cfg.prefix_share_frac:.2f}._"
    )
    return lines


def render_mode_b_s2(
    args, gpu, mem, parallel, optimizer_s2, dram_cfg,
    batch_sizes: List[int],
) -> List[str]:
    """Mode B (session × bs trade-off matrix) for Scenario 2."""
    lines: List[str] = []
    lines.append("## Cell format (Mode B: session × batch_size × {base, opt})")
    lines.append("")
    lines.append(
        "Each cell holds **two** stacked sub-blocks (separated by a "
        "dashed line) for the SAME requested batch size:"
    )
    lines.append("")
    lines.append(
        "- **base**: HBM-only baseline. If the requested bs exceeds HBM "
        "capacity, it is **degraded** to `hbm_max_batch` and the actual bs "
        'is shown as `bs=4 (req=64, HBM-cap)`.'
    )
    lines.append(
        "- **opt (shared-prefix)**: A single shared prefix copy lives in "
        "DRAM and is layer-stride prefetched. The per-session unique KV "
        "is filled into HBM layer-by-layer; if HBM cannot hold the full "
        "(1-α) × session_kv × bs, the TAIL k layers' unique spills to "
        "DRAM and is also layer-stride prefetched (`spill: k/L layers` "
        "shown in cell). `bs_bound_by` may be `HBM-cap` (capacity OK in "
        "HBM only, k=0), `DRAM-spill-cap` (capacity OK with spill, k>0), "
        "`DRAM-prefix-cap` (single prefix copy exceeds DRAM — infeasible), "
        "or `model-unbounded` (α=1.0 corner)."
    )
    lines.append(
        "- **prefix prefetch overlap %**: light path — fraction of "
        "(α × session_kv / L) DMA time hidden under (attn + ffn) per layer. "
        "100% means the prefix DMA is fully hidden."
    )
    lines.append(
        "- **heavy overlap %** (only when `k > 0`): fraction of "
        "(prefix_per_layer + bs × per_layer_unique) DMA time hidden under "
        "compute on each spilled layer. This is typically the dominant "
        "penalty source once spill kicks in (the bs multiplier dominates)."
    )
    lines.append("")

    lines.append(
        "- **speedup** (per opt block) = optimized cluster tput / baseline "
        "cluster tput at the SAME requested bs."
    )
    lines.append("")

    lines.append('<div class="cap-scroll">')
    lines.append('<table class="cap-tbl">')
    lines.append('<thead>')
    lines.append('<tr class="lvl1">'
                 f'<th rowspan="2">session_length<br/><i>(HBM max bs)</i></th>'
                 f'<th colspan="{len(batch_sizes)}">'
                 f'requested batch_size — base / opt (shared-prefix) '
                 f'(α={dram_cfg.prefix_share_frac:.2f}, '
                 f'+{dram_cfg.dram_capacity_gb:.0f} GiB @ '
                 f'{dram_cfg.dram_interconnect_bandwidth_gbs:.0f} GB/s)'
                 '</th>'
                 '</tr>')
    lines.append('<tr class="lvl2">'
                 + ''.join(f'<th>bs={bs}</th>' for bs in batch_sizes)
                 + '</tr>')
    lines.append('</thead>')

    lines.append('<tbody>')
    for s in args.sessions:
        baseline_cap = analyze_capacity(
            gpu=gpu, mem=mem,
            model_name=args.model,
            parallel=parallel,
            session_length=s,
            context_length=args.context_length,
            dtype_kv=Dtype(args.kv_dtype),
        )
        hbm_max = (
            baseline_cap.max_batch_per_gpu if baseline_cap.feasible else 0
        )
        row_label = (
            f"<b>{session_label(s)}</b><br/>({s:,})"
            f"<br/><i>HBM max bs={hbm_max}</i>"
        )
        row_html = [f'<td class="row-label">{row_label}</td>']

        for bs in batch_sizes:
            actual_base_bs, base_rep, base_tput, was_degraded = (
                _baseline_at_bs(
                    gpu=gpu, mem=mem, parallel=parallel, args=args,
                    baseline_capacity=baseline_cap, req_bs=bs,
                )
            )
            opt_s2 = optimizer_s2.apply(
                baseline_cap,
                gpu=gpu, mem=mem, parallel=parallel,
                batch_size_override=bs,
            )
            row_html.append(fmt_combined_cell_s2(
                req_bs=bs,
                actual_base_bs=actual_base_bs,
                base_rep=base_rep,
                base_tput=base_tput,
                was_degraded=was_degraded,
                opt_s2=opt_s2,
                mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
                tpot_lo=args.tpot_lo, tpot_hi=args.tpot_hi,
            ))
        lines.append('<tr>' + ''.join(row_html) + '</tr>')
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')
    return lines


# ============================================================================
#  Main
# ============================================================================


def _bw_filename(out_path: Path, bw_gbs: float) -> Path:
    """Append a `_bwNN` suffix before the extension, e.g.
    `dram_analysis.md` + 100.0 -> `dram_analysis_bw100.md`.
    Fractional BW (e.g. 12.5) becomes `bw12p5`.
    """
    if bw_gbs == int(bw_gbs):
        tag = f"bw{int(bw_gbs)}"
    else:
        tag = f"bw{bw_gbs}".replace(".", "p")
    return out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")


def _hit_filename(out_path: Path, hit_rate: float) -> Path:
    """Append a `_hitNN` suffix before the extension, encoding hit_rate as
    integer percent (0.99 -> hit99, 0.4 -> hit40, 0.0 -> hit0).
    Fractional percents (e.g. 0.995) become `hit99p5`.
    """
    pct = hit_rate * 100.0
    if pct == int(pct):
        tag = f"hit{int(pct)}"
    else:
        tag = f"hit{pct}".replace(".", "p")
    return out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")


def _pf_filename(out_path: Path, prefix_share: float) -> Path:
    """Append a `_pfNN` suffix before the extension, encoding
    prefix_share_frac as integer percent (1.0 -> pf100, 0.4 -> pf40,
    0.0 -> pf0). Fractional percents (e.g. 0.005) become `pf0p5`.
    """
    pct = prefix_share * 100.0
    if pct == int(pct):
        tag = f"pf{int(pct)}"
    else:
        tag = f"pf{pct}".replace(".", "p")
    return out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")


def _emit_one_report(
    *,
    args, gpu, mem, parallel,
    bw_gbs: float,
    hit_rate: float,
    prefix_share: float = 0.5,
    out_path: Path,
) -> None:
    """Render and write one report for a (bandwidth, hit_rate, prefix_share) triple.

    Routes by `args.scenario`:
      * `sparse_on_demand`: each cell has THREE sub-blocks (base /
        opt-idx-HBM / opt-idx-DRAM). `prefix_share` is ignored.
      * `shared_prefix`: each cell has TWO sub-blocks (base /
        opt-shared-prefix). `hit_rate` is ignored. `prefix_share` is
        the headline knob (passed as `prefix_share_frac`).
    """
    bs_list = parse_batch_sizes(args.batch_sizes)
    mode = "B" if bs_list else "A"

    if args.scenario == "shared_prefix":
        dram_cfg_s2 = DramPoolingConfig(
            dram_capacity_gb=args.dram_capacity_gb,
            dram_interconnect_bandwidth_gbs=bw_gbs,
            mode="shared_prefix",
            prefix_share_frac=prefix_share,
        )
        optimizer_s2 = DramPoolingOptimization(dram_cfg_s2)

        lines = render_header_s2(args, gpu, mem, parallel, dram_cfg_s2, mode)
        if mode == "A":
            lines.extend(render_mode_a_s2(
                args, gpu, mem, parallel, optimizer_s2, dram_cfg_s2,
            ))
        else:
            lines.extend(render_mode_b_s2(
                args, gpu, mem, parallel, optimizer_s2, dram_cfg_s2, bs_list,
            ))

        out_path.write_text("\n".join(lines) + "\n")
        print(
            f"wrote {out_path.resolve()} "
            f"(scenario shared_prefix, mode {mode}, "
            f"sessions={len(args.sessions)}"
            + (f", batch_sizes={len(bs_list)}" if bs_list else "")
            + f", dram_bw={bw_gbs:g} GB/s, α={prefix_share:g})"
        )
        return

    # ---------- Scenario 1 (default) ----------
    common_kwargs = dict(
        dram_capacity_gb=args.dram_capacity_gb,
        dram_interconnect_bandwidth_gbs=bw_gbs,
        kv_cache_hit_rate=hit_rate,
        # None → DramPoolingConfig falls back to `new_batch` at runtime
        # (pessimistic batch-wide miss upper bound).
        n_missing_sessions=(
            None if args.n_missing_sessions is None
            else int(args.n_missing_sessions)
        ),
        hot_slots=args.hot_slots,
    )
    dram_cfg_hbm = DramPoolingConfig(
        **common_kwargs, indexer_in_dram=False,
    )
    dram_cfg_dram = DramPoolingConfig(
        **common_kwargs, indexer_in_dram=True,
    )
    optimizer_hbm = DramPoolingOptimization(dram_cfg_hbm)
    optimizer_dram = DramPoolingOptimization(dram_cfg_dram)

    # The header & Mode A only need a single representative cfg (the two
    # configs differ only in the indexer_in_dram flag, which the header
    # describes prose-style now). Use the HBM-resident one for headers.
    dram_cfg = dram_cfg_hbm

    lines = render_header(args, gpu, mem, parallel, dram_cfg, mode)
    if mode == "A":
        # Mode A is legacy (single-opt 2-column table). Keep it
        # backwards-compatible: render the indexer-in-HBM optimizer only.
        lines.extend(render_mode_a(args, gpu, mem, parallel, optimizer_hbm, dram_cfg))
    else:
        lines.extend(render_mode_b(
            args, gpu, mem, parallel,
            optimizer_hbm, optimizer_dram,
            dram_cfg, bs_list,
        ))

    out_path.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {out_path.resolve()} "
        f"(scenario sparse_on_demand, mode {mode}, "
        f"sessions={len(args.sessions)}"
        + (f", batch_sizes={len(bs_list)}" if bs_list else "")
        + f", dram_bw={bw_gbs:g} GB/s, hit={hit_rate:g})"
    )


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    # Switching-parallel constraint (decode-only): DP × TP = EP.
    # Validate up-front so the user gets a clear error before we spend
    # any time generating reports.
    if args.dp * args.tp != args.ep:
        sys.stderr.write(
            f"ERROR: switching-parallel constraint violated: "
            f"DP × TP = {args.dp} × {args.tp} = {args.dp * args.tp}, "
            f"but EP = {args.ep}.\n"
            f"  V3/V3.2 decode requires the same physical GPUs to run "
            f"TP×DP for attention and reshape to EP for FFN.\n"
            f"  Adjust --dp / --tp / --ep so that DP × TP == EP "
            f"(e.g. --tp 4 --dp 8 --ep 32, or --tp 8 --dp 8 --ep 64).\n"
        )
        sys.exit(2)

    # Handle deprecated alias --dram-bandwidth-gbs.
    if args.dram_bandwidth_gbs is not None:
        sys.stderr.write(
            "WARN: --dram-bandwidth-gbs is deprecated; "
            "use --dram-interconnect-bandwidth-gbs instead. "
            f"Mapping value {args.dram_bandwidth_gbs} -> "
            "--dram-interconnect-bandwidth-gbs.\n"
        )
        args.dram_interconnect_bandwidth_gbs = args.dram_bandwidth_gbs

    # Handle deprecated alias --runtime-context-length.
    if args._legacy_runtime_context_length is not None:
        sys.stderr.write(
            "WARN: --runtime-context-length is deprecated; "
            "use --context-length instead. Mapping value "
            f"{args._legacy_runtime_context_length} -> --context-length.\n"
        )
        args.context_length = args._legacy_runtime_context_length

    # Sentinel: --context-length -1 means "use session_length per row".
    # Internally we represent that as None (analyze_capacity fallback).
    if args.context_length is not None and args.context_length < 0:
        args.context_length = None
    # ---- Resolve specs -----------------------------------------------------
    gpu = GPU_PRESETS[args.gpu]
    if args.hbm_capacity_gb is not None:
        mem = MemoryProfile(
            hbm_capacity_gb=args.hbm_capacity_gb,
            weight_overhead_frac=args.hbm_overhead_frac,
            overhead_fixed_gb=args.hbm_overhead_fixed_gb,
        )
    else:
        base_mem = MEMORY_PRESETS.get(
            args.gpu, MemoryProfile(hbm_capacity_gb=80.0)
        )
        mem = MemoryProfile(
            hbm_capacity_gb=base_mem.hbm_capacity_gb,
            weight_overhead_frac=args.hbm_overhead_frac,
            overhead_fixed_gb=args.hbm_overhead_fixed_gb,
        )

    parallel = ParallelConfig(
        tp=args.tp, ep=args.ep, pp=args.pp, dp=args.dp,
    )

    out_path = Path(args.out)

    # ---- Scenario / sweep flag consistency ---------------------------------
    n_sweep_flags = sum(
        bool(x) for x in (args.bw_sweep, args.hit_rate_sweep, args.prefix_share_sweep)
    )
    if n_sweep_flags > 1:
        sys.stderr.write(
            "ERROR: --bw-sweep, --hit-rate-sweep, and --prefix-share-sweep "
            "are mutually exclusive. Pick at most one.\n"
        )
        sys.exit(2)

    if args.scenario == "sparse_on_demand" and args.prefix_share_sweep:
        sys.stderr.write(
            "ERROR: --prefix-share-sweep requires --scenario shared_prefix.\n"
        )
        sys.exit(2)

    if args.scenario == "shared_prefix" and args.hit_rate_sweep:
        sys.stderr.write(
            "ERROR: --hit-rate-sweep is meaningless under --scenario "
            "shared_prefix (no sparse-attention hit_rate concept). Use "
            "--prefix-share-sweep instead.\n"
        )
        sys.exit(2)

    if args.bw_sweep:
        bw_values = [float(x.strip()) for x in args.bw_sweep.split(",") if x.strip()]
        if not bw_values:
            sys.stderr.write("ERROR: --bw-sweep is empty after parsing.\n")
            sys.exit(2)
        for bw in bw_values:
            _emit_one_report(
                args=args, gpu=gpu, mem=mem, parallel=parallel,
                bw_gbs=bw,
                hit_rate=args.hit_rate,
                prefix_share=args.prefix_share_frac,
                out_path=_bw_filename(out_path, bw),
            )
    elif args.hit_rate_sweep:
        hit_values = [float(x.strip()) for x in args.hit_rate_sweep.split(",") if x.strip()]
        if not hit_values:
            sys.stderr.write("ERROR: --hit-rate-sweep is empty after parsing.\n")
            sys.exit(2)
        for h in hit_values:
            if not 0.0 <= h <= 1.0:
                sys.stderr.write(
                    f"ERROR: --hit-rate-sweep value {h} out of range [0, 1].\n"
                )
                sys.exit(2)
        for h in hit_values:
            _emit_one_report(
                args=args, gpu=gpu, mem=mem, parallel=parallel,
                bw_gbs=args.dram_interconnect_bandwidth_gbs,
                hit_rate=h,
                prefix_share=args.prefix_share_frac,
                out_path=_hit_filename(out_path, h),
            )
    elif args.prefix_share_sweep:
        pf_values = [float(x.strip()) for x in args.prefix_share_sweep.split(",") if x.strip()]
        if not pf_values:
            sys.stderr.write("ERROR: --prefix-share-sweep is empty after parsing.\n")
            sys.exit(2)
        for pf in pf_values:
            if not 0.0 <= pf <= 1.0:
                sys.stderr.write(
                    f"ERROR: --prefix-share-sweep value {pf} out of range [0, 1].\n"
                )
                sys.exit(2)
        for pf in pf_values:
            _emit_one_report(
                args=args, gpu=gpu, mem=mem, parallel=parallel,
                bw_gbs=args.dram_interconnect_bandwidth_gbs,
                hit_rate=args.hit_rate,
                prefix_share=pf,
                out_path=_pf_filename(out_path, pf),
            )
    else:
        _emit_one_report(
            args=args, gpu=gpu, mem=mem, parallel=parallel,
            bw_gbs=args.dram_interconnect_bandwidth_gbs,
            hit_rate=args.hit_rate,
            prefix_share=args.prefix_share_frac,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
