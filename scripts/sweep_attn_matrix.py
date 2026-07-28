#!/usr/bin/env python3
"""Sweep attention cost across (context_length × batch_size) and emit a Markdown report.

Each cell shows 5 metrics:
    FLOPs (GFLOPs), Bytes (MiB), MFU (%), TPOT (ms/token), Tput (tok/s, cluster).

TPOT here is the attention-stage wall-clock only (upper bound for the full step);
real decode TPOT = attn + ffn (see sweep_e2e_matrix).

Usage:
    cd scripts/inference-modeling
    python3 -m scripts.sweep_attn_matrix
    python3 -m scripts.sweep_attn_matrix --gpu GB200 --model deepseek-v3.2 \
        --dp 4 --tp 2 --ep 8 --out attn_matrix.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from simulator import GPU_PRESETS, ParallelConfig, Phase, WorkloadConfig, evaluate_stage
from simulator.models import MODEL_REGISTRY, get_model


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu",   default="GB200", choices=list(GPU_PRESETS.keys()))
    ap.add_argument("--model", default="deepseek-v3.2",
                    choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument("--tp",    type=int, default=1)
    ap.add_argument("--dp",    type=int, default=1)
    ap.add_argument("--ep",    type=int, default=1)
    ap.add_argument("--ctx",   type=parse_int_list,
                    default=[2048, 4096, 9216, 16384, 30720])
    ap.add_argument("--batch", type=parse_int_list,
                    default=[32, 64, 128, 256])
    ap.add_argument("--out",   default="attn_matrix.md",
                    help="output markdown file path (relative to scripts/)")
    args = ap.parse_args()

    # Switching-parallel constraint (decode-only): DP × TP = EP.
    if args.dp * args.tp != args.ep:
        sys.stderr.write(
            f"ERROR: switching-parallel constraint violated: "
            f"DP × TP = {args.dp} × {args.tp} = {args.dp * args.tp}, "
            f"but EP = {args.ep}.\n"
            f"  Adjust --dp / --tp / --ep so that DP × TP == EP "
            f"(e.g. --tp 4 --dp 8 --ep 32).\n"
        )
        sys.exit(2)

    gpu = GPU_PRESETS[args.gpu]
    est = get_model(args.model)
    parallel = ParallelConfig(tp=args.tp, ep=args.ep, pp=1, dp=args.dp)
    boundary_ai = (gpu.peak_tflops * 1000.0) / gpu.bandwidth_gbs

    lines: List[str] = []
    lines.append(f"# Attention sweep: context_length × batch_size (per GPU)")
    lines.append("")
    lines.append(f"- **Model**: {args.model}")
    lines.append(f"- **GPU**: {gpu.name} "
                 f"(peak {gpu.peak_tflops:.0f} TF/s, BW {gpu.bandwidth_gbs:.0f} GB/s, "
                 f"boundary AI {boundary_ai:.0f} F/B)")
    lines.append(f"- **Parallelism**: TP={args.tp}, DP={args.dp}, EP={args.ep}")
    lines.append(f"  - attn per-GPU metrics are invariant in DP "
                 f"(DP replicates; only affects cluster-wide throughput).")
    lines.append(f"  - cluster token throughput per step = "
                 f"batch_per_GPU × DP × TP = batch × {args.dp * args.tp}")
    lines.append(f"- **Cell metrics** (5 rows per cell):")
    lines.append(f"  - FLOPs (GFLOPs)")
    lines.append(f"  - Bytes (MiB)")
    lines.append(f"  - MFU (%)")
    lines.append(f"  - TPOT (ms/token; for attn-only stage = t_attn)")
    lines.append(f"  - Tput (tok/s, cluster, attn-stage upper bound = "
                 f"batch × DP × TP / t_attn)")
    lines.append("")

    # Header row: batch sizes as columns, ctx as rows.
    header = ["N \\\\ batch"] + [f"{b}" for b in args.batch]
    sep    = ["---"]        + ["---:" for _ in args.batch]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep)    + " |")

    cluster_factor = args.dp * args.tp  # tokens-per-step multiplier
    for N in args.ctx:
        row_cells = [f"**{N:,}**"]
        for bs in args.batch:
            wl = WorkloadConfig(phase=Phase.DECODE, batch_size=bs, context_length=N)
            cost = est.compute_attn_cost(parallel, wl)
            rep = evaluate_stage("attention", cost, gpu)
            tput = (bs * cluster_factor / rep.time_seconds) if rep.time_seconds > 0 else 0.0
            cell = (
                f"{rep.flops/1e9:,.2f} GFLOPs<br/>"
                f"{rep.bytes_/(1024**2):,.2f} MiB<br/>"
                f"{rep.mfu*100:.2f}%<br/>"
                f"TPOT {rep.time_seconds*1e3:.3f} ms<br/>"
                f"{tput:,.0f} tok/s"
            )
            row_cells.append(cell)
        lines.append("| " + " | ".join(row_cells) + " |")

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
