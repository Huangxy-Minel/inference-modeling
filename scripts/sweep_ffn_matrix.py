#!/usr/bin/env python3
"""Sweep FFN cost across (EP × batch_size) and emit a Markdown report.

Each cell shows 5 metrics:
    FLOPs (GFLOPs), Bytes (MiB), MFU (%), TPOT (ms/token), Tput (tok/s, cluster).

TPOT here is the FFN-stage wall-clock only (upper bound for the full step);
real decode TPOT = attn + ffn (see sweep_e2e_matrix).

Note: for decode, FFN cost is independent of context length N.

Switching-parallel constraint (decode-only): DP × TP = EP. Since EP is the
sweep axis here, DP is derived per row as `dp = ep / tp` (must be exact;
non-divisible combos fail-fast). TP is fixed across the sweep.

Usage:
    cd scripts/inference-modeling
    python3 -m scripts.sweep_ffn_matrix
    python3 -m scripts.sweep_ffn_matrix --gpu GB200 --model deepseek-v3.2 \
        --tp 4 --ep 8,16,32,64,128 --out ffn_matrix.md
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
    ap.add_argument("--tp",    type=int, default=1,
                    help="TP factor (fixed across the EP sweep). "
                         "DP is auto-derived per row as dp = ep / tp.")
    ap.add_argument("--ctx",   type=int, default=9216,
                    help="context length (FFN cost is independent of N for decode)")
    ap.add_argument("--ep",    type=parse_int_list,
                    default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--batch", type=parse_int_list,
                    default=[32, 64, 128, 256])
    ap.add_argument("--out",   default="ffn_matrix.md")
    args = ap.parse_args()

    # Switching-parallel constraint: validate EP % TP == 0 for every row.
    bad_eps = [ep for ep in args.ep if ep % args.tp != 0]
    if bad_eps:
        sys.stderr.write(
            f"ERROR: switching-parallel constraint violated: TP={args.tp} "
            f"does not divide the following EP values: {bad_eps}.\n"
            f"  Decode requires DP × TP = EP, so DP = EP / TP must be an "
            f"integer. Either change --tp or remove the bad EP values.\n"
        )
        sys.exit(2)

    gpu = GPU_PRESETS[args.gpu]
    est = get_model(args.model)
    boundary_ai = (gpu.peak_tflops * 1000.0) / gpu.bandwidth_gbs

    lines: List[str] = []
    lines.append(f"# FFN sweep: EP × batch_size (per GPU)")
    lines.append("")
    lines.append(f"- **Model**: {args.model}")
    lines.append(f"- **GPU**: {gpu.name} "
                 f"(peak {gpu.peak_tflops:.0f} TF/s, BW {gpu.bandwidth_gbs:.0f} GB/s, "
                 f"boundary AI {boundary_ai:.0f} F/B)")
    lines.append(f"- **Parallelism**: TP={args.tp} (fixed), "
                 f"DP=EP/TP (auto-derived per row), ctx={args.ctx:,} "
                 f"(FFN cost is N-invariant for decode)")
    lines.append(f"- **Cell metrics** (5 rows per cell):")
    lines.append(f"  - FLOPs (GFLOPs)")
    lines.append(f"  - Bytes (MiB)")
    lines.append(f"  - MFU (%)")
    lines.append(f"  - TPOT (ms/token; for ffn-only stage = t_ffn)")
    lines.append(f"  - Tput (tok/s, cluster, ffn-stage upper bound = "
                 f"batch × DP × TP / t_ffn)")
    lines.append("")

    header = ["EP \\\\ batch (DP)"] + [f"{b}" for b in args.batch]
    sep    = ["---"]                + ["---:" for _ in args.batch]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep)    + " |")

    for EP in args.ep:
        DP = EP // args.tp
        parallel = ParallelConfig(tp=args.tp, ep=EP, pp=1, dp=DP)
        cluster_factor = DP * args.tp  # = EP, by construction
        row_cells = [f"**{EP}** (DP={DP})"]
        for bs in args.batch:
            wl = WorkloadConfig(phase=Phase.DECODE, batch_size=bs, context_length=args.ctx)
            cost = est.compute_ffn_cost(parallel, wl)
            rep = evaluate_stage("ffn", cost, gpu)
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
