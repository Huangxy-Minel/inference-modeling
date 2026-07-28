#!/usr/bin/env python3
"""End-to-end (attn + FFN) decode sweep across (cluster_size × batch_size).

Constraints:
    cluster_size  = number of GPUs in one decode serving unit
    EP            = cluster_size                  (FFN sharding)
    DP * TP       = EP                            (attn parallelism)
    The split between DP and TP is auto-chosen (TP preferred from --prefer-tp).

Each cell is a 3-column **mini-table** showing **ATTN / FFN / TOTAL** side by
side. Each mini-column shows 5 metrics:
    FLOPs (G), Bytes (MiB), MFU (%), TPOT (ms/token), Tput (tok/s, cluster).

Three "interesting" metrics (MFU, TPOT, Tput) are color-graded with three tiers
using `<span style="color:...">` inline HTML (renders in VSCode / Typora /
Obsidian / GitHub).

Tier thresholds (default; tweak via CLI flags):
    MFU%        :  <20 red ;  20-60 yellow ;  >=60 green   (higher better)
    TPOT total  :  >30  red ;  10-30 yellow ; <=10 green ms (lower better)
    TPOT stage  :  >5   red ;  1-5  yellow ; <=1  green ms (lower better)

Only `total.TPOT` is the real per-token decode latency (shared by all B users
in the batch). attn/ffn TPOT/tput are per-stage upper bounds.

Usage:
    cd scripts/inference-modeling
    python3 -m scripts.sweep_e2e_matrix
    python3 -m scripts.sweep_e2e_matrix --ctx 9216 --out e2e_matrix.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from simulator import (
    GPU_PRESETS,
    ParallelConfig,
    Phase,
    WorkloadConfig,
    estimate_perf,
)
from simulator.models import MODEL_REGISTRY


# ============================================================================
#  Color helpers (HTML <span> inline styling — works everywhere).
# ============================================================================
#
#  We use <span style="color:..."> which renders correctly inside HTML
#  <table> blocks in VSCode / Typora / Obsidian / GitHub. This avoids the
#  limitation of $\color{...}{...}$ LaTeX math mode which only works in
#  markdown text nodes but NOT inside raw HTML blocks.

_COLOR_RED    = "red"
_COLOR_YELLOW = "orange"   # "yellow" renders almost invisibly on white bg
_COLOR_GREEN  = "green"


def _html_color(text: str, color: str) -> str:
    """Wrap `text` in `<span style="color:...">` for inline coloring."""
    return f'<span style="color:{color}">{text}</span>'


def color_higher_is_better(value: float, lo: float, hi: float, fmt: str) -> str:
    """Render `value` colored:  <lo red,  [lo,hi) yellow,  >=hi green."""
    text = format(value, fmt)
    if value < lo:
        c = _COLOR_RED
    elif value < hi:
        c = _COLOR_YELLOW
    else:
        c = _COLOR_GREEN
    return _html_color(text, c)


def color_lower_is_better(value: float, lo: float, hi: float, fmt: str) -> str:
    """Render `value` colored:  <=lo green,  (lo,hi] yellow,  >hi red."""
    text = format(value, fmt)
    if value <= lo:
        c = _COLOR_GREEN
    elif value <= hi:
        c = _COLOR_YELLOW
    else:
        c = _COLOR_RED
    return _html_color(text, c)


# ============================================================================
#  Sweep
# ============================================================================


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def split_dp_tp(ep: int, prefer_tp: List[int]) -> Tuple[int, int]:
    """Pick (DP, TP) such that DP*TP = EP, with TP preferred from `prefer_tp`."""
    for tp in prefer_tp:
        if ep % tp == 0:
            return ep // tp, tp
    return ep, 1


def fmt_stage_lines(
    rep,
    tokens_per_step: int,
    is_total: bool,
    mfu_lo: float,
    mfu_hi: float,
    tpot_total_lo: float,
    tpot_total_hi: float,
    tpot_stage_lo: float,
    tpot_stage_hi: float,
) -> str:
    """Render one stage report as 5 `<br/>`-joined lines, with MFU / TPOT / tput
    color-graded. Returned string fits inside one `<td>` of the mini-table."""
    tput = (tokens_per_step / rep.time_seconds) if rep.time_seconds > 0 else 0.0
    tpot_ms = rep.time_seconds * 1e3

    mfu_text  = color_higher_is_better(rep.mfu * 100, mfu_lo, mfu_hi, ".1f") + "%"
    if is_total:
        tpot_text = color_lower_is_better(tpot_ms, tpot_total_lo, tpot_total_hi, ".2f")
    else:
        tpot_text = color_lower_is_better(tpot_ms, tpot_stage_lo, tpot_stage_hi, ".2f")
    # Tput: simple log-ish tier on the raw number; only color the order of mag.
    # We bucket on "M tok/s" to keep the visual signal aligned with cluster scale.
    tput_M = tput / 1e6
    if tput_M >= 1.0:
        tput_text = _html_color(f"{tput:,.0f}", _COLOR_GREEN)
    elif tput_M >= 0.1:
        tput_text = _html_color(f"{tput:,.0f}", _COLOR_YELLOW)
    else:
        tput_text = _html_color(f"{tput:,.0f}", _COLOR_RED)

    return (
        f"{rep.flops/1e9:,.1f} G<br/>"
        f"{rep.bytes_/(1024**2):,.1f} MiB<br/>"
        f"MFU {mfu_text}<br/>"
        f"TPOT {tpot_text} ms<br/>"
        f"{tput_text} tok/s"
    )


def fmt_three_tds(
    rep,
    tokens_per_step: int,
    mfu_lo: float, mfu_hi: float,
    tpot_total_lo: float, tpot_total_hi: float,
    tpot_stage_lo: float, tpot_stage_hi: float,
) -> str:
    """Render one (cluster, batch) cell as THREE sibling <td>'s for ATTN / FFN
    / TOTAL. Each <td> contains a 5-line stage report (colorized MFU / TPOT
    / tput).

    Returned HTML is meant to be concatenated into a parent <tr>; we output
    three <td>s so the outer table can lay out batch × {ATTN,FFN,TOTAL} as a
    flat grid (no nested tables). This avoids the auto-wrap problem you saw
    when each cell had its own mini-table.
    """
    common = dict(
        tokens_per_step=tokens_per_step,
        mfu_lo=mfu_lo, mfu_hi=mfu_hi,
        tpot_total_lo=tpot_total_lo, tpot_total_hi=tpot_total_hi,
        tpot_stage_lo=tpot_stage_lo, tpot_stage_hi=tpot_stage_hi,
    )
    attn_html  = fmt_stage_lines(rep.attn,  is_total=False, **common)
    ffn_html   = fmt_stage_lines(rep.ffn,   is_total=False, **common)
    total_html = fmt_stage_lines(rep.total, is_total=True,  **common)

    return (
        f'<td class="m-attn">{attn_html}</td>'
        f'<td class="m-ffn">{ffn_html}</td>'
        f'<td class="m-tot">{total_html}</td>'
    )


# ============================================================================
#  Style block: keeps the matrix readable even when many batch columns are
#  rendered in a narrow viewport. Inlined at the top of the markdown so it
#  works in VSCode / Typora / Obsidian out-of-the-box. (GitHub web strips
#  some style attrs but keeps `white-space: nowrap`, which is the critical
#  one.)
# ============================================================================

_STYLE_BLOCK = """\
<style>
.mfutbl { border-collapse: collapse; min-width: 100%; font-size: 13px; }
.mfutbl th, .mfutbl td {
  white-space: nowrap;            /* never wrap numbers / units */
  padding: 4px 8px;
  border: 1px solid #555;
  vertical-align: top;
  text-align: left;
}
.mfutbl thead th { background: #2b2b2b; color: #eee; }
.mfutbl thead tr.lvl1 th { text-align: center; }
.mfutbl thead tr.lvl2 th { font-weight: 600; }
.mfutbl tbody td.row-label { font-weight: 600; background: #232323; color: #eee; }
.mfutbl td.m-attn { background: #181f25; }   /* subtle column tints */
.mfutbl td.m-ffn  { background: #1f1820; }
.mfutbl td.m-tot  { background: #1a2218; }
/* horizontal scroll wrapper so a wide matrix gets a scrollbar instead of
   wrapping the whole row to a new line */
.mfu-scroll { overflow-x: auto; max-width: 100%; }
</style>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu",   default="GB200", choices=list(GPU_PRESETS.keys()))
    ap.add_argument("--model", default="deepseek-v3.2",
                    choices=sorted(MODEL_REGISTRY.keys()))
    ap.add_argument("--ctx",   type=int, default=9216)
    ap.add_argument("--cluster", type=parse_int_list, default=[8, 16, 32, 64, 128],
                    help="cluster sizes to sweep (= EP)")
    ap.add_argument("--batch",   type=parse_int_list, default=[16, 32, 64, 128, 256])
    ap.add_argument("--prefer-tp", type=parse_int_list, default=[8, 4, 2, 1],
                    help="preferred TP values, in order")
    ap.add_argument("--out",   default="e2e_matrix.md")
    # Color tier thresholds.
    ap.add_argument("--mfu-lo",  type=float, default=20.0,
                    help="MFU%% below this is red (default 20)")
    ap.add_argument("--mfu-hi",  type=float, default=60.0,
                    help="MFU%% at/above this is green (default 60)")
    ap.add_argument("--tpot-total-lo", type=float, default=10.0,
                    help="total TPOT (ms) at/below this is green (default 10)")
    ap.add_argument("--tpot-total-hi", type=float, default=30.0,
                    help="total TPOT (ms) above this is red (default 30)")
    ap.add_argument("--tpot-stage-lo", type=float, default=1.0,
                    help="single-stage TPOT (ms) at/below this is green (default 1)")
    ap.add_argument("--tpot-stage-hi", type=float, default=5.0,
                    help="single-stage TPOT (ms) above this is red (default 5)")
    args = ap.parse_args()

    gpu = GPU_PRESETS[args.gpu]
    boundary_ai = (gpu.peak_tflops * 1000.0) / gpu.bandwidth_gbs

    lines: List[str] = []
    # Style block goes first so renderers see it before the table.
    lines.append(_STYLE_BLOCK)
    lines.append(f"# End-to-end decode sweep: cluster_size × batch_size")
    lines.append("")
    lines.append(f"- **Model**: {args.model}")
    lines.append(f"- **GPU**: {gpu.name} "
                 f"(peak {gpu.peak_tflops:.0f} TF/s, BW {gpu.bandwidth_gbs:.0f} GB/s, "
                 f"boundary AI {boundary_ai:.0f} F/B)")
    lines.append(f"- **Context length**: {args.ctx:,}")
    lines.append(f"- **Constraint**: EP = cluster_size, DP × TP = EP "
                 f"(TP preferred from {args.prefer_tp})")
    lines.append("")
    lines.append(f"## Cell format")
    lines.append(f"Each `batch` column is split into 3 sub-columns — "
                 f"**ATTN / FFN / TOTAL** — with 5 metrics per sub-column: "
                 f"FLOPs (G) / Bytes (MiB) / MFU% / TPOT (ms/token) / tput "
                 f"(tok/s, per cluster).")
    lines.append("")
    lines.append("Note: only the **TOTAL** sub-column's TPOT is the real per-token "
                 "decode latency. ATTN/FFN TPOT and tput are per-stage upper "
                 "bounds (`batch × DP × TP / t_stage`).")
    lines.append("")
    lines.append("### Color legend (MFU / TPOT / tput)")
    lines.append("")
    lines.append(
        f"- **MFU%**: "
        f"{_html_color('< ' + format(args.mfu_lo, '.0f'), _COLOR_RED)} red, "
        f"{_html_color(format(args.mfu_lo, '.0f') + '-' + format(args.mfu_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= ' + format(args.mfu_hi, '.0f'), _COLOR_GREEN)} green "
        "(higher is better)."
    )
    lines.append(
        f"- **TOTAL TPOT (ms)**: "
        f"{_html_color('<= ' + format(args.tpot_total_lo, '.0f'), _COLOR_GREEN)} green, "
        f"{_html_color(format(args.tpot_total_lo, '.0f') + '-' + format(args.tpot_total_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('> ' + format(args.tpot_total_hi, '.0f'), _COLOR_RED)} red "
        "(lower is better)."
    )
    lines.append(
        f"- **single-stage TPOT (ms)**: "
        f"{_html_color('<= ' + format(args.tpot_stage_lo, '.0f'), _COLOR_GREEN)} green, "
        f"{_html_color(format(args.tpot_stage_lo, '.0f') + '-' + format(args.tpot_stage_hi, '.0f'), _COLOR_YELLOW)} yellow, "
        f"{_html_color('> ' + format(args.tpot_stage_hi, '.0f'), _COLOR_RED)} red "
        "(lower is better)."
    )
    lines.append(
        f"- **tput (tok/s)**: "
        f"{_html_color('< 100K', _COLOR_RED)} red, "
        f"{_html_color('100K-1M', _COLOR_YELLOW)} yellow, "
        f"{_html_color('>= 1M', _COLOR_GREEN)} green."
    )
    lines.append("")

    lines.append(f"## Resolved (DP, TP) per cluster")
    lines.append("")
    lines.append("| cluster | DP | TP |")
    lines.append("| --- | ---: | ---: |")
    for cs in args.cluster:
        dp, tp = split_dp_tp(cs, args.prefer_tp)
        lines.append(f"| {cs} | {dp} | {tp} |")
    lines.append("")

    # ---- main matrix as a single HTML <table>, NOT a markdown table ---------
    # Two-level header:
    #   Level 1 (lvl1):  cluster \\ batch  | batch=16 (colspan=3) | batch=32 ...
    #   Level 2 (lvl2):  (empty)           | ATTN  FFN  TOTAL     | ATTN ...
    #
    # The wrapping <div class="mfu-scroll"> gives a horizontal scrollbar when
    # the matrix is wider than the container, so wrapping never has to happen.
    lines.append('<div class="mfu-scroll">')
    lines.append('<table class="mfutbl">')
    lines.append('<thead>')

    # Level 1
    h1 = ['<th rowspan="2">cluster \\ batch</th>']
    for b in args.batch:
        h1.append(f'<th colspan="3">batch={b}</th>')
    lines.append('<tr class="lvl1">' + ''.join(h1) + '</tr>')

    # Level 2
    h2 = []
    for _ in args.batch:
        h2.append('<th>ATTN</th><th>FFN</th><th>TOTAL</th>')
    lines.append('<tr class="lvl2">' + ''.join(h2) + '</tr>')
    lines.append('</thead>')

    lines.append('<tbody>')
    for cs in args.cluster:
        dp, tp = split_dp_tp(cs, args.prefer_tp)
        ep = cs
        parallel = ParallelConfig(tp=tp, ep=ep, pp=1, dp=dp)
        row_label = f"<b>{cs}</b><br/>(DP={dp},TP={tp},EP={ep})"
        row_html = [f'<td class="row-label">{row_label}</td>']
        for bs in args.batch:
            wl = WorkloadConfig(phase=Phase.DECODE,
                                batch_size=bs,
                                context_length=args.ctx)
            rep = estimate_perf(gpu=gpu, model_name=args.model,
                               parallel=parallel, workload=wl)
            tokens_per_step = bs * dp * tp
            row_html.append(fmt_three_tds(
                rep, tokens_per_step,
                mfu_lo=args.mfu_lo, mfu_hi=args.mfu_hi,
                tpot_total_lo=args.tpot_total_lo,
                tpot_total_hi=args.tpot_total_hi,
                tpot_stage_lo=args.tpot_stage_lo,
                tpot_stage_hi=args.tpot_stage_hi,
            ))
        lines.append('<tr>' + ''.join(row_html) + '</tr>')
    lines.append('</tbody>')
    lines.append('</table>')
    lines.append('</div>')

    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
