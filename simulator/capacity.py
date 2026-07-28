"""Decode-phase capacity analysis (HBM-only baseline).

Answers a single question: under HBM-only constraints, given a model + GPU +
parallel config + session length budget, what is the maximum batch size
this deployment can serve, and what does its decode performance look like?

This module is intentionally model-agnostic. It calls two black-box methods
on `ModelCostEstimator`:
    * `session_kv_bytes(session_length, parallel, dtype_kv)` -> per-GPU bytes
    * `weight_bytes_per_gpu(parallel, dtype_param)`           -> per-GPU bytes

and treats the rest of the model internals as opaque. Performance numbers
come from the existing perf engine (`simulator.core.estimate_perf`).

Optimizations that *break* the HBM-only assumption (DRAM Pooling, prefetch,
overlap, KV offloading, ...) live in `simulator.optimizations.*` and consume the
baseline `CapacityReport` produced here.

Terminology — two distinct sequence-length parameters:

    session_length  (capacity dimension):
        Deployment-level upper bound on per-session KV history. Sets the
        KV-cache budget and therefore `max_batch_per_gpu` under HBM
        constraints. Typical values: 128*1024, 1024*1024, 4*1024*1024.

    context_length  (runtime dimension, i.e. WorkloadConfig.context_length):
        Number of tokens the model attends over in ONE decode step. Drives
        attn FLOPs / bytes / MFU / TPOT via `estimate_perf`. Independent
        of session_length: a session sized for 128K may be at runtime step
        N=9216 (early in the conversation) or N=128K (late, near capacity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .core import (
    Dtype,
    GPUSpec,
    PerfReport,
    ParallelConfig,
    Phase,
    WorkloadConfig,
    estimate_perf,
)
from .models import get_model


# ============================================================================
#  Memory profile (HBM-only)
#
#  GPUSpec only carries compute / bandwidth specs. We add HBM capacity as a
#  separate dataclass so the capacity tool stays independent of GPUSpec and
#  doesn't force a breaking change there. Optimizations like DRAM Pooling
#  carry their own config (in simulator.optimizations.*) and DO NOT extend this.
# ============================================================================


@dataclass(frozen=True)
class MemoryProfile:
    """HBM capacity (and other on-device storage) for one GPU.

    Overhead is modelled in TWO parts (total = proportional + fixed):

      * `weight_overhead_frac` — PROPORTIONAL: a fraction of HBM capacity,
        for state that scales with device size (allocator reserve, kernel
        scratch, ...). Default 0.05 = 5%.
      * `overhead_fixed_gb` — FIXED (absolute GiB), for state that is
        essentially constant regardless of HBM size: CUDA-graph capture,
        DeepEP/NCCL comm buffers, framework runtime reserve. For DeepEP-based
        MoE decode this dominates (~15-20 GiB even on a 96 GiB card) and the
        5% proportional term badly under-counts it.
        `None` (default) means "not specified here" → the capacity engine
        falls back to the per-model, per-GPU CALIBRATED value
        (`ModelCostEstimator.overhead_fixed_gb_by_gpu`), or 0 if that model
        has no calibration. An explicit float always wins.

    Attributes:
        hbm_capacity_gb: Total HBM physical capacity, in GiB.
        weight_overhead_frac: Proportional overhead as a fraction of HBM.
        overhead_fixed_gb: Fixed overhead in GiB, or None to defer to the
            model's calibration.
    """

    hbm_capacity_gb: float
    weight_overhead_frac: float = 0.05
    overhead_fixed_gb: Optional[float] = None

    @property
    def hbm_capacity_bytes(self) -> float:
        return self.hbm_capacity_gb * (1024 ** 3)

    def total_overhead_bytes(self, fixed_gb: float = 0.0) -> float:
        """Total reserved (non-KV, non-weight) overhead in bytes:
        proportional (frac × HBM) + fixed (absolute GiB).

        `fixed_gb` is the RESOLVED fixed overhead (see analyze_capacity's
        precedence: explicit MemoryProfile value > model calibration > 0).
        """
        return (
            self.hbm_capacity_bytes * self.weight_overhead_frac
            + fixed_gb * (1024 ** 3)
        )


# Preset HBM capacities (per GPU, GiB).
MEMORY_PRESETS: dict[str, MemoryProfile] = {
    "A100":       MemoryProfile(hbm_capacity_gb=80.0),    # A100-80GB SXM
    "H100":       MemoryProfile(hbm_capacity_gb=80.0),    # H100-80GB SXM5
    "H200":       MemoryProfile(hbm_capacity_gb=141.0),   # H200 SXM5
    "GB200":      MemoryProfile(hbm_capacity_gb=192.0),   # GB200 (per-GPU half of Grace+B200)
    "GB300":      MemoryProfile(hbm_capacity_gb=288.0),   # GB300 / Blackwell Ultra B300
    "B200":       MemoryProfile(hbm_capacity_gb=192.0),   # B200 SXM
    "Ascend910C": MemoryProfile(hbm_capacity_gb=96.0),    # Huawei Ascend 910C, high-spec HBM3
    "H20":        MemoryProfile(hbm_capacity_gb=96.0),    # H20 96GB HBM3
}


# ============================================================================
#  Capacity report
# ============================================================================


@dataclass
class CapacityReport:
    """Result of an HBM-only capacity analysis at one (gpu, model, S) point.

    All byte fields are raw bytes (multiply by 2**30 -> GiB). All TPOT/time
    fields are seconds. Tput is tokens/sec (per cluster).
    """

    # ---- Inputs (echoed for traceability) ----
    gpu_name: str
    model_name: str
    session_length: int
    parallel: ParallelConfig
    workload: WorkloadConfig

    # ---- Memory accounting (per GPU) ----
    hbm_capacity_bytes: float
    weight_per_gpu_bytes: float
    overhead_bytes: float           # = proportional (frac×HBM) + fixed (GiB)
    hbm_avail_bytes: float          # hbm_capacity - weight - overhead
    session_kv_bytes: float         # KV bytes for ONE session (session_length tokens)
    max_batch_per_gpu: int          # floor(hbm_avail / session_kv_bytes); 0 if infeasible
    feasible: bool                  # max_batch_per_gpu >= 1

    # ---- Performance (from estimate_perf at max_batch_per_gpu) ----
    # Optional: only populated when feasible is True.
    perf_report: Optional[PerfReport] = None

    # ---- Misc ----
    notes: list[str] = field(default_factory=list)

    # ---- Derived helpers ----
    @property
    def kv_fraction(self) -> float:
        """KV (at max batch) / hbm_capacity. Should be ~1 when feasible."""
        if self.hbm_capacity_bytes <= 0:
            return 0.0
        return (self.session_kv_bytes * self.max_batch_per_gpu) / self.hbm_capacity_bytes

    @property
    def weight_fraction(self) -> float:
        if self.hbm_capacity_bytes <= 0:
            return 0.0
        return self.weight_per_gpu_bytes / self.hbm_capacity_bytes

    @property
    def tpot_seconds(self) -> Optional[float]:
        """Decode wall-clock per token, in seconds.

        Delegates to PerfReport.tpot_seconds (= perf_report.total.time_seconds).
        """
        if self.perf_report is None:
            return None
        return self.perf_report.tpot_seconds

    @property
    def cluster_tput_tps(self) -> Optional[float]:
        """Cluster-wide decode throughput, tokens/sec.

        Delegates to PerfReport.cluster_tput_tps. The PerfReport already
        encodes (workload.batch_size, parallel.dp, total.time_seconds) so
        the math lives there; this property exists for backwards-compat
        callers that hold a CapacityReport rather than a PerfReport.

        See PerfReport.cluster_tput_tps for the assumed deployment model.
        """
        if self.perf_report is None:
            return None
        return self.perf_report.cluster_tput_tps


# ============================================================================
#  Public API
# ============================================================================


def analyze_capacity(
    gpu: GPUSpec,
    mem: MemoryProfile,
    model_name: str,
    parallel: ParallelConfig,
    session_length: int,
    *,
    context_length: Optional[int] = None,
    dtype_param: Dtype = Dtype.FP8,
    dtype_kv: Dtype = Dtype.MIXED_BF16_FP8,
    model_overrides: Mapping[str, Any] | None = None,
    new_tokens: int = 1,
    skip_perf: bool = False,
) -> CapacityReport:
    """Compute HBM-only baseline capacity and performance.

    Two distinct sequence-length parameters (do NOT confuse them):

      * `session_length`: capacity dimension. The maximum number of
        tokens the KV cache must hold for one session. Determines
        `session_kv_bytes` and therefore the HBM-bound `max_batch_per_gpu`.

      * `context_length`: runtime attention dimension. The number of
        tokens the model actually attends over in ONE decode step. This
        feeds into `WorkloadConfig.context_length` and is what
        `estimate_perf` uses to compute attn FLOPs / bytes (and therefore
        MFU and TPOT). For sparse-attention models it is also the input
        to indexer-side O(N) costs (main attn is clamped at top_k).

      The two are independent: a session sized for `S=128K` may be at
      runtime step `N=9216` (early in the conversation) or `N=128K`
      (late, near capacity). Defaults to `context_length=session_length`
      (worst-case, i.e. the largest decode step the session will ever do).

    Args:
        gpu: GPU compute / bandwidth spec (from `GPU_PRESETS` or custom).
        mem: HBM capacity profile (from `MEMORY_PRESETS` or custom).
        model_name: Registered model id, e.g. "deepseek-v3.2".
        parallel: TP/EP/PP/DP/SP layout.
        session_length: Per-session KV upper bound (e.g. 128*1024 for 128K).
        context_length: Runtime attention length N. Defaults to
            `session_length` (worst-case decode step). Pass a smaller value
            (e.g. 9216) to model an "average decode step" snapshot.
        dtype_param / dtype_kv: passed to WorkloadConfig (decode defaults).
        model_overrides: passed through to the model estimator.
        new_tokens: forwarded to WorkloadConfig (decode default = 1).
        skip_perf: If True, skip the estimate_perf call (just compute batch
            sizing). Useful for fast sweeps.

    Returns:
        CapacityReport with memory accounting and (optionally) PerfReport.
    """
    model = get_model(model_name)

    # ---- Memory accounting -------------------------------------------------
    weight_bytes = model.weight_bytes_per_gpu(
        parallel, dtype_param, model_overrides=model_overrides
    )

    notes: list[str] = []

    # Resolve the FIXED HBM overhead by precedence:
    #   1. explicit MemoryProfile.overhead_fixed_gb (CLI / config)  — wins
    #   2. model's per-GPU calibration (overhead_fixed_gb_by_gpu)
    #   3. 0.0 (only the proportional term applies)
    if mem.overhead_fixed_gb is not None:
        fixed_gb = float(mem.overhead_fixed_gb)
        fixed_src = "explicit"
    else:
        calibrated = model.resolve_overhead_fixed_gb(gpu.name)
        if calibrated is not None:
            fixed_gb = float(calibrated)
            fixed_src = f"model-calibrated ({model_name}@{gpu.name})"
        else:
            fixed_gb = 0.0
            fixed_src = "none"
            notes.append(
                f"no fixed-overhead calibration for '{model_name}' on "
                f"'{gpu.name}' and none passed explicitly; using 0 "
                f"(only {mem.weight_overhead_frac*100:.0f}% proportional "
                f"overhead). KV-budget / max-batch may be optimistic."
            )

    # Two-part overhead: proportional (frac × HBM) + resolved fixed (GiB).
    overhead_bytes = mem.total_overhead_bytes(fixed_gb)
    hbm_avail = mem.hbm_capacity_bytes - weight_bytes - overhead_bytes

    session_kv = model.session_kv_bytes(
        session_length, parallel, dtype_kv, model_overrides=model_overrides
    )
    if fixed_src.startswith("model-calibrated"):
        notes.append(f"fixed overhead = {fixed_gb:.1f} GiB [{fixed_src}]")
    if weight_bytes <= 0:
        notes.append(
            f"weight_bytes_per_gpu returned 0 for model '{model_name}' "
            "(estimator may not implement capacity API yet)."
        )
    if session_kv <= 0:
        notes.append(
            f"session_kv_bytes returned 0 for model '{model_name}' "
            "(estimator may not implement capacity API yet)."
        )

    # max_batch: floor((HBM - weight - overhead) / session_kv)
    if session_kv > 0 and hbm_avail > 0:
        max_batch = int(hbm_avail // session_kv)
    else:
        max_batch = 0
    feasible = max_batch >= 1

    if not feasible:
        if hbm_avail <= 0:
            notes.append(
                f"infeasible: weight ({weight_bytes/2**30:.2f} GiB) + overhead "
                f"({overhead_bytes/2**30:.2f} GiB) exceeds HBM "
                f"({mem.hbm_capacity_gb:.2f} GiB)."
            )
        elif session_kv > hbm_avail:
            notes.append(
                f"infeasible: one session at S={session_length} needs "
                f"{session_kv/2**30:.2f} GiB KV but only "
                f"{hbm_avail/2**30:.2f} GiB HBM is free for KV."
            )

    # ---- Performance via core perf engine ---------------------------------
    runtime_ctx = (
        context_length if context_length is not None else session_length
    )
    perf_rep: Optional[PerfReport] = None
    if feasible and not skip_perf:
        wl = WorkloadConfig(
            phase=Phase.DECODE,
            batch_size=max_batch,
            context_length=int(runtime_ctx),
            new_tokens=new_tokens,
            dtype_param=dtype_param,
            dtype_kv=dtype_kv,
        )
        perf_rep = estimate_perf(
            gpu=gpu,
            model_name=model_name,
            parallel=parallel,
            workload=wl,
            model_overrides=model_overrides,
        )

    # Build a representative WorkloadConfig for echoing (even when skipped/infeasible).
    echo_wl = WorkloadConfig(
        phase=Phase.DECODE,
        batch_size=max(1, max_batch),
        context_length=int(runtime_ctx),
        new_tokens=new_tokens,
        dtype_param=dtype_param,
        dtype_kv=dtype_kv,
    )

    return CapacityReport(
        gpu_name=gpu.name,
        model_name=model_name,
        session_length=int(session_length),
        parallel=parallel,
        workload=echo_wl,
        hbm_capacity_bytes=mem.hbm_capacity_bytes,
        weight_per_gpu_bytes=float(weight_bytes),
        overhead_bytes=float(overhead_bytes),
        hbm_avail_bytes=float(hbm_avail),
        session_kv_bytes=float(session_kv),
        max_batch_per_gpu=int(max_batch),
        feasible=bool(feasible),
        perf_report=perf_rep,
        notes=notes,
    )


def format_capacity_report(rep: CapacityReport) -> str:
    """Pretty text dump of a single CapacityReport, for terminal use."""
    GiB = 2 ** 30
    lines = [
        f"== Capacity report ==",
        f"  GPU       : {rep.gpu_name}  (HBM={rep.hbm_capacity_bytes/GiB:.1f} GiB)",
        f"  Model     : {rep.model_name}",
        f"  Parallel  : TP={rep.parallel.tp} EP={rep.parallel.ep} "
        f"PP={rep.parallel.pp} DP={rep.parallel.dp}",
        f"  Session   : S={rep.session_length:,} tokens",
        f"",
        f"  Memory (per GPU):",
        f"    weight       = {rep.weight_per_gpu_bytes/GiB:8.2f} GiB "
        f"({rep.weight_fraction*100:5.1f}%)",
        f"    overhead     = {rep.overhead_bytes/GiB:8.2f} GiB",
        f"    HBM avail    = {rep.hbm_avail_bytes/GiB:8.2f} GiB",
        f"    session KV   = {rep.session_kv_bytes/GiB:8.2f} GiB / session",
        f"    max batch    = {rep.max_batch_per_gpu}  (feasible={rep.feasible})",
    ]
    if rep.perf_report is not None:
        r = rep.perf_report
        lines.extend([
            f"",
            f"  Performance:",
            f"    TPOT (total) = {r.total.time_seconds*1000:.2f} ms",
            f"    MFU (total)  = {r.total.mfu*100:.2f}%",
            f"    AI  (total)  = {r.total.ai:.2f} FLOP/Byte",
        ])
        if rep.cluster_tput_tps is not None:
            lines.append(f"    cluster tput = {rep.cluster_tput_tps:,.0f} tok/s")
    if rep.notes:
        lines.append("")
        lines.append("  Notes:")
        for n in rep.notes:
            lines.append(f"    - {n}")
    return "\n".join(lines)


__all__ = [
    "MemoryProfile",
    "MEMORY_PRESETS",
    "CapacityReport",
    "analyze_capacity",
    "format_capacity_report",
]
