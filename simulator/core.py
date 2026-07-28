"""Core perf machinery: GPU/parallel/workload dataclasses, roofline,
orchestrator, and YAML deployment loader.

This module is model-agnostic. Model-specific cost models live in
simulator/models/.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .models import get_model
from .models.base import StageCost


# ============================================================================
#  1. Roofline core (kept identical to scripts/roofline.py::roofline_perf so
#     the two modules stay consistent without forcing a matplotlib dep).
# ============================================================================


def roofline_perf(ai: np.ndarray, peak_tflops: float, bandwidth_gbs: float) -> np.ndarray:
    """Attainable performance (TFLOP/s) given Arithmetic Intensity (FLOP/Byte)."""
    memory_limited_tflops = (bandwidth_gbs * ai) / 1000.0
    return np.minimum(memory_limited_tflops, peak_tflops)


# ============================================================================
#  2. Dataclasses: GPU / Parallel / Workload
# ============================================================================


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Dtype(str, Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"
    MIXED_BF16_FP8 = "mixed_bf16_fp8"


def dtype_bytes(dtype: Dtype) -> float:
    return {
        Dtype.BF16: 2.0,
        Dtype.FP8: 1.0,
        Dtype.FP4: 0.5,
        Dtype.MIXED_BF16_FP8: 1.0,   # rough blend; better to split when needed
    }[dtype]


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_tflops: float
    bandwidth_gbs: float


GPU_PRESETS: Dict[str, GPUSpec] = {
    "A100":       GPUSpec("A100",       peak_tflops=312.0,  bandwidth_gbs=2039.0),  # FP16 dense
    "H100":       GPUSpec("H100",       peak_tflops=1979.0, bandwidth_gbs=3350.0),  # FP8 dense
    "H200":       GPUSpec("H200",       peak_tflops=1979.0, bandwidth_gbs=4800.0),  # FP8 dense
    "GB200":      GPUSpec("GB200",      peak_tflops=5000.0, bandwidth_gbs=8000.0),  # FP8 dense, per GPU
    "GB300":      GPUSpec("GB300",      peak_tflops=7000.0, bandwidth_gbs=12000.0), # FP8, Blackwell Ultra B300
    "Ascend910C": GPUSpec("Ascend910C", peak_tflops=486.0,  bandwidth_gbs=4000.0),  # Huawei Ascend 910C
    "H20":        GPUSpec("H20",        peak_tflops=300.0,  bandwidth_gbs=4000.0),  # FP8 dense
}


@dataclass(frozen=True)
class ParallelConfig:
    """Decode-phase parallel config under V3/V3.2-style **switching parallel**.

    Hard constraint (decode-only, enforced in __post_init__):
        DP × TP == EP  (and EP >= 1)

    Physical interpretation:
        The same physical GPUs run TP×DP for attention and reshape to EP
        for FFN. Therefore the number of physical GPUs is simply:
            world_size = EP × PP   (= DP × TP × PP)
        rather than TP × EP × DP × PP — DP and TP are NOT independent
        dimensions on top of EP, they ARE the EP grid factored differently.

    Standard 3D parallelism (where TP × EP × DP × PP would all be
    independent) is NOT supported by this config since we focus on
    decode here. Add a separate ParallelConfig variant if a future
    workload truly needs independent dims.
    """

    tp: int = 1
    ep: int = 1
    pp: int = 1
    dp: int = 1
    sp: int = 1

    def __post_init__(self):
        # Validate the switching-parallel constraint.
        for name, val in [("tp", self.tp), ("ep", self.ep),
                          ("pp", self.pp), ("dp", self.dp), ("sp", self.sp)]:
            if val < 1:
                raise ValueError(
                    f"ParallelConfig.{name} must be >= 1, got {val}"
                )
        if self.dp * self.tp != self.ep:
            raise ValueError(
                f"ParallelConfig violates DP × TP = EP constraint "
                f"(decode-only switching parallel): "
                f"DP={self.dp} × TP={self.tp} = {self.dp * self.tp}, "
                f"but EP={self.ep}. The same GPUs must run TP×DP for "
                f"attention and reshape to EP for FFN. "
                f"Adjust DP, TP, or EP so that DP × TP == EP."
            )

    @property
    def world_size(self) -> int:
        """Physical GPU count = EP × PP (= DP × TP × PP under the constraint)."""
        return self.ep * self.pp


@dataclass(frozen=True)
class WorkloadConfig:
    phase: Phase
    batch_size: int
    context_length: int
    new_tokens: int = 1
    dtype_param: Dtype = Dtype.FP8
    dtype_kv: Dtype = Dtype.MIXED_BF16_FP8


# ============================================================================
#  3. Stage report and orchestrator
# ============================================================================


@dataclass
class StageReport:
    name: str
    flops: float
    bytes_: float
    ai: float
    attainable_tflops: float
    peak_tflops: float
    mfu: float
    is_mem_bound: bool
    time_seconds: float


def evaluate_stage(name: str, cost: StageCost, gpu: GPUSpec) -> StageReport:
    """Apply Roofline to a stage cost and return MFU + wall-clock."""
    if cost.flops <= 0 and cost.bytes_ <= 0:
        return StageReport(
            name=name, flops=0.0, bytes_=0.0, ai=0.0,
            attainable_tflops=0.0, peak_tflops=gpu.peak_tflops,
            mfu=0.0, is_mem_bound=True, time_seconds=0.0,
        )

    ai = cost.flops / cost.bytes_ if cost.bytes_ > 0 else float("inf")
    attainable_tflops = float(
        roofline_perf(np.array([ai]), gpu.peak_tflops, gpu.bandwidth_gbs)[0]
    )
    peak_tflops = gpu.peak_tflops
    mfu = attainable_tflops / peak_tflops if peak_tflops > 0 else 0.0

    boundary_ai = (peak_tflops * 1000.0) / gpu.bandwidth_gbs
    is_mem_bound = ai < boundary_ai
    if is_mem_bound:
        time_s = cost.bytes_ / (gpu.bandwidth_gbs * 1e9)
    else:
        time_s = cost.flops / (peak_tflops * 1e12)

    return StageReport(
        name=name,
        flops=cost.flops,
        bytes_=cost.bytes_,
        ai=ai,
        attainable_tflops=attainable_tflops,
        peak_tflops=peak_tflops,
        mfu=mfu,
        is_mem_bound=is_mem_bound,
        time_seconds=time_s,
    )


@dataclass
class PerfReport:
    """Performance report for one decode/prefill step on the given GPU.

    Captures FLOPs, bytes, AI, attainable TFLOPs, time, and MFU at three
    granularities: attn stage, ffn stage, and the (attn + ffn) total.

    Single-step view: this is one "step" through the model — for decode,
    that means generating ONE token. Multi-step / cluster-level metrics
    are derived properties below.
    """

    gpu: str
    model: str
    parallel: ParallelConfig
    workload: WorkloadConfig
    attn: StageReport
    ffn: StageReport
    total: StageReport

    # ---- Derived helpers ------------------------------------------------

    @property
    def tpot_seconds(self) -> float:
        """Time Per Output Token, in seconds.

        For decode (one step generates one token) this is identically the
        total stage wall-clock. For prefill it's the time for the full
        prompt; callers using PerfReport for prefill should reinterpret.
        """
        return self.total.time_seconds

    @property
    def cluster_tput_tps(self) -> Optional[float]:
        """Cluster-wide decode throughput, tokens/sec.

            cluster_tput = samples_in_flight / TPOT
            samples_in_flight = batch_per_GPU × DP

        Assumes the standard switching-parallel deployment (TP×EP×DP=EP×PP
        physical GPUs, every DP replica serves a disjoint sample set, and
        each sample produces one token per TPOT). Special deployments
        (P-D split, asymmetric EP, etc.) should bypass this property and
        compute their own throughput at the script layer.

        Returns None if TPOT is undefined (zero-flop step).
        """
        if self.tpot_seconds <= 0:
            return None
        bs = self.workload.batch_size
        dp = max(1, self.parallel.dp)
        return (bs * dp) / self.tpot_seconds


def estimate_perf(
    gpu: GPUSpec,
    model_name: str,
    parallel: ParallelConfig,
    workload: WorkloadConfig,
    model_overrides: Mapping[str, Any] | None = None,
) -> PerfReport:
    """End-to-end performance estimator. Looks up the model estimator by
    name and composes per-stage roofline reports into a PerfReport.
    """
    estimator = get_model(model_name)
    attn_cost = estimator.compute_attn_cost(parallel, workload, model_overrides)
    ffn_cost  = estimator.compute_ffn_cost(parallel, workload, model_overrides)

    attn_rep = evaluate_stage("attention", attn_cost, gpu)
    ffn_rep  = evaluate_stage("ffn",       ffn_cost,  gpu)

    total_cost = StageCost(
        flops=attn_cost.flops + ffn_cost.flops,
        bytes_=attn_cost.bytes_ + ffn_cost.bytes_,
    )
    total_time = attn_rep.time_seconds + ffn_rep.time_seconds
    achieved_tflops = (total_cost.flops / total_time / 1e12) if total_time > 0 else 0.0
    total_ai = total_cost.flops / total_cost.bytes_ if total_cost.bytes_ > 0 else float("inf")
    total_rep = StageReport(
        name="total",
        flops=total_cost.flops,
        bytes_=total_cost.bytes_,
        ai=total_ai,
        attainable_tflops=achieved_tflops,
        peak_tflops=gpu.peak_tflops,
        mfu=(achieved_tflops / gpu.peak_tflops) if gpu.peak_tflops > 0 else 0.0,
        is_mem_bound=(total_ai < (gpu.peak_tflops * 1000.0) / gpu.bandwidth_gbs),
        time_seconds=total_time,
    )

    return PerfReport(
        gpu=gpu.name,
        model=model_name,
        parallel=parallel,
        workload=workload,
        attn=attn_rep,
        ffn=ffn_rep,
        total=total_rep,
    )


# ============================================================================
#  4. Pretty-print
# ============================================================================


def format_report(r: PerfReport) -> str:
    def fmt_stage(s: StageReport) -> str:
        return (
            f"  {s.name:<10} "
            f"FLOPs={s.flops:.3e}  "
            f"Bytes={s.bytes_:.3e}  "
            f"AI={s.ai:8.2f} F/B  "
            f"attain={s.attainable_tflops:7.1f}/{s.peak_tflops:.0f} TF/s  "
            f"MFU={s.mfu*100:5.1f}%  "
            f"{'mem' if s.is_mem_bound else 'cmp'}-bound  "
            f"t={s.time_seconds*1e3:.3f} ms"
        )

    head = (
        f"=== Perf report ===\n"
        f"  GPU       : {r.gpu}\n"
        f"  Model     : {r.model}\n"
        f"  Parallel  : TP={r.parallel.tp} EP={r.parallel.ep} "
        f"PP={r.parallel.pp} DP={r.parallel.dp}\n"
        f"  Workload  : phase={r.workload.phase.value} "
        f"batch={r.workload.batch_size} ctx={r.workload.context_length} "
        f"new_tokens={r.workload.new_tokens}\n"
    )
    return head + "\n".join(fmt_stage(s) for s in [r.attn, r.ffn, r.total])


# ============================================================================
#  5. YAML deployment config loader
# ============================================================================


def _require(d: Mapping[str, Any], *keys: str) -> Any:
    """Walk a nested dict; raise KeyError with a useful message if missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            path = ".".join(keys)
            raise KeyError(f"deployment config missing '{path}'")
        cur = cur[k]
    return cur


def load_deployment(path: str | Path) -> Dict[str, Any]:
    """Load a YAML deployment file. Returns a dict with parsed objects.

    Expected schema (see simulator/configs/*.yaml for examples):

        gpu: GB200            # preset name OR an inline dict {name, peak_tflops, bandwidth_gbs}
        model: deepseek-v3.2  # registered model name
        model_overrides: {...}# optional per-deployment overrides into model config
        parallel:
          tp: 4
          ep: 32
          pp: 1
          dp: 1
        workload:
          phase: decode       # 'prefill' or 'decode'
          batch_size: 64
          context_length: 9216
          new_tokens: 1
          dtype_param: fp8
          dtype_kv: mixed_bf16_fp8
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PyYAML required for YAML configs. Install with: pip install pyyaml"
        ) from e

    raw = yaml.safe_load(Path(path).read_text())

    # GPU
    gpu_field = _require(raw, "gpu")
    if isinstance(gpu_field, str):
        if gpu_field not in GPU_PRESETS:
            avail = ", ".join(GPU_PRESETS.keys())
            raise KeyError(f"Unknown GPU preset '{gpu_field}'. Available: {avail}")
        gpu = GPU_PRESETS[gpu_field]
    elif isinstance(gpu_field, Mapping):
        gpu = GPUSpec(
            name=gpu_field["name"],
            peak_tflops=float(gpu_field["peak_tflops"]),
            bandwidth_gbs=float(gpu_field["bandwidth_gbs"]),
        )
    else:
        raise TypeError("gpu must be a string preset name or a dict")

    # Model
    model_name = _require(raw, "model")
    model_overrides = raw.get("model_overrides") or {}

    # Parallel
    p = _require(raw, "parallel")
    parallel = ParallelConfig(
        tp=int(p.get("tp", 1)),
        ep=int(p.get("ep", 1)),
        pp=int(p.get("pp", 1)),
        dp=int(p.get("dp", 1)),
        sp=int(p.get("sp", 1)),
    )

    # Workload
    w = _require(raw, "workload")
    phase = Phase(w["phase"])
    new_tokens = int(w.get("new_tokens", w["context_length"] if phase == Phase.PREFILL else 1))
    workload = WorkloadConfig(
        phase=phase,
        batch_size=int(w["batch_size"]),
        context_length=int(w["context_length"]),
        new_tokens=new_tokens,
        dtype_param=Dtype(w.get("dtype_param", Dtype.FP8.value)),
        dtype_kv=Dtype(w.get("dtype_kv", Dtype.MIXED_BF16_FP8.value)),
    )

    return {
        "gpu": gpu,
        "model": model_name,
        "model_overrides": model_overrides,
        "parallel": parallel,
        "workload": workload,
    }


def run_from_yaml(path: str | Path) -> PerfReport:
    """Convenience: load a deployment YAML and return its PerfReport."""
    cfg = load_deployment(path)
    return estimate_perf(
        gpu=cfg["gpu"],
        model_name=cfg["model"],
        parallel=cfg["parallel"],
        workload=cfg["workload"],
        model_overrides=cfg["model_overrides"],
    )


# ============================================================================
#  6. CLI
# ============================================================================


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Compute LLM MFU under a given deployment (pure roofline; "
            "no communication / overlap modeling). "
            "Reports FLOPs, HBM bytes, AI, attainable TFLOPS, MFU, "
            "memory/compute bound, and stage wall-clock (= TPOT for total)."
        ),
    )
    p.add_argument("--config", type=str,
                   help="Path to a YAML deployment config (overrides inline flags).")
    # Inline override (single-shot use, no YAML needed):
    p.add_argument("--gpu",   default="GB200", choices=list(GPU_PRESETS.keys()),
                   help="GPU preset (default: GB200).")
    p.add_argument("--model", default="deepseek-v3.2",
                   help="Registered model name (default: deepseek-v3.2).")
    p.add_argument("--phase", default="decode", choices=[ph.value for ph in Phase],
                   help="prefill or decode (default: decode).")
    p.add_argument("--tp",    type=int, default=4,
                   help="Tensor parallel size (attn heads sharding).")
    p.add_argument("--ep",    type=int, default=32,
                   help="Expert parallel size (MoE experts sharding).")
    p.add_argument("--pp",    type=int, default=1,
                   help="Pipeline parallel size (currently not modeled).")
    p.add_argument("--dp",    type=int, default=8,
                   help="Data parallel size (replicates attn; affects cluster tput only).")
    p.add_argument("--batch", type=int, default=64,
                   help="Per-DP batch size (not cluster-wide).")
    p.add_argument("--ctx",   type=int, default=9216,
                   help="Context length N (in tokens).")
    p.add_argument("--new-tokens", type=int, default=1,
                   help="New tokens to generate this step "
                        "(decode default 1; prefill auto-sets to ctx).")
    args = p.parse_args()

    if args.config:
        report = run_from_yaml(args.config)
    else:
        gpu = GPU_PRESETS[args.gpu]
        parallel = ParallelConfig(tp=args.tp, ep=args.ep, pp=args.pp, dp=args.dp)
        new_toks = args.ctx if args.phase == Phase.PREFILL.value else args.new_tokens
        workload = WorkloadConfig(
            phase=Phase(args.phase),
            batch_size=args.batch,
            context_length=args.ctx,
            new_tokens=new_toks,
        )
        report = estimate_perf(
            gpu=gpu,
            model_name=args.model,
            parallel=parallel,
            workload=workload,
        )

    print(format_report(report))


if __name__ == "__main__":
    main()
