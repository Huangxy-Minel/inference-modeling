"""Inference-modeling simulator.

Top-level package. Public API:

    # Core perf engine (Roofline + per-stage breakdown)
    from simulator import (
        GPUSpec, ParallelConfig, WorkloadConfig, Phase, Dtype,
        StageCost, StageReport, PerfReport,
        estimate_perf, format_report,
        load_deployment, run_from_yaml,
    )

    # Capacity analysis (HBM-only baseline)
    from simulator import (
        MemoryProfile, MEMORY_PRESETS,
        CapacityReport, analyze_capacity, format_capacity_report,
    )

    # Optimizations (pluggable)
    from simulator import (
        CapacityOptimization, OptimizedCapacityReport,
        DramPoolingConfig, DramPoolingOptimization,
    )
"""

from .core import (
    Dtype,
    GPUSpec,
    GPU_PRESETS,
    ParallelConfig,
    PerfReport,
    Phase,
    StageCost,
    StageReport,
    WorkloadConfig,
    estimate_perf,
    evaluate_stage,
    format_report,
    load_deployment,
    roofline_perf,
    run_from_yaml,
)
from .capacity import (
    CapacityReport,
    MEMORY_PRESETS,
    MemoryProfile,
    analyze_capacity,
    format_capacity_report,
)
from .optimizations import (
    CapacityOptimization,
    DramPoolingConfig,
    DramPoolingOptimization,
    OptimizedCapacityReport,
)

__all__ = [
    # core
    "Dtype",
    "GPUSpec",
    "GPU_PRESETS",
    "PerfReport",
    "ParallelConfig",
    "Phase",
    "StageCost",
    "StageReport",
    "WorkloadConfig",
    "estimate_perf",
    "evaluate_stage",
    "format_report",
    "load_deployment",
    "roofline_perf",
    "run_from_yaml",
    # capacity
    "CapacityReport",
    "MEMORY_PRESETS",
    "MemoryProfile",
    "analyze_capacity",
    "format_capacity_report",
    # optimizations
    "CapacityOptimization",
    "DramPoolingConfig",
    "DramPoolingOptimization",
    "OptimizedCapacityReport",
]
