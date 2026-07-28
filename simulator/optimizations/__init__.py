"""Capacity optimizations for decode-phase serving.

Each optimization sub-module contributes a `CapacityOptimization` subclass
that consumes a baseline `CapacityReport` (from `simulator.capacity`) and returns
an `OptimizedCapacityReport` describing the new batch sizing, performance,
and slow-tier penalty.

Public API:
    from simulator.optimizations import (
        CapacityOptimization,
        OptimizedCapacityReport,
        DramPoolingConfig,
        DramPoolingOptimization,
    )

Roadmap (TODO):
    - prefetch.py / overlap.py : hide DRAM penalty under compute
    - kv_offload.py            : tier KV across HBM/DRAM/SSD with eviction
    - kv_compression.py        : on-the-fly quant / sparsification
"""

from .base import CapacityOptimization, OptimizedCapacityReport
from .dram_pooling import DramPoolingConfig, DramPoolingOptimization

__all__ = [
    "CapacityOptimization",
    "OptimizedCapacityReport",
    "DramPoolingConfig",
    "DramPoolingOptimization",
]
