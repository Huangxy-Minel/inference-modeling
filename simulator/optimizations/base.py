"""Abstract interface for capacity optimizations.

A *capacity optimization* takes an HBM-only baseline (`CapacityReport`) and
returns an `OptimizedCapacityReport` describing what changes when the
optimization is applied:

    - new max_batch_per_gpu        (typically larger)
    - new TPOT / MFU / throughput  (typically with some penalty)
    - extra metadata               (bandwidth saturation, eviction stats, ...)

Examples of optimizations:
    * DRAM Pooling          - extend KV capacity into host DRAM (this PR)
    * Prefetch / Overlap    - hide DRAM access latency under compute
    * KV offloading         - tier KV across HBM/DRAM/SSD with eviction
    * Compression           - reduce session_kv_bytes via on-the-fly quant

Each optimization lives in its own module (e.g. `dram_pooling.py`) and
subclasses `CapacityOptimization`. They compose: `optA.apply(baseline)` then
`optB.apply(...)` is allowed (with the obvious caveat that the second
optimization sees an `OptimizedCapacityReport`, not a raw `CapacityReport`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..capacity import CapacityReport, MemoryProfile
from ..core import GPUSpec, PerfReport, ParallelConfig


@dataclass
class OptimizedCapacityReport:
    """Result of applying one (or more) capacity optimizations to a baseline.

    The baseline is preserved as `baseline` so callers can do side-by-side
    comparisons. All bytes fields are raw bytes; all time fields are seconds.
    """

    # ---- Provenance ----
    optimization_name: str
    baseline: CapacityReport

    # ---- New batch sizing ----
    max_batch_per_gpu: int
    feasible: bool

    # ---- Memory accounting (per GPU, for the optimization tier) ----
    extra_capacity_bytes: float       # how much extra room the optimization buys

    # ---- Performance ----
    # `perf_report` is from estimate_perf(batch=new_max_batch), i.e. compute /
    # HBM-side performance assuming all KV fits in HBM-bandwidth-class memory.
    # `penalty_seconds` is the per-token additional latency from the slower
    # tier (e.g. DRAM) accesses. Final TPOT = perf_report.total.time_seconds +
    # penalty_seconds.
    perf_report: Optional[PerfReport] = None
    penalty_seconds: float = 0.0

    # ---- Capacity diagnostics (DRAM Pool, may be 0 for other optimizations) ----
    # Which budget binds the chosen batch:
    #   Scenario 1 (sparse_on_demand):
    #     "DRAM-cap"          — total KV capacity (HBM_avail + DRAM)
    #     "indexer-cap"       — indexer K-cache HBM budget (when indexer in HBM)
    #     "user-override"     — explicit batch_size_override consumed
    #   Scenario 2 (shared_prefix):
    #     "HBM-cap"           — HBM cannot fit even one session's full unique
    #     "DRAM-prefix-cap"   — single shared prefix exceeds DRAM capacity
    #     "DRAM-spill-cap"    — prefix fits but spilled-unique exceeds remaining DRAM
    #     "model-unbounded"   — α=1.0 corner: unique vanishes, bs not bound by HBM
    #     "user-override"     — explicit batch_size_override consumed
    bs_bound_by: str = "unknown"
    # Indexer K-cache that must be HBM-resident (0 if --indexer-in-dram).
    indexer_kv_bytes_total: float = 0.0
    # Hot-cache fittability: how many full layers of post-batch hot KV
    # (top-k × v_token × bs × num_layers, but expressed per-layer) fit in
    # the HBM left after the optimization's other reservations. < 1.0 means
    # we can't even cache one layer's hot working set; user must accept
    # hit_rate = 0 in that regime.
    hot_cache_layers_fittable: float = 0.0
    # Prefetch overlap of the LIGHT path:
    #   * Scenario 1 (indexer_in_dram=True): indexer K-cache prefetch overlap.
    #   * Scenario 2 (shared_prefix): prefix-only prefetch overlap (the L-k
    #     "light" layers — those whose unique KV is HBM-resident — only need
    #     to prefetch the single shared prefix per layer).
    # Equals min(1, t_layer / t_prefetch_per_layer): fraction of fetch time
    # hidden under the previous layer's (attn + ffn) compute window.
    # None = N/A (indexer in HBM, or no prefix in Scenario 2).
    indexer_prefetch_overlap_effective: Optional[float] = None
    # ---- Scenario 2 (shared_prefix) layer-spill diagnostics ----
    # When the per-batch unique KV does not fit in HBM, the tail (last) k
    # layers' unique KV are spilled to DRAM and prefetched layer-stride
    # together with the shared prefix. These two fields surface the spill
    # state to callers / report renderers; both are None in Scenario 1
    # and in Scenario 2 with k=0 (no spill).
    spilled_layers_count: Optional[int] = None
    # Heavy-path prefetch overlap: prefetch volume per spilled layer is
    # (prefix_per_layer + per_layer_unique × bs); typically << 1 once
    # spill kicks in (the bs multiplier dominates).
    spilled_unique_overlap_effective: Optional[float] = None

    # ---- DRAM pool occupancy (actually-used, per GPU) ----
    # How much of the DRAM pool this optimization actually consumes at the
    # chosen batch (NOT the pool size `extra_capacity_bytes`, which is the
    # capacity ceiling). 0 for optimizations that don't use DRAM.
    #   Scenario 1 (sparse_on_demand): bs × (cold-tail main KV + indexer if
    #     indexer_in_dram).
    #   Scenario 2 (shared_prefix): single shared prefix + spilled tail-k
    #     unique KV × bs.
    dram_used_bytes: float = 0.0

    # ---- Misc ----
    notes: list[str] = field(default_factory=list)

    # ---- Derived helpers ----
    @property
    def tpot_seconds(self) -> Optional[float]:
        """Effective TPOT (decode wall-clock per token) including DRAM penalty.

        = PerfReport.tpot_seconds + penalty_seconds

        The penalty is layered on top of the bare GPU stage time because it
        represents wait time on a slower memory tier (DRAM) that PerfReport
        knows nothing about.
        """
        if self.perf_report is None:
            return None
        return self.perf_report.tpot_seconds + self.penalty_seconds

    @property
    def cluster_tput_tps(self) -> Optional[float]:
        """Cluster-wide decode throughput, tokens/sec, with DRAM penalty.

        Same math as PerfReport.cluster_tput_tps (samples_in_flight / TPOT)
        but uses (a) the optimization's max_batch_per_gpu (which differs from
        PerfReport.workload.batch_size whenever the optimization grew the
        batch) and (b) the penalty-inclusive tpot_seconds above. We can't
        delegate to PerfReport here because both inputs change.
        """
        if self.perf_report is None or self.tpot_seconds is None or self.tpot_seconds <= 0:
            return None
        bs_per_gpu = self.max_batch_per_gpu
        samples_in_flight = bs_per_gpu * max(1, self.baseline.parallel.dp)
        return samples_in_flight / self.tpot_seconds

    @property
    def speedup_vs_baseline(self) -> Optional[float]:
        """Cluster throughput speedup factor relative to baseline (>1 = better)."""
        b_tput = self.baseline.cluster_tput_tps
        o_tput = self.cluster_tput_tps
        if not b_tput or not o_tput or b_tput <= 0:
            return None
        return o_tput / b_tput


class CapacityOptimization(ABC):
    """One pluggable optimization that relaxes the HBM-only constraint."""

    name: str = "<unnamed>"

    @abstractmethod
    def apply(
        self,
        baseline: CapacityReport,
        *,
        gpu: GPUSpec,
        mem: MemoryProfile,
        parallel: ParallelConfig,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> OptimizedCapacityReport:
        """Run the optimization on top of `baseline`.

        Implementations must:
          * preserve `baseline` unchanged (return a new report);
          * recompute `max_batch_per_gpu` per the optimization's accounting;
          * call `estimate_perf` with the new batch to get HBM-side performance;
          * compute `penalty_seconds` for additional slow-tier access cost.
        """


__all__ = [
    "CapacityOptimization",
    "OptimizedCapacityReport",
]
