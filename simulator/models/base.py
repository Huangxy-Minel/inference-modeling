"""Abstract base class for model cost estimators.

A model estimator answers two questions for a given (parallel, workload):
    1. Attention stage: how many FLOPs, how many HBM bytes?
    2. FFN stage:       how many FLOPs, how many HBM bytes?

The roofline / perf layer (simulator/core.py) doesn't need to know anything about
MLA / DSA / CSA / HCA / SWA / MoE internals; it just consumes (flops, bytes)
per stage.

Adding a new model family:
    1. Create scripts/simulator/models/<my_model>.py
    2. Subclass ModelCostEstimator and implement the two abstract methods.
    3. Decorate with @register_model("my-model-id").
    4. Add an import in scripts/simulator/models/__init__.py so registration runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping


# Forward declarations to avoid circular imports. core.py defines these
# concretely; we use string-based type hints below where possible.
@dataclass
class StageCost:
    """FLOPs and HBM bytes for one logical stage."""

    flops: float
    bytes_: float


# ----- Registry -----------------------------------------------------------

MODEL_REGISTRY: Dict[str, "ModelCostEstimator"] = {}


def register_model(name: str):
    """Class decorator to register a ModelCostEstimator subclass under `name`."""

    def deco(cls):
        if not issubclass(cls, ModelCostEstimator):
            raise TypeError(f"{cls.__name__} must subclass ModelCostEstimator")
        instance = cls()
        MODEL_REGISTRY[name] = instance
        instance._registered_name = name
        return cls

    return deco


# ----- Base class ---------------------------------------------------------


class ModelCostEstimator(ABC):
    """Abstract interface that each model family must implement."""

    # Subclasses can declare a default config dict (hidden, layers, ...) and
    # also accept overrides from a YAML deployment file.
    default_model_config: Mapping[str, Any] = {}

    # Per-model, per-GPU CALIBRATED fixed HBM overhead (GiB) — the constant,
    # non-KV, non-weight HBM that a real serving stack (SGLang / vLLM +
    # DeepEP/NCCL, CUDA graphs, framework runtime caches, ...) reserves and
    # does NOT hand to the KV cache. It is measured from an HBM profiling
    # run of THIS model (see subclass docstrings for the source log) and is
    # keyed by GPU preset name (e.g. "H20", "GB200"). The capacity engine
    # uses it as the default `overhead_fixed_gb` when the caller does not
    # pass an explicit value. Empty = no calibration for this model yet.
    #
    # NOTE: this captures BOTH true fixed infra (active + native/DeepEP +
    # CUDA graph) AND the framework's runtime-dynamic reserve (e.g. PyTorch
    # inactive caching-allocator blocks) that is never released to KV. It is
    # therefore deployment-stack-specific, not a pure hardware number.
    overhead_fixed_gb_by_gpu: Mapping[str, float] = {}

    def __init__(self):
        self._registered_name: str | None = None

    def resolve_overhead_fixed_gb(self, gpu_name: str) -> float | None:
        """Return the calibrated fixed HBM overhead (GiB) for `gpu_name`, or
        None if this model has no calibration for that GPU.

        The capacity engine falls back to None → 0 (plus the proportional
        term) when unset, and any explicit CLI/config value always wins.
        """
        return self.overhead_fixed_gb_by_gpu.get(gpu_name)

    # ---- API used by simulator.core -------------------------------------

    @abstractmethod
    def compute_attn_cost(
        self,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        workload: "WorkloadConfig",          # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        """Return total (FLOPs, HBM bytes) for the attention stage of this step.

        Conventions (must be consistent with compute_ffn_cost):
          * FLOPs counted as multiply-add pairs => 2 per MAC.
          * HBM bytes include weight reads (TP-sharded), KV cache reads/writes,
            and activation reads/writes that hit HBM. Exclude inter-GPU
            comm bytes (NVLink/IB) unless you have a separate budget for it.
          * Aggregate over **all attention layers** of the model.
          * For prefill, count over `workload.new_tokens` tokens; for decode,
            over the per-step cost (typically 1 token).
        """

    @abstractmethod
    def compute_ffn_cost(
        self,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        workload: "WorkloadConfig",          # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        """Return total (FLOPs, HBM bytes) for the FFN stage.

        FFN here covers Dense FFN layers + MoE layers (routed + shared experts),
        sharded by EP/TP as configured in `parallel`.
        """

    # ---- API used by simulator.capacity (decode-only capacity analysis) -------
    #
    # IMPORTANT semantic distinction:
    #   * `WorkloadConfig.context_length` is a *runtime* attention length N
    #     (e.g. 9216 for one specific RL training step). It varies per
    #     forward pass and is consumed by compute_{attn,ffn}_cost above.
    #   * `session_length` (below) is a *deployment* upper bound (e.g. 128K /
    #     1M / 4M) — the maximum KV cache budget per session. It is a
    #     constant for the capacity-analysis sweep and is *independent* from
    #     the runtime context_length.
    #
    # These two methods are NOT abstract: they default to 0 so existing model
    # stubs (e.g. DeepSeek V4) keep loading. The capacity engine will print a
    # warning when a model returns 0.

    def session_kv_bytes(
        self,
        session_length: int,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        dtype_kv: "Dtype",                   # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """KV cache bytes consumed by ONE decode session of length `session_length`.

        The returned value is **already**:
          - summed across all attention layers,
          - sharded by the model's own parallel strategy (TP/EP/SP),
          - measured per-GPU.

        The capacity engine treats this number as a black box. Override in
        each concrete estimator. Default returns 0.0.
        """
        return 0.0

    def indexer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        dtype_kv: "Dtype",                   # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Sparse-attention indexer K-cache bytes per session, summed across
        layers that have an indexer.

        Used by DRAM Pooling capacity analysis to size the indexer working
        set (which by default must be HBM-resident). Default returns 0.0:
            * V3 base (no DSA)         → 0
            * V4 HCA / SWA layers      → 0
            * V3.2, V4 CSA layers      → > 0 (override to compute)
        """
        return 0.0

    def cold_layer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        dtype_kv: "Dtype",                   # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """KV bytes that must be fetched from DRAM per missed layer per session.

        For sparse attention this is *much* smaller than `session_kv_bytes /
        num_layers`: a CSA / DSA layer only needs to fetch the top-k entries,
        not all N tokens of that layer. For dense attention (V3 base) it
        equals the full per-layer KV.

        Default falls back to `session_kv_bytes / num_layers` (one full
        layer's KV), which is the conservative upper bound used when the
        model doesn't override.
        """
        sess = self.session_kv_bytes(
            session_length, parallel, dtype_kv, model_overrides
        )
        cfg = self.merged_config(model_overrides)
        L = max(1, int(cfg.get("num_layers", 1)))
        return sess / L

    def weight_bytes_per_gpu(
        self,
        parallel: "ParallelConfig",          # type: ignore[name-defined]
        dtype_param: "Dtype",                # type: ignore[name-defined]
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Total static weight bytes resident on ONE GPU (attn + FFN).

        Already TP/EP-sharded per the given `parallel` config. Used by the
        capacity engine as `weight_bytes_per_gpu` to compute available HBM
        budget for KV cache:

            hbm_avail = hbm_capacity - weight_bytes_per_gpu

        Default returns 0.0 (override in concrete estimators).
        """
        return 0.0

    # ---- Helpers --------------------------------------------------------

    def merged_config(self, overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
        """Return default config merged with optional per-deployment overrides."""
        cfg = dict(self.default_model_config)
        if overrides:
            cfg.update(overrides)
        return cfg

    def __repr__(self) -> str:
        n = self._registered_name or self.__class__.__name__
        return f"<ModelCostEstimator {n}>"
