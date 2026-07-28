"""DRAM Pooling: extend KV capacity into host / pooled DRAM.

Two scenarios are implemented in a single `DramPoolingOptimization`
class, selected via `DramPoolingConfig.mode`:

    Scenario 1 — Sparse On-Demand  (mode="sparse_on_demand", default)
    -----------------------------------------------------------------
    For sparse-attention models (DeepSeek DSA, GLM, NSA, ...), each decode
    layer only needs *part* of the KV cache. The HBM keeps a hot working
    set; cold KV pages live in DRAM and are pulled on demand once the
    indexer (or top-k selector) decides which pages this layer needs.

    Critically, **this fetch CANNOT be prefetched**: top-k indices depend
    on the indexer output of the same layer, so the fetch is on the
    critical path. The unit of stall is one *layer*: if any KV page in a
    layer misses, that layer must wait for DRAM.

    Penalty model (per decode step):

        miss_layers     = num_layers * (1 - hit_rate)
        cold_layer_kv   = (session_kv_bytes / num_layers)
                          * n_missing_sessions
                          * cold_layer_correction
        fetch_per_miss  = cold_layer_kv / dram_interconnect_bandwidth
        penalty         = miss_layers * fetch_per_miss

    `hit_rate` is the per-layer hit probability supplied by the user.
    `n_missing_sessions` defaults to `batch` (pessimistic upper bound:
    the entire batch misses synchronously on a missed layer); set
    explicitly to 1 for the optimistic lower bound (de-correlated
    per-session miss timing). Note that `(1-hit_rate) × n_missing` is
    the effective fraction of cold traffic per layer, so the two knobs
    interact multiplicatively — keep this in mind when sweeping.
    `cold_layer_correction` defaults to 1.0 (worst case: a missed layer
    re-reads all of that session's KV at that layer).

    Naive degenerates from this model: `hit_rate=0` + `batch_size=auto-max`
    means every layer must stream from DRAM, with the entire batch's
    cold KV pulled per missed layer (the new default). Pass
    `n_missing_sessions=1` to recover the optimistic lower-bound
    behaviour from earlier revisions of this code.

    Optionally with `indexer_in_dram=True`, the indexer K-cache is moved
    out of HBM and prefetched per layer with a roofline overlap formula
    identical in shape to Scenario 2's prefix prefetch, but the volume
    DOES scale with bs (per-session indexer state).

    Scenario 2 — Shared-Prefix Prefetch  (mode="shared_prefix")
    ----------------------------------------------------------
    Cross-session shared prefix (system prompt, tool catalog, RAG
    context, ...) lives as a SINGLE shared copy in DRAM and IS
    prefetchable per layer (no indexer dependency). All `bs` sessions
    consume the same prefix, so the DMA volume does NOT scale with bs.

    Capacity (this baseline; see Future Work below):

        α = prefix_share_frac
        # DRAM holds a single shared copy of the prefix:
        DRAM-cap (binary):  α × session_kv ≤ DRAM_capacity
        # HBM holds the per-session unique part:
        HBM-cap:  bs × (1 - α) × session_kv ≤ HBM_avail
        ⇒ bs_max  =  HBM_avail / ((1 - α) × session_kv)

    Penalty (roofline prefetch overlap, same shape as Scenario 1's
    indexer-in-DRAM term but **without** the bs multiplier):

        t_layer    = (mfu.attn.time + mfu.ffn.time) / num_layers
        prefix_per_layer = α × session_kv / num_layers
        t_prefetch = prefix_per_layer / dram_bw
        overlap    = min(1, t_layer / t_prefetch)
        penalty    = (α × session_kv / dram_bw) × (1 - overlap)

    The sparse-attention knobs (`kv_cache_hit_rate`, `n_missing_sessions`,
    `indexer_in_dram`, `tokens_per_miss_layer`) are ignored in this mode:
    the prefix is dense latent KV with no top-k semantics.

    Layer-spill extension (IMPLEMENTED — production-flow accurate)
    --------------------------------------------------------------
    When `bs × (1-α) × session_kv > HBM_avail`, the per-batch unique KV
    no longer fits in HBM. Following the PD-disagg production flow
    (prefill streams KV layer-by-layer; KV pool page-table is
    layer × token), the spill happens along the LAYER axis: HBM keeps
    the first `L - k` layers' unique KV, the tail `k` layers' unique KV
    moves to DRAM (alongside the single shared prefix copy).

        k(bs) = ceil((bs × (1-α) × S_kv - HBM_avail) / (bs × PLU))   ∈ [0, L]
        bs_max = (HBM_avail + (DRAM - α × S_kv)) / ((1 - α) × S_kv)
                  (then page-aligned: back off bs by 1 if integer-k
                   exceeds DRAM at the analytic bound)

    Prefetch volume splits into two paths (each layer has its own
    independent next-layer prefetch window of t_layer):

        light layers (L - k):  prefetch = α × S_kv / L           # prefix only
        heavy layers (k):      prefetch = α × S_kv / L + bs × PLU
                                                       # prefix + per-layer unique

        penalty = (L-k) × t_pref_light × (1 - overlap_light)
                + k     × t_pref_heavy × (1 - overlap_heavy)

    The heavy-path overlap is typically the bottleneck (the bs
    multiplier dominates) and is what the report highlights via
    `spilled_unique_overlap_effective`; `spilled_layers_count` exposes
    `k`. When `k=0` the path collapses to the original prefix-only
    formula and these two fields are None.

    `bs_bound_by` distinguishes the four feasibility regimes:
    `DRAM-prefix-cap` / `DRAM-spill-cap` / `HBM-cap` / `model-unbounded`
    (plus `user-override`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Union

from ..capacity import CapacityReport, MemoryProfile
from ..core import GPUSpec, ParallelConfig, Phase, WorkloadConfig, estimate_perf
from ..models import get_model
from .base import CapacityOptimization, OptimizedCapacityReport


# ============================================================================
#  Unified DRAM Pooling Config (covers both scenarios via `mode` flag)
# ============================================================================


@dataclass(frozen=True)
class DramPoolingConfig:
    """Per-GPU DRAM pool configuration. Covers both DRAM-pooling scenarios.

    The active scenario is selected via the `mode` flag:
      * `"sparse_on_demand"` (default): Scenario 1 — sparse-attention,
        per-layer on-demand fetch. Uses `kv_cache_hit_rate`,
        `n_missing_sessions`, `indexer_in_dram`, `tokens_per_miss_layer`.
      * `"shared_prefix"`: Scenario 2 — cross-session shared prefix in
        DRAM, layer-stride prefetch. Uses `prefix_share_frac` only;
        sparse-attention knobs are ignored.

    Attributes:
        dram_capacity_gb:
            Per-GPU DRAM slice (GiB, binary). In Scenario 1 sets the
            upper bound on `max_batch_per_gpu` via the total KV-capacity
            constraint. In Scenario 2 acts as a binary feasibility
            check: the single shared-prefix copy must fit
            (`prefix_share_frac × session_kv ≤ DRAM`); bs is bound by
            HBM only (this baseline; future work allows DRAM to absorb
            unique KV overflow — see module docstring).
        dram_interconnect_bandwidth_gbs:
            Effective DRAM read bandwidth visible to one GPU, in GB/s
            (decimal, matches GPUSpec convention). For host DRAM via
            PCIe Gen5 ~ 64 GB/s; RDMA NIC ~ 25/50/100 GB/s; NVLink-C2C
            (GH200/GB200) ~ 450 GB/s/direction. Default 50 GB/s
            represents a typical PCIe / RDMA-tier deployment.
        mode:
            Which scenario to apply. See class-level docstring.
        prefix_share_frac:
            (Scenario 2 only.) Fraction of per-session KV that is
            cross-session-shared prefix, in [0, 1). The shared prefix
            lives in DRAM as a single copy (NOT replicated per session)
            and is layer-stride prefetched. The remaining
            `(1 - prefix_share_frac)` fraction is the unique KV that
            stays HBM-resident. Ignored in `sparse_on_demand` mode.
            Edge cases:
              * 0.0 → degenerates to base capacity (no DRAM benefit).
              * → 1.0 → unique part vanishes; bs_max grows unboundedly
                in this model (real deployments are then bound by KV
                pool / indexer / page-table — out of scope here).
        kv_cache_hit_rate:
            (Scenario 1 only.) Per-layer hit rate in [0, 1]. Probability
            that a given decode layer reuses the top-k tokens already
            fetched in a previous decode step (sparse-attention
            temporal locality). Independent of HBM cache size — driven
            by sparse-attention selection stability across consecutive
            steps.
            * 0.0 = naive (every layer misses → fetch from DRAM)
            * 1.0 = ideal reuse (consecutive steps select identical top-k)
        n_missing_sessions:
            (Scenario 1 only.) How many of the `batch` sessions miss on
            a given missed layer.
              * `None` (default): batch-wide pessimistic upper bound —
                resolved at runtime to `new_batch` (i.e. the entire
                batch missed simultaneously). This represents
                "all-sessions-synchronously-miss" — useful as the
                worst-case planning number.
              * `int`: explicit override. Pass 1 for the optimistic
                lower bound (only one session triggered the miss; the
                rest stayed hot — corresponds to fully de-correlated
                miss timing across sessions).
            Note: this knob and `kv_cache_hit_rate` interact
            multiplicatively in the penalty (cold traffic ∝
            (1-hit_rate) × n_missing). Some semantic overlap is a
            known modeling assumption; the two are kept separate so
            users can sweep them independently when needed.
        indexer_in_dram:
            (Scenario 1 only.) If False (default), indexer K-cache is
            HBM-resident and binds `max_batch_per_gpu` from above (the
            indexer working set must fit in HBM along with the KV hot
            cache).
            If True, indexer is moved to DRAM. A per-layer async DMA is
            assumed to be issued at the start of layer i to fetch
            layer i+1's indexer K-cache (the K_indexer for layer i+1 is
            independent of layer i's outputs). The fetch can overlap with
            the entire (attn + ffn) execution of layer i; only the
            un-hidden tail counts as a stall. The effective overlap is
            computed automatically as min(1, t_layer / t_prefetch_per_layer)
            and is reported via `indexer_prefetch_overlap_effective`.
        tokens_per_miss_layer:
            (Scenario 1 only.) Override how many tokens we assume to
            fetch per missed layer per session. Three forms:
              * `None` (default): ask the model
                (`cold_layer_kv_bytes_per_session`). Sparse-attention
                models return top_k × v_token (V3.2: ~1.25 MiB/layer).
                This is the *lower* bound — what fine-grained
                fetch-just-the-top-k could achieve in principle.
              * `int`: explicit token count, e.g. 8192 to model a
                "fetch the page that contains each top-k token" mid-
                granularity. Useful for sweeping page-size sensitivity.
              * `"page"`: one missed layer fetches the WHOLE layer's
                KV (= session_length × per-token-per-layer). This is
                a placeholder upper-bound representing
                "page-managed KV store, untuned page size, every miss
                drags in the entire layer". **NOTE:** this is NOT a
                physically calibrated number — production page-fetch
                amplification depends on page size, top-k spread, and
                eviction policy, and we have not measured those. A
                DeprecationWarning is emitted when this value is used
                so callers do not silently rely on it.
            Not exposed via CLI yet — set programmatically.
        hot_slots:
            (Scenario 1 only.) CAPACITY-side hot-cache window size, in
            tokens per layer per session. The capacity model is a physical
            hot/cold split:
              * HBM holds the hot window: min(S, hot_slots) tokens of the
                main (latent) KV per layer per session, resident across
                ALL layers (+ indexer K-cache if HBM-resident).
              * DRAM holds the cold tail: max(0, S - hot_slots) tokens of
                the main KV per layer per session (+ indexer if
                DRAM-resident).
            `max_batch_per_gpu` = min(HBM-hot bound, DRAM-cold bound).

            Resolution:
              * `None` (default) → use the model's top-k (`dsa_topk`) as
                the hot window. This is the physical MINIMUM: a decode
                layer attends top-k tokens, so at least top-k must be
                HBM-resident (and stay resident across a step for any
                cross-step reuse / nonzero hit_rate).
              * An explicit int is FLOORED at top-k (a smaller window is
                physically meaningless) and CAPPED at S.
            Real deployments often reserve MORE than top-k (e.g. 4096 for
            a 2048 top-k) to lower the miss rate — pass that value here.

            This knob only affects CAPACITY / batch sizing; it does NOT
            change the penalty model (hit_rate stays an independent perf
            knob). There is intentionally NO "no hot cache" mode: a
            nonzero hit_rate without a resident hot cache is unphysical.
    """

    dram_capacity_gb: float
    dram_interconnect_bandwidth_gbs: float = 50.0
    mode: Literal["sparse_on_demand", "shared_prefix"] = "sparse_on_demand"
    prefix_share_frac: float = 0.0
    kv_cache_hit_rate: float = 0.0
    n_missing_sessions: Optional[int] = None
    indexer_in_dram: bool = False
    hot_slots: Optional[int] = None
    # Default is None → use the model's native cold-fetch estimate
    # (e.g. V3.2 top-k × v_token). Switched from "page" on 2026-05-25
    # because "page" assumed a 100% layer-fetch amplification with no
    # empirical backing — it forced hot-cache fittable to ~75× smaller
    # values than the true sparse-attention upper bound and made
    # capacity reports overly pessimistic. See the field docstring
    # above for the deprecation note.
    tokens_per_miss_layer: Union[int, str, None] = None

    def __post_init__(self) -> None:
        if self.mode not in ("sparse_on_demand", "shared_prefix"):
            raise ValueError(
                f"DramPoolingConfig.mode must be 'sparse_on_demand' or "
                f"'shared_prefix', got {self.mode!r}."
            )
        if self.mode == "shared_prefix":
            # Allow [0, 1]: 0 degenerates to base; 1.0 is the "all KV is
            # shared prefix" corner — unique part vanishes and bs is no
            # longer bound by HBM in this model. The optimization
            # routine annotates this case with bs_bound_by="model-
            # unbounded" and reports a sentinel large bs; real
            # deployments are then bound by KV pool / indexer / page-
            # table — out of scope here.
            if not (0.0 <= self.prefix_share_frac <= 1.0):
                raise ValueError(
                    f"DramPoolingConfig.prefix_share_frac must be in "
                    f"[0, 1] for mode='shared_prefix', got "
                    f"{self.prefix_share_frac!r}."
                )

    @property
    def dram_capacity_bytes(self) -> float:
        return self.dram_capacity_gb * (1024 ** 3)

    @property
    def dram_interconnect_bw_bytes_per_sec(self) -> float:
        # GB/s in decimal (matches GPUSpec.bandwidth_gbs convention).
        return self.dram_interconnect_bandwidth_gbs * (10 ** 9)


class DramPoolingOptimization(CapacityOptimization):
    """DRAM pooling capacity optimization (two scenarios via `mode` flag).

    **Scenario 1 — `mode="sparse_on_demand"` (default).**
    Capacity uses a physical hot-cache / cold-tail split (see
    `DramPoolingConfig.hot_slots`). HBM holds a hot window of
    `hot_slots` tokens/layer/session (default = model top-k, floored at
    top-k) resident across all layers, plus the indexer K-cache if
    HBM-resident; DRAM holds the cold tail. The two binding constraints:
      * HBM-hot: bs × (hot_main + indexer_hbm) ≤ HBM_avail
      * DRAM-cold: bs × (cold_main + indexer_dram) ≤ DRAM_capacity

    `max_batch_per_gpu = min(HBM-hot bound, DRAM-cold bound)`; the binding
    one is reported as `bs_bound_by` ("HBM-hot-cap" / "DRAM-cold-cap" /
    "user-override"). The penalty is the per-layer on-demand fetch cost
    (cannot be prefetched — depends on this layer's indexer output),
    plus an optional roofline-overlapped indexer-in-DRAM term.

    **Scenario 2 — `mode="shared_prefix"`.**
    Cross-session shared prefix (system prompt, tool catalog, RAG
    context, ...) lives as a SINGLE shared copy in DRAM and is
    prefetchable per-layer (no indexer dependency). The per-session
    unique KV stays HBM-resident.

    Capacity (this baseline; see Future Work in module docstring):
      * HBM-cap: bs × (1 - α) × session_kv ≤ HBM_avail
      * DRAM-cap (binary feasibility): α × session_kv ≤ DRAM
        (single shared copy, NOT replicated per session)

    Penalty (roofline overlap, identical structure to Scenario 1's
    indexer-in-DRAM term but WITHOUT the bs multiplier):
      * t_layer            = (attn + ffn) / num_layers
      * prefix_per_layer   = α × session_kv / num_layers   (NOT × bs)
      * t_prefetch         = prefix_per_layer / dram_bw
      * overlap            = min(1, t_layer / t_prefetch)
      * penalty_per_step   = (α × session_kv / dram_bw) × (1 - overlap)
    """

    name = "dram_pooling"

    def __init__(self, config: DramPoolingConfig):
        self.config = config

    # ------------------------------------------------------------------
    def apply(
        self,
        baseline: CapacityReport,
        *,
        gpu: GPUSpec,
        mem: MemoryProfile,
        parallel: ParallelConfig,
        model_overrides: Mapping[str, Any] | None = None,
        batch_size_override: Optional[int] = None,
    ) -> OptimizedCapacityReport:
        """Apply DRAM pooling. Pure router — dispatches to a per-scenario
        implementation by `self.config.mode`.

        Each scenario's full physics (capacity model, penalty, knob
        semantics) lives in its own private method:

          * ``sparse_on_demand`` → :meth:`_apply_sparse_on_demand`
          * ``shared_prefix``    → :meth:`_apply_shared_prefix`

        See class docstring for the high-level scenario contrast.
        """
        common_kwargs = dict(
            gpu=gpu,
            mem=mem,
            parallel=parallel,
            model_overrides=model_overrides,
            batch_size_override=batch_size_override,
        )
        mode = self.config.mode
        if mode == "shared_prefix":
            return self._apply_shared_prefix(baseline, **common_kwargs)
        if mode == "sparse_on_demand":
            return self._apply_sparse_on_demand(baseline, **common_kwargs)
        raise ValueError(
            f"Unknown DramPoolingConfig.mode={mode!r}; expected one of "
            f"'sparse_on_demand' | 'shared_prefix'."
        )

    # ------------------------------------------------------------------
    def _apply_sparse_on_demand(
        self,
        baseline: CapacityReport,
        *,
        gpu: GPUSpec,
        mem: MemoryProfile,
        parallel: ParallelConfig,
        model_overrides: Mapping[str, Any] | None = None,
        batch_size_override: Optional[int] = None,
    ) -> OptimizedCapacityReport:
        """Scenario 1 — Sparse-attention on-demand DRAM pool.

        See the class docstring for the high-level physics summary; this
        method is the full implementation. Knobs that participate:

          * ``kv_cache_hit_rate``      — per-layer hit rate ∈ [0, 1].
          * ``n_missing_sessions``     — distinct sessions whose top-k
            page-set must be fetched on a miss. ``None`` (default)
            falls back to ``new_batch`` at runtime (pessimistic
            batch-wide upper bound); pass an explicit ``int`` (typically
            ``1``) for the optimistic lower bound.
          * ``tokens_per_miss_layer``  — None (model-native, e.g. V3.2
            top-k), int (explicit token count), or "page" (entire layer
            page = upper bound).
          * ``indexer_in_dram``        — keep indexer K-cache in DRAM
            (relieves HBM pressure but adds a layer-stride DRAM prefetch
            term, which is auto-overlapped against (attn + ffn) compute).
        """
        cfg = self.config
        notes: list[str] = []

        # 1) Resolve model + per-session quantities ------------------------
        try:
            model = get_model(baseline.model_name)
            merged_cfg = model.merged_config(model_overrides)
            num_layers = int(merged_cfg.get("num_layers", 0))
        except KeyError as exc:
            raise ValueError(
                f"Cannot apply DramPoolingOptimization: model "
                f"{baseline.model_name!r} not registered ({exc})."
            ) from exc
        if num_layers <= 0:
            raise ValueError(
                f"Cannot apply DramPoolingOptimization: model "
                f"{baseline.model_name!r} has no 'num_layers' in merged_config."
            )

        S = int(baseline.session_length)
        # Indexer & cold-layer per-session bytes (model-aware).
        indexer_kv_per_session = float(model.indexer_kv_bytes_per_session(
            S, parallel,
            baseline.workload.dtype_kv,
            model_overrides=model_overrides,
        ))
        cold_layer_per_session_default = float(model.cold_layer_kv_bytes_per_session(
            S, parallel,
            baseline.workload.dtype_kv,
            model_overrides=model_overrides,
        ))
        # `cold_layer_per_session_default` is the per-session, per-layer cold
        # fetch in the model's native semantics (e.g. V3.2 top-k × v_token).
        # When the user overrides `tokens_per_miss_layer` we recompute on a
        # per-token basis from `session_kv_bytes / (n_layers × S)`.
        per_token_per_layer_kv = (
            baseline.session_kv_bytes / max(1, num_layers * S)
            if (num_layers > 0 and S > 0) else 0.0
        )
        tpml = cfg.tokens_per_miss_layer
        if tpml is None:
            # Model-aware sparse-aware lower bound (e.g. V3.2 top_k × v_token).
            cold_layer_per_session = cold_layer_per_session_default
        elif isinstance(tpml, str):
            if tpml.lower() == "page":
                # Placeholder upper-bound: assume a missed page-managed
                # KV store drags in the WHOLE layer per session. This
                # overestimates the true page-fetch amplification (the
                # real factor depends on page size × top-k spread ×
                # eviction policy, none of which we have measured).
                # Emit a DeprecationWarning so callers do not silently
                # rely on this number.
                import warnings
                warnings.warn(
                    "DramPoolingConfig.tokens_per_miss_layer='page' is a "
                    "placeholder upper bound (= session_length tokens, i.e. "
                    "one full layer KV per session) and is NOT empirically "
                    "calibrated. Real page-fetch amplification depends on "
                    "page size, top-k spread, and eviction policy. Prefer "
                    "the default (None → model-native top-k lower bound) "
                    "or pass an explicit measured int token count.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                cold_layer_per_session = S * per_token_per_layer_kv
            else:
                raise ValueError(
                    f"tokens_per_miss_layer string must be 'page', "
                    f"got {tpml!r}."
                )
        else:
            # Explicit token count: e.g. 8192 to model a "fetch the page
            # containing each top-k entry" mid-granularity case.
            cold_layer_per_session = int(tpml) * per_token_per_layer_kv

        # 2) Batch-size capacity constraints (hot-cache / cold-tail split) --
        # Physical model: each decode layer attends `top_k` tokens, so HBM
        # MUST hold at least top_k tokens/layer/session resident to serve
        # attention AND to enable cross-step reuse (a nonzero hit_rate is
        # impossible without a persistent hot cache). Deployments often
        # reserve MORE slots than top_k (e.g. 4096 for a 2048 top-k) to
        # lower the miss rate. HBM holds the hot window across all L layers;
        # DRAM holds the cold tail. `max_batch` = min(HBM-hot, DRAM-cold).
        #
        # `hot_slots` resolution: default = model top-k (dsa_topk); floored
        # at top-k (a smaller window is physically meaningless); capped at S
        # (cannot cache more than the whole session).
        # Model-native sparse top-k: DSA models expose `dsa_topk`; V4-family
        # hybrid attention exposes `csa_topk`. Fall back to S (dense → whole
        # session hot) only if the model is truly non-sparse.
        topk_raw = merged_cfg.get("dsa_topk") or merged_cfg.get("csa_topk")
        top_k_tokens = int(topk_raw) if topk_raw else S  # dense → all N
        if cfg.hot_slots is None:
            eff_slots = top_k_tokens                     # default = model top-k
        else:
            eff_slots = int(cfg.hot_slots)
            if eff_slots < top_k_tokens:
                notes.append(
                    f"hot_slots={eff_slots} below top-k floor "
                    f"({top_k_tokens}); clamped up (cannot cache fewer than "
                    f"the tokens each layer attends)."
                )
                eff_slots = top_k_tokens
        eff_slots = min(max(1, S), eff_slots)            # cannot exceed session

        S_tok = max(1, S)
        # Main (latent+rope) KV excludes the indexer K-cache, accounted
        # separately per its residency.
        main_kv_per_session = max(
            0.0, baseline.session_kv_bytes - indexer_kv_per_session
        )
        hot_frac = eff_slots / S_tok
        hot_main = hot_frac * main_kv_per_session
        cold_main = max(0.0, main_kv_per_session - hot_main)
        hbm_per_session = hot_main + (
            0.0 if cfg.indexer_in_dram else indexer_kv_per_session
        )
        dram_per_session = cold_main + (
            indexer_kv_per_session if cfg.indexer_in_dram else 0.0
        )
        auto_max_hbm = (
            int(baseline.hbm_avail_bytes // hbm_per_session)
            if hbm_per_session > 0 else 10**18
        )
        auto_max_dram = (
            int(cfg.dram_capacity_bytes // dram_per_session)
            if dram_per_session > 0 else 10**18
        )
        auto_max = min(auto_max_hbm, auto_max_dram)
        bound_label = (
            "DRAM-cold-cap"
            if auto_max == auto_max_dram <= auto_max_hbm
            else "HBM-hot-cap"
        )
        # Aliases so the downstream infeasible-note block stays generic.
        auto_max_capacity = auto_max_dram
        auto_max_indexer = auto_max_hbm

        # 3) Resolve batch_size + bound annotation -------------------------
        bs_bound_by = "unknown"
        if batch_size_override is None:
            new_batch = auto_max
            bs_bound_by = bound_label
        else:
            req = int(batch_size_override)
            if req < 1:
                new_batch = 0
                notes.append(f"infeasible: batch_size_override={req} < 1.")
                bs_bound_by = "user-override"
            elif req > auto_max:
                new_batch = auto_max
                notes.append(
                    f"batch_size_override={req} clamped to auto_max="
                    f"{auto_max} ({bound_label})."
                )
                bs_bound_by = bound_label
            else:
                new_batch = req
                bs_bound_by = "user-override"

        feasible = new_batch >= 1

        # 4) Hot-cache residency (diagnostic; consistent with capacity) ----
        # The capacity model (Step 2) ALREADY reserves the hot window in HBM
        # — eff_slots tokens/layer across all L layers — so the persistent
        # hot cache required for cross-step reuse is resident by construction.
        # A nonzero hit_rate is therefore always supportable and is taken
        # directly from the user knob. (The old "force hit_rate=0 if <1 layer
        # fits" guard belonged to the legacy total-KV model, where nothing
        # reserved the hot set; it is obsolete under the hot_slots model.)
        effective_hit = max(0.0, min(1.0, cfg.kv_cache_hit_rate))
        # Diagnostic only: how many layers' worth of the (dtype-aware) hot
        # window fit in the HBM left after the indexer reservation. By
        # construction this is >= num_layers; a larger value means HBM has
        # headroom (bs is DRAM-cold-bound, not HBM-hot-bound). Uses the SAME
        # hot-window bytes as the capacity split (`hot_main`), NOT the
        # FP8 cold-fetch estimate, so the two stay consistent.
        indexer_hbm_reserved = (
            new_batch * indexer_kv_per_session
            if not cfg.indexer_in_dram else 0.0
        )
        hbm_left_for_hot = max(
            0.0, baseline.hbm_avail_bytes - indexer_hbm_reserved
        )
        hot_window_one_layer = new_batch * hot_main / max(1, num_layers)
        hot_cache_layers_fittable = (
            hbm_left_for_hot / hot_window_one_layer
            if hot_window_one_layer > 0 else 0.0
        )

        # 5) Compute-side performance via estimate_perf -------------------
        perf_rep = None
        penalty_s = 0.0
        prefetch_overlap_effective: Optional[float] = None
        if feasible:
            wl = WorkloadConfig(
                phase=Phase.DECODE,
                batch_size=new_batch,
                context_length=baseline.workload.context_length,
                new_tokens=baseline.workload.new_tokens,
                dtype_param=baseline.workload.dtype_param,
                dtype_kv=baseline.workload.dtype_kv,
            )
            perf_rep = estimate_perf(
                gpu=gpu,
                model_name=baseline.model_name,
                parallel=parallel,
                workload=wl,
                model_overrides=model_overrides,
            )

            # 6) Penalty model ---------------------------------------------
            #
            # Per-layer barrier penalty (sparse on-demand main attn):
            #   miss_layers     = num_layers × (1 - effective_hit)
            #   cold_per_miss   = cold_layer_per_session × n_missing_sessions
            #   penalty_main    = miss_layers × cold_per_miss / dram_bw
            #
            # Indexer-in-DRAM penalty with roofline-derived prefetch overlap:
            #   The DMA for layer i+1's indexer K can be issued at the start
            #   of layer i (the K_indexer of layer i+1 is independent of
            #   layer i's outputs), and must complete before layer i+1's
            #   indexer compute starts. So the overlap window is the full
            #   layer i execution time:
            #
            #     t_layer = (perf_rep.attn.time + perf_rep.ffn.time) / num_layers
            #               (perf_rep.attn already includes indexer compute,
            #                bundled into the attn stage cost — no double
            #                counting.)
            #     t_prefetch_per_layer
            #             = (bs × indexer_kv_per_session) / num_layers / dram_bw
            #     effective_overlap
            #             = min(1, t_layer / t_prefetch_per_layer)
            #     stall_per_layer
            #             = max(0, t_prefetch_per_layer - t_layer)
            #     penalty_indexer
            #             = num_layers × stall_per_layer
            #             = (bs × indexer_kv_per_session / dram_bw)
            #               × (1 - effective_overlap)
            #
            # Overlap is fully determined by physical quantities (no user
            # knob): if prefetch is faster than the layer-time window it's
            # fully hidden; otherwise the un-hidden tail counts as stall.
            #
            # Total: penalty = penalty_main + penalty_indexer
            miss_layers = num_layers * (1.0 - effective_hit)
            # n_missing_sessions: None → fall back to the resolved batch
            # size (pessimistic batch-wide miss upper bound). Otherwise
            # honour the user-provided int (clamped to [1, new_batch] so
            # an over-large override is silently capped at the actual
            # batch).
            if cfg.n_missing_sessions is None:
                n_missing = new_batch
            else:
                n_missing = max(1, min(int(cfg.n_missing_sessions), new_batch))
            cold_per_miss = cold_layer_per_session * n_missing
            bw = cfg.dram_interconnect_bw_bytes_per_sec
            if bw > 0:
                penalty_main = miss_layers * cold_per_miss / bw
                penalty_indexer = 0.0
                if cfg.indexer_in_dram and indexer_kv_per_session > 0:
                    indexer_per_step = new_batch * indexer_kv_per_session
                    indexer_per_layer = indexer_per_step / max(1, num_layers)
                    t_prefetch_per_layer = indexer_per_layer / bw
                    if perf_rep is not None and num_layers > 0:
                        t_layer = (
                            perf_rep.attn.time_seconds
                            + perf_rep.ffn.time_seconds
                        ) / num_layers
                    else:
                        t_layer = 0.0
                    if t_prefetch_per_layer > 0:
                        overlap = min(1.0, t_layer / t_prefetch_per_layer)
                    else:
                        overlap = 1.0
                    prefetch_overlap_effective = overlap
                    penalty_indexer = (
                        indexer_per_step / bw
                    ) * (1.0 - overlap)
                penalty_s = penalty_main + penalty_indexer
            else:
                penalty_s = 0.0
                notes.append(
                    "dram_interconnect_bandwidth_gbs <= 0; "
                    "skipping penalty calculation."
                )
        else:
            if not notes:
                hint = (
                    " Try --indexer-in-dram."
                    if (bound_label == "HBM-hot-cap"
                        and not cfg.indexer_in_dram
                        and indexer_kv_per_session > 0)
                    else ""
                )
                notes.append(
                    f"infeasible (hot_slots={eff_slots} tok/layer): even one "
                    f"session's hot window + cold tail cannot be placed "
                    f"(bound={bound_label}; HBM_avail="
                    f"{baseline.hbm_avail_bytes/2**30:.2f} GiB, DRAM="
                    f"{cfg.dram_capacity_bytes/2**30:.2f} GiB).{hint}"
                )

        # mem param is part of the ABC signature but not consulted here.
        del mem

        return OptimizedCapacityReport(
            optimization_name=self.name,
            baseline=baseline,
            max_batch_per_gpu=int(new_batch),
            feasible=bool(feasible),
            extra_capacity_bytes=float(cfg.dram_capacity_bytes),
            perf_report=perf_rep,
            penalty_seconds=float(penalty_s),
            notes=notes,
            bs_bound_by=bs_bound_by,
            indexer_kv_bytes_total=float(
                new_batch * indexer_kv_per_session
                if not cfg.indexer_in_dram else 0.0
            ),
            hot_cache_layers_fittable=float(hot_cache_layers_fittable),
            indexer_prefetch_overlap_effective=prefetch_overlap_effective,
            # DRAM actually used = bs × cold-tail (+ indexer if in DRAM).
            dram_used_bytes=float(new_batch * dram_per_session),
        )


    # ------------------------------------------------------------------
    def _apply_shared_prefix(
        self,
        baseline: CapacityReport,
        *,
        gpu: GPUSpec,
        mem: MemoryProfile,
        parallel: ParallelConfig,
        model_overrides: Mapping[str, Any] | None = None,
        batch_size_override: Optional[int] = None,
    ) -> OptimizedCapacityReport:
        """Scenario 2 — Shared-Prefix Prefetch DRAM Pooling (with layer-spill).

        Physics overview
        ----------------
            α    = cfg.prefix_share_frac        ∈ [0, 1]
            S_kv = baseline.session_kv_bytes
            L    = num_layers, bw = dram_bw

            shared_prefix_bytes = α × S_kv               # single copy, NOT × bs
            unique_per_session  = (1 - α) × S_kv         # per-session unique KV
            PLU                 = unique_per_session / L # per-layer unique
                                                          # bytes per session

        Capacity (with automatic layer-stripe spill to DRAM)
        ----------------------------------------------------
        Production fact (motivates layer-split, not session-split): in PD-
        disagg deployments the decode node receives KV from prefill
        layer-by-layer (NCCL/NIXL streaming), and the KV pool's page table
        is also organized layer × token. So when HBM cannot hold the full
        per-batch unique KV, the natural physical spill unit is the TAIL k
        layers — they're the last to be filled and the last to be consumed
        within a decode step.

            spilled_unique_total(bs) = max(0, bs × (1-α) × S_kv - HBM_avail)
            k(bs) = ceil(spilled_unique_total(bs) / (bs × PLU))   # in [0, L]

            DRAM occupancy at bs:
              prefix             = α × S_kv
              spilled_unique     = k × bs × PLU
              total_dram_used    = prefix + spilled_unique

            Feasibility:
              (P) prefix      ≤ DRAM_capacity                # else "DRAM-prefix-cap"
              (U) spilled_uniq ≤ DRAM_capacity - prefix      # else "DRAM-spill-cap"
              (Z) k(bs) ≤ L                                  # otherwise infeasible

            Auto-max bs (no override):
              bs_max_analytic = (HBM_avail + (DRAM_cap - α × S_kv))
                                / ((1 - α) × S_kv)
            Then re-derive k(bs_max), and back off bs by 1 if the integer
            ceil(k) violates DRAM at the analytic bound (page-align edge).

        Penalty (layer-split roofline overlap)
        --------------------------------------
        Each decode step iterates layers 0..L-1; at the start of layer i
        an async DMA is issued for layer i+1's KV. The shared prefix is
        always pulled from DRAM (single copy). The unique part of layer
        i+1 is pulled from DRAM only if i+1 is one of the spilled tail-k
        layers; otherwise it is HBM-resident and incurs no DMA.

            t_layer            = (attn + ffn) / L
            light layers       = L - k                  # unique in HBM
            heavy layers       = k                      # unique in DRAM

            t_prefetch_light   = (α × S_kv / L) / bw
            overlap_light      = min(1, t_layer / t_prefetch_light)
            penalty_light      = (L - k) × t_prefetch_light × (1 - overlap_light)

            t_prefetch_heavy   = (α × S_kv / L + bs × PLU) / bw
            overlap_heavy      = min(1, t_layer / t_prefetch_heavy)
            penalty_heavy      = k × t_prefetch_heavy × (1 - overlap_heavy)

            penalty_total      = penalty_light + penalty_heavy

        Reuse of report fields
        ----------------------
          * bs_bound_by:
              "DRAM-prefix-cap" — prefix alone exceeds DRAM (infeasible)
              "DRAM-spill-cap"  — spilled-unique exceeds remaining DRAM
              "HBM-cap"         — even with k=L spill, HBM can't fit the
                                  HBM-resident leftover (rare; typically
                                  k=L means everything spilled)
              "model-unbounded" — α=1 corner: unique vanishes
              "user-override"   — explicit batch_size_override consumed
          * indexer_prefetch_overlap_effective: light-path overlap.
          * spilled_layers_count: k.
          * spilled_unique_overlap_effective: heavy-path overlap.
            None when k=0 (no spill, baseline behaviour).
          * hot_cache_layers_fittable: +∞ (concept doesn't apply).
          * indexer_kv_bytes_total: 0 (no indexer in this mode).
          * extra_capacity_bytes: full DRAM slice (pool size, not used).
        """
        cfg = self.config
        notes: list[str] = []

        # 1) Resolve model + per-session quantities ------------------------
        try:
            model = get_model(baseline.model_name)
            merged_cfg = model.merged_config(model_overrides)
            num_layers = int(merged_cfg.get("num_layers", 0))
        except KeyError as exc:
            raise ValueError(
                f"Cannot apply DramPoolingOptimization (shared_prefix): "
                f"model {baseline.model_name!r} not registered ({exc})."
            ) from exc
        if num_layers <= 0:
            raise ValueError(
                f"Cannot apply DramPoolingOptimization (shared_prefix): "
                f"model {baseline.model_name!r} has no 'num_layers' in "
                f"merged_config."
            )

        alpha = float(cfg.prefix_share_frac)
        S_kv = float(baseline.session_kv_bytes)
        unique_per_session = (1.0 - alpha) * S_kv
        shared_prefix_bytes = alpha * S_kv
        per_layer_unique_per_session = unique_per_session / float(num_layers)
        L = num_layers

        dram_cap = float(cfg.dram_capacity_bytes)
        hbm_avail = max(0.0, float(baseline.hbm_avail_bytes))

        # ------------------------------------------------------------------
        # Local helpers (kept inside the method to avoid leaking state).
        # ------------------------------------------------------------------
        import math

        def _k_for_bs(bs: int) -> int:
            """Number of tail layers spilled to DRAM at batch size `bs`.

            Layer-page aligned: ceil up so an integer number of full
            layers absorb the HBM overflow. Returns a value in [0, L].
            """
            if bs <= 0 or unique_per_session <= 0:
                return 0
            unique_total = bs * unique_per_session
            spilled = unique_total - hbm_avail
            if spilled <= 0:
                return 0
            per_layer_unique_total = bs * per_layer_unique_per_session
            if per_layer_unique_total <= 0:
                return 0
            k = math.ceil(spilled / per_layer_unique_total)
            return min(L, max(0, k))

        def _dram_used_for(bs: int, k: int) -> float:
            """DRAM bytes consumed at (bs, k)."""
            return shared_prefix_bytes + k * bs * per_layer_unique_per_session

        def _bs_feasibility(bs: int) -> tuple[bool, str, int]:
            """Check if `bs` is feasible. Return (ok, reason, k)."""
            if shared_prefix_bytes > dram_cap:
                return False, "DRAM-prefix-cap", 0
            if bs <= 0:
                return False, "HBM-cap", 0
            k = _k_for_bs(bs)
            if k > L:
                # math says k ∈ [0, L] already, but be defensive.
                return False, "HBM-cap", k
            if k == L and bs * unique_per_session > L * bs * per_layer_unique_per_session:
                # Sanity (cannot trigger by construction; left for clarity).
                return False, "HBM-cap", k
            if _dram_used_for(bs, k) > dram_cap:
                return False, "DRAM-spill-cap", k
            return True, "feasible", k

        # 2) Compute auto-max bs --------------------------------------------
        prefix_fits = shared_prefix_bytes <= dram_cap
        unique_is_zero = unique_per_session <= 0

        if not prefix_fits:
            auto_max = 0
        elif unique_is_zero:
            # α=1.0 corner — bs not bound by HBM in this model.
            base_bs = max(1, baseline.max_batch_per_gpu)
            auto_max = max(1024, 16 * base_bs)
        else:
            # Analytic: HBM + (DRAM - prefix) absorbs all per-batch unique.
            dram_for_unique = max(0.0, dram_cap - shared_prefix_bytes)
            denom = unique_per_session
            if denom > 0:
                bs_analytic = int((hbm_avail + dram_for_unique) // denom)
            else:
                bs_analytic = 0
            # Page-align safety: at the analytic bs, reconfirm DRAM fits;
            # if not (because k must be integer), back off by 1.
            auto_max = max(0, bs_analytic)
            while auto_max > 0:
                ok, _why, _k = _bs_feasibility(auto_max)
                if ok:
                    break
                auto_max -= 1

        # 3) Resolve batch_size + bound annotation -------------------------
        bs_bound_by = "unknown"
        new_batch = 0
        k_chosen = 0
        feas_reason = "feasible"

        if not prefix_fits:
            new_batch = 0
            notes.append(
                f"infeasible: shared prefix "
                f"({shared_prefix_bytes/2**30:.2f} GiB, α="
                f"{alpha:.3f}) exceeds DRAM capacity "
                f"({dram_cap/2**30:.2f} GiB)."
            )
            bs_bound_by = "DRAM-prefix-cap"
        elif batch_size_override is None:
            new_batch = auto_max
            if unique_is_zero:
                bs_bound_by = "model-unbounded"
                k_chosen = 0
                notes.append(
                    f"α=1.0 corner: unique KV per session = 0; bs not "
                    f"bound by HBM in this model. Reporting sentinel "
                    f"bs={auto_max} (= 16× HBM-only baseline). Real "
                    f"deployments are bound by KV pool / indexer / "
                    f"page-table — out of scope here."
                )
            elif new_batch < 1:
                bs_bound_by = "HBM-cap"
                notes.append(
                    f"infeasible: HBM_avail "
                    f"({hbm_avail/2**30:.2f} GiB) + DRAM spill "
                    f"({max(0.0, dram_cap-shared_prefix_bytes)/2**30:.2f} GiB) "
                    f"insufficient for one session's unique KV "
                    f"({unique_per_session/2**30:.2f} GiB, α={alpha:.3f})."
                )
            else:
                k_chosen = _k_for_bs(new_batch)
                # Distinguish HBM-only (k=0) from spill (k>0).
                if k_chosen == 0:
                    bs_bound_by = "HBM-cap"
                else:
                    bs_bound_by = "DRAM-spill-cap"
        else:
            req = int(batch_size_override)
            if req < 1:
                new_batch = 0
                notes.append(f"infeasible: batch_size_override={req} < 1.")
                bs_bound_by = "user-override"
            else:
                ok, why, k = _bs_feasibility(req)
                if ok:
                    new_batch = req
                    k_chosen = k
                    bs_bound_by = "user-override"
                else:
                    # Clamp to auto_max and re-derive k.
                    new_batch = auto_max
                    if new_batch >= 1 and not unique_is_zero:
                        k_chosen = _k_for_bs(new_batch)
                    bs_bound_by = why  # DRAM-prefix-cap / DRAM-spill-cap / HBM-cap
                    notes.append(
                        f"batch_size_override={req} infeasible ({why}); "
                        f"clamped to auto_max={auto_max}."
                    )

        feasible = new_batch >= 1
        if feasible and k_chosen > 0:
            spilled_bytes = k_chosen * new_batch * per_layer_unique_per_session
            notes.append(
                f"layer-spill: {k_chosen}/{L} tail layers' unique KV "
                f"({spilled_bytes/2**30:.2f} GiB total) spilled to DRAM; "
                f"HBM holds first {L-k_chosen} layers' unique."
            )

        # 4) Compute-side performance via estimate_perf -------------------
        perf_rep = None
        penalty_s = 0.0
        light_overlap_effective: Optional[float] = None
        heavy_overlap_effective: Optional[float] = None
        if feasible:
            wl = WorkloadConfig(
                phase=Phase.DECODE,
                batch_size=new_batch,
                context_length=baseline.workload.context_length,
                new_tokens=baseline.workload.new_tokens,
                dtype_param=baseline.workload.dtype_param,
                dtype_kv=baseline.workload.dtype_kv,
            )
            perf_rep = estimate_perf(
                gpu=gpu,
                model_name=baseline.model_name,
                parallel=parallel,
                workload=wl,
                model_overrides=model_overrides,
            )

            # 5) Layer-split roofline prefetch penalty --------------------
            bw = cfg.dram_interconnect_bw_bytes_per_sec
            if bw > 0:
                if perf_rep is not None:
                    t_layer = (
                        perf_rep.attn.time_seconds
                        + perf_rep.ffn.time_seconds
                    ) / float(L)
                else:
                    t_layer = 0.0

                prefix_per_layer = shared_prefix_bytes / float(L)
                t_pref_light = prefix_per_layer / bw
                if t_pref_light > 0:
                    light_overlap = min(1.0, t_layer / t_pref_light)
                else:
                    light_overlap = 1.0  # no prefix to fetch (α=0)

                if k_chosen > 0:
                    per_layer_unique_total = (
                        new_batch * per_layer_unique_per_session
                    )
                    t_pref_heavy = (
                        prefix_per_layer + per_layer_unique_total
                    ) / bw
                    if t_pref_heavy > 0:
                        heavy_overlap = min(1.0, t_layer / t_pref_heavy)
                    else:
                        heavy_overlap = 1.0

                    n_light = L - k_chosen
                    penalty_light = (
                        n_light * t_pref_light * (1.0 - light_overlap)
                    )
                    penalty_heavy = (
                        k_chosen * t_pref_heavy * (1.0 - heavy_overlap)
                    )
                    penalty_s = penalty_light + penalty_heavy

                    # Surface only when meaningful.
                    light_overlap_effective = (
                        light_overlap if shared_prefix_bytes > 0 else None
                    )
                    heavy_overlap_effective = heavy_overlap
                else:
                    # No spill — only prefix prefetch contributes.
                    if shared_prefix_bytes > 0:
                        penalty_s = (
                            L * t_pref_light * (1.0 - light_overlap)
                        )
                        light_overlap_effective = light_overlap
                    else:
                        penalty_s = 0.0
                        light_overlap_effective = None
                    heavy_overlap_effective = None
            else:
                notes.append(
                    "dram_interconnect_bandwidth_gbs <= 0; "
                    "skipping penalty calculation."
                )

        # mem param is part of the ABC signature but not consulted here.
        del mem
        del feas_reason

        return OptimizedCapacityReport(
            optimization_name=self.name,
            baseline=baseline,
            max_batch_per_gpu=int(new_batch),
            feasible=bool(feasible),
            extra_capacity_bytes=float(cfg.dram_capacity_bytes),
            perf_report=perf_rep,
            penalty_seconds=float(penalty_s),
            notes=notes,
            bs_bound_by=bs_bound_by,
            indexer_kv_bytes_total=0.0,  # not applicable in shared_prefix mode
            hot_cache_layers_fittable=float("inf"),  # concept does not apply
            indexer_prefetch_overlap_effective=light_overlap_effective,
            spilled_layers_count=(int(k_chosen) if feasible else None),
            spilled_unique_overlap_effective=heavy_overlap_effective,
            # DRAM used = shared prefix (single copy) + spilled tail-k unique
            # KV × bs. 0 when infeasible.
            dram_used_bytes=(
                float(_dram_used_for(int(new_batch), int(k_chosen)))
                if feasible else 0.0
            ),
        )


__all__ = [
    "DramPoolingConfig",
    "DramPoolingOptimization",
]
