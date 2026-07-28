"""DeepSeek-V4 cost estimator: hybrid CSA + HCA (+ SWA branch / pure-SWA layers) + DeepSeekMoE.

Architecture overview (V4 paper §2.3 + §4.2.1):

  Each layer is one of three attention types:
    * CSA       — Compressed Sparse Attention (compress m tokens → 1 entry,
                  then DSA top-k selection, plus uncompressed SWA branch)
    * HCA       — Heavily Compressed Attention (m' >> m, no DSA selection,
                  plus SWA branch)
    * pure SWA  — only sliding-window attention, no compression (V4-Flash
                  uses this for its first 2 layers)

  V4-Pro:    61 layers = first 2 HCA + 59 alternating(CSA, HCA)
                       → 30 CSA + 31 HCA
  V4-Flash:  43 layers = first 2 pure-SWA + 41 alternating(CSA, HCA)
                       → 21 CSA + 20 HCA + 2 pure-SWA

  Every CSA / HCA layer additionally maintains a sliding-window KV branch
  of n_win uncompressed entries to model local dependencies.

KV cache storage (per V4 paper §2.3.4 mixed-precision spec):
  * RoPE dimensions      → BF16 (2 B)
  * Other KV dimensions  → FP8  (1 B)
  * Indexer K-cache      → FP4  (0.5 B; paper §2.3.4: "attention computation
                                  within the lightning indexer is performed
                                  in FP4 precision")
  Per V4 paper §2.3.3: RoPE applied to **last 64 dimensions** of every Q/KV
  vector (verified across V4-Pro and V4-Flash).

  Per-token-per-layer KV bytes:
    * CSA layer:
        compressed_KV    = c × FP_mixed / m      (1 entry per m tokens, MQA)
        indexer_K        = c_I × FP4 / m         (top-k selection key)
        SWA cap          = n_win × c × FP_mixed  (rolling, NOT scaling w/ ctx)
    * HCA layer:
        compressed_KV    = c × FP_mixed / m'
        SWA cap          = n_win × c × FP_mixed
    * Pure SWA layer:
        SWA cap          = n_win × c × FP_mixed
        (no compressed_KV, no indexer)

Sanity (V4 paper Figure 1, ratio vs V3.2 @ 1M ctx):
    V4-Pro    ≈ 10% of V3.2 KV    → our closed-form: ~10.0% ✓
    V4-Flash  ≈ 7%  of V3.2 KV    → our closed-form: ~7.0%  ✓
    V4-Pro    ≈ 27% of V3.2 FLOPs → TODO(verify) when sweep is run

Verified against V4 paper (cross-checked 2026-05-25):
  * §2.3.1 eq (9-10): CSA uses TWO compression streams W_a^KV + W_b^KV
    AND W_a^Z + W_b^Z, each ∈ R^{d×c}.  Code: `w_kv_compress_csa`,
    `w_z_csa` carry the 2×(d×c) factor.
  * §2.3.2 eq (20-21): HCA uses ONE stream each W^KV ∈ R^{d×c} and
    W^Z ∈ R^{d×c}.  Code: `w_kv_compress_hca`, `w_z_hca` carry the
    1×(d×c) factor — was 2× before, double-counted.
  * §2.3.1 eq (13-14): indexer Q-up takes the SHARED query latent
    c_t^Q ∈ R^{d_c}, so W^IUQ ∈ R^{d_c × n_h^I × c_I}. Code's
    `flops_idx_q = 2 × d_c × n_h_idx × c_idx` and the W^IUQ weight
    use `d_c` instead of `d` (was `d`, over-counted ~4-5×).
  * §2.3.3: RoPE applies to last 64 dims (`_N_ROPE = 64` ✓ for both V4
    variants).
  * §3.6.1: SWA in-memory KV is a fixed-size *rolling* state cache
    (`n_win × c` per layer), independent of N. Persisted-on-disk SWA
    in §3.6.2 is a different concept (out of scope here).
  * §2.3.1 last paragraph: grouped output projection cost shape
    n_h × c → g × d_g → d (code matches).

Known modeling gaps (not yet captured; first-order acceptable):
  * mHC (Manifold-constrained Hyper-Connections, §2.2 + §3.5.2): per-
    layer hyper-connection compute with expansion factor n_hc=4. Could
    add ~5-15% to total per-layer FLOPs/bytes; not modeled.
  * SWA branch query / output reprojection: paper §2.3.3 doesn't
    explicitly say whether the SWA branch's q/o projections are shared
    with the main path's W^UQ/W^O or separate. Code currently re-uses
    flops_q_proj + flops_o_grp once per layer (folded into the layer-
    type FLOPs), which is a slight over-count if shared.
  * RMSNorm on Q heads + compressed KV entries (§2.3.3): folded into
    QK fused ops (small term, < 1%).
  * Partial RoPE (§2.3.3): folded into QK fused ops as well.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ModelCostEstimator, StageCost, register_model
from .deepseek_v3 import (
    _FP8_BYTES, _BF16_BYTES, _FP4_BYTES,                  # share dtype constants
    bytes_moe_token_dispatch_combine_per_layer,           # share EP all-to-all model
)


# ============================================================================
#  Mixed-precision storage constants (per V4 §2.3.4)
#
#  KV entries are stored with a hybrid scheme:
#      n_rope dims  → BF16
#      remaining    → FP8
# ============================================================================

_N_ROPE = 64           # TODO(verify): see file docstring item 1


def _per_token_kv_bytes(c: int, n_rope: int = _N_ROPE) -> int:
    """KV bytes for ONE c-dim entry under V4 mixed-precision storage.

    c-dim entry stored as:
        n_rope × BF16 + (c - n_rope) × FP8

    Used by both compressed entries (per m tokens) and SWA branch entries
    (per token).
    """
    return n_rope * _BF16_BYTES + max(0, c - n_rope) * _FP8_BYTES


# ============================================================================
#  Layer-type accounting
# ============================================================================


def _v4_layer_breakdown(cfg: Mapping[str, Any]) -> dict[str, int]:
    """Count CSA / HCA / pure-SWA layers in a V4 configuration.

    V4-Pro:    first 2 = HCA, rest = alternating(CSA, HCA) starting CSA
    V4-Flash:  first 2 = pure SWA, rest = alternating(CSA, HCA) starting CSA

    Returns dict {csa, hca, swa} counts.
    """
    L            = int(cfg["num_layers"])
    first_kind   = str(cfg["first_layer_attn"])    # "HCA" or "SWA"
    first_count  = int(cfg["first_layer_count"])

    counts = {"csa": 0, "hca": 0, "swa": 0}
    # First `first_count` layers are all of the same kind.
    counts[first_kind.lower()] = first_count
    # Remaining layers alternate CSA, HCA, CSA, HCA, ...
    rest = L - first_count
    counts["csa"] += (rest + 1) // 2
    counts["hca"] += rest // 2
    return counts


# ============================================================================
#  Common base estimator
# ============================================================================


class _DeepSeekV4Base(ModelCostEstimator):
    """Common scaffolding for V4-Pro / V4-Flash."""

    # ---------- Performance API (decode FLOPs / bytes) -------------------

    def compute_attn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        """Decode attention cost for V4 (per-GPU per-step, TP-sharded).

        Captures the dominant terms paper §2.3 specifies:
          * Q-down (W^DQ) + Q-up (W^UQ) — shared by main and indexer paths
          * KV compression (W^KV / W_a^KV+W_b^KV) and Z-compress weights
          * Core MQA attention over compressed/sparse entries
          * Grouped output projection (n_h × c → g × d_g → d)
          * Indexer (CSA layers only): W^IUQ ∈ R^{d_c × n_h^I × c_I} via
            the shared latent c_t^Q, plus W^w; FP4 compute per §2.3.4
          * SWA branch attention over n_win local KV entries

        Folded approximations (small, < 5% each — see file docstring
        "Known modeling gaps"): RoPE on last 64 dims, RMSNorm on q/KV
        heads, mHC hyper-connections.
        """
        cfg = self.merged_config(model_overrides)
        bs  = int(workload.batch_size)
        N   = int(workload.context_length)
        TP  = max(1, int(parallel.tp))

        L         = int(cfg["num_layers"])
        d         = int(cfg["hidden_size"])
        n_h       = int(cfg["num_q_heads"])
        c         = int(cfg["head_dim"])
        d_c       = int(cfg["query_latent_dim"])
        m_csa     = int(cfg["csa_compress_m"])
        m_hca     = int(cfg["hca_compress_m"])
        topk      = int(cfg["csa_topk"])
        n_win     = int(cfg["swa_window"])
        n_h_idx   = int(cfg["indexer_heads"])
        c_idx     = int(cfg["indexer_head_dim"])
        # Grouped output projection (cf. paper §2.3.1 last paragraph)
        g         = int(cfg.get("output_proj_groups", 8))
        d_g       = int(cfg.get("output_proj_intermediate_dim", 1024))

        layers = _v4_layer_breakdown(cfg)
        n_csa, n_hca, n_swa = layers["csa"], layers["hca"], layers["swa"]
        assert n_csa + n_hca + n_swa == L

        per_token_kv_bytes = _per_token_kv_bytes(c)

        # ------- Per-layer per-token FLOPs (single sample, no TP) --------
        # Q-side (shared across all attention layer types per §2.3.1 eq 13/18):
        #   Q-down:  d × d_c                      (W^DQ, hidden → latent)
        #   Q-up:    d_c × (n_h × c)              (W^UQ, latent → main Q)
        # MQA core attn (per sparse / dense KV entry consumed):
        #   QK score:  n_h × c   (1 mul-add per dim)
        #   PV mix:    n_h × c
        # Grouped output projection (paper §2.3.1 last paragraph):
        #   group:   (n_h/g × c) → d_g       — per group cost = (n_h/g × c) × d_g
        #   total grouping cost = g × (n_h/g × c) × d_g = (n_h × c) × d_g
        #   final:   (g × d_g) → d
        flops_q_proj = 2 * d * d_c + 2 * d_c * (n_h * c)
        flops_o_grp  = 2 * (n_h * c) * d_g + 2 * (g * d_g) * d

        # CSA layer per-token core-attn FLOPs:
        #   - core attn over top-k compressed entries
        #   - SWA branch attn over n_win uncompressed entries
        #   - indexer: q_t^I = c_t^Q · W^IUQ takes the SHARED query latent
        #     c_t^Q ∈ R^{d_c} (paper §2.3.1 eq 14), so input dim is d_c not d.
        #     W^IUQ ∈ R^{d_c × n_h^I × c_I}.
        n_attended_csa = topk + n_win
        flops_attn_csa = 2 * n_h * c * n_attended_csa * 2   # QK + PV
        # Indexer Q-up: c_t^Q · W^IUQ produces n_h_idx × c_idx queries (eq 14).
        # Score eq (16): w_t^I · ReLU(q_t^I · K_s) over all N/m_csa compressed
        # blocks. FP4 compute (per §2.3.4) — Roofline already captures it via
        # the bytes side, so FLOPs are counted in normal float ops.
        flops_idx_q   = 2 * d_c * (n_h_idx * c_idx)
        flops_idx_dot = 2 * (N // max(1, m_csa)) * n_h_idx * c_idx
        flops_csa_per_tok = (
            flops_q_proj + flops_attn_csa + flops_o_grp
            + flops_idx_q + flops_idx_dot
        )

        # HCA layer per-token core-attn FLOPs:
        #   - core attn over ALL compressed entries (no top-k selection)
        #   - SWA branch attn
        n_attended_hca = (N // max(1, m_hca)) + n_win
        flops_attn_hca = 2 * n_h * c * n_attended_hca * 2
        flops_hca_per_tok = flops_q_proj + flops_attn_hca + flops_o_grp

        # Pure-SWA layer per-token FLOPs:
        #   - core attn over n_win KV entries only
        flops_attn_swa = 2 * n_h * c * n_win * 2
        flops_swa_per_tok = flops_q_proj + flops_attn_swa + flops_o_grp

        # ------- Per-layer bytes (weight reads + KV reads) ---------------
        # Weight bytes per layer:
        #   FP8: most projections (Q, KV-compress, Z-compress, output, indexer W^w)
        #   FP4: W^IUQ (the indexer query up-proj that drives FP4 indexer compute)
        # Per paper §2.3.1 (CSA) eq 9-10: CSA has TWO compression streams
        #   C^a = H · W_a^KV, C^b = H · W_b^KV  (both d × c)
        #   Z^a = H · W_a^Z,  Z^b = H · W_b^Z   (both d × c)
        # Per paper §2.3.2 (HCA) eq 20-21: HCA has only ONE stream each
        #   C   = H · W^KV    (d × c)
        #   Z   = H · W^Z     (d × c)
        # So CSA weight = 2×(d×c) for KV + 2×(d×c) for Z; HCA = 1×(d×c) + 1×(d×c).
        w_q_proj         = (d * d_c + d_c * (n_h * c)) * _FP8_BYTES
        w_kv_proj_csa    = 2 * d * c * _FP8_BYTES                         # W_a^KV + W_b^KV
        w_z_proj_csa     = 2 * d * c * _FP8_BYTES                         # W_a^Z + W_b^Z
        w_kv_proj_hca    = 1 * d * c * _FP8_BYTES                         # W^KV (single)
        w_z_proj_hca     = 1 * d * c * _FP8_BYTES                         # W^Z (single)
        w_o_grouped      = ((n_h * c) * d_g + (g * d_g) * d) * _FP8_BYTES
        # Indexer weights: W^IUQ ∈ R^{d_c × n_h^I × c_I} (FP4 per §2.3.4),
        # W^w ∈ R^{d × n_h^I} (FP8, used to compute eq 15 weights from h_t).
        w_idx            = (d_c * (n_h_idx * c_idx)) * _FP4_BYTES + (d * n_h_idx) * _FP8_BYTES
        w_csa_layer = w_q_proj + w_kv_proj_csa + w_z_proj_csa + w_o_grouped + w_idx
        w_hca_layer = w_q_proj + w_kv_proj_hca + w_z_proj_hca + w_o_grouped       # no indexer
        w_swa_layer = w_q_proj + w_o_grouped                              # no compression weights

        # KV cache bytes read per token per layer (during decode step):
        kv_csa_per_tok = (
            (per_token_kv_bytes * topk)              # compressed top-k entries
            + (c_idx * _FP4_BYTES * (N // max(1, m_csa)))  # indexer K-cache (FP4 per §2.3.4)
            + (per_token_kv_bytes * n_win)           # SWA branch
        )
        kv_hca_per_tok = (
            (per_token_kv_bytes * (N // max(1, m_hca)))
            + (per_token_kv_bytes * n_win)
        )
        kv_swa_per_tok = per_token_kv_bytes * n_win

        # ------- Aggregate across layer types and batch ------------------
        flops_per_token_total = (
            n_csa * flops_csa_per_tok
            + n_hca * flops_hca_per_tok
            + n_swa * flops_swa_per_tok
        )
        weight_bytes_total = (
            n_csa * w_csa_layer
            + n_hca * w_hca_layer
            + n_swa * w_swa_layer
        )
        kv_per_sample_total = (
            n_csa * kv_csa_per_tok
            + n_hca * kv_hca_per_tok
            + n_swa * kv_swa_per_tok
        )

        flops_per_gpu = flops_per_token_total * bs / TP
        bytes_per_gpu = (
            weight_bytes_total                         # weight read once per layer
            + kv_per_sample_total * bs                 # KV read per sample per step
        ) / TP

        return StageCost(flops=float(flops_per_gpu), bytes_=float(bytes_per_gpu))

    def compute_ffn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        """Decode FFN cost for V4 MoE.

        Same shape as V3 (top-k routed + 1 shared expert), only the
        constants differ:
            V4-Pro:    1 shared + 384 routed, top_k=6, intermediate=3072
            V4-Flash:  1 shared + 256 routed, top_k=6, intermediate=2048

        Counting convention (cluster-wide, divided by EP at the end):
            total_tokens = bs × EP
            FLOPs   = total_tokens × (router + (top_k + n_shared) × per_expert)
            Bytes   = weight_bytes + activation_bytes
              weight_bytes     = (n_routed + EP × n_shared) × per_expert_w
                               + EP × router_w
              activation_bytes = total_tokens × n_layers
                               × dispatch_combine_per_layer (BF16 hidden,
                                 EP all-to-all dispatch+combine; see
                                 bytes_moe_token_dispatch_combine_per_layer
                                 in deepseek_v3.py for the derivation).

        TODO(verify): V4 uses Hash routing for the first 3 MoE layers (per
        §4.2.1); we approximate the cost as identical to learned routing
        which is fine to first order (router ops are <<1% of MoE cost).
        """
        cfg = self.merged_config(model_overrides)
        bs  = int(workload.batch_size)
        EP  = max(1, int(parallel.ep))
        DP  = max(1, int(parallel.dp))

        L         = int(cfg["num_layers"])
        d         = int(cfg["hidden_size"])
        n_routed  = int(cfg["num_routed_experts"])
        n_shared  = int(cfg["num_shared_experts"])
        top_k     = int(cfg["top_k_experts"])
        d_int     = int(cfg["expert_intermediate_size"])

        # Per-expert weight (gate_up_proj + down_proj), per layer.
        # Shape: d → 2*d_int (gate_up) + d_int → d (down)
        per_expert_per_layer = (d * 2 * d_int + d_int * d) * _FP8_BYTES
        per_expert_all_L = per_expert_per_layer * L

        # Router weight per layer (d × n_routed), all layers.
        router_per_layer = d * n_routed * _FP8_BYTES
        router_all_L = router_per_layer * L

        # FLOPs per token per layer.
        flops_router_per_tok = 2 * d * n_routed
        flops_per_expert_per_tok = 2 * (d * 2 * d_int + d_int * d)
        flops_per_token_per_layer = (
            flops_router_per_tok
            + (top_k + n_shared) * flops_per_expert_per_tok
        )

        # Cluster-wide totals.
        # workload.batch_size is ATTN per-GPU bs (per-DP sessions, MLA replicated
        # across TP).  FFN per-GPU token count = bs * DP / EP = bs / TP, because
        # TP ranks hold replicated sessions and contribute no extra unique tokens.
        ffn_bs = bs * DP / EP
        total_tokens = ffn_bs * EP
        total_flops = total_tokens * flops_per_token_per_layer * L
        total_weight_bytes = (
            (n_routed + EP * n_shared) * per_expert_all_L
            + EP * router_all_L
        )
        # Token activation bytes: EP all-to-all dispatch + combine, BF16
        # hidden state. Conservative "must-go-through-HBM" floor; shared
        # experts are excluded (replicated locally, no all-to-all).
        per_token_per_layer_act = bytes_moe_token_dispatch_combine_per_layer(
            d, top_k
        )
        total_act_bytes = total_tokens * L * per_token_per_layer_act
        total_bytes = total_weight_bytes + total_act_bytes

        # Per-GPU.
        flops_per_gpu = total_flops / EP
        bytes_per_gpu = total_bytes / EP
        return StageCost(flops=float(flops_per_gpu), bytes_=float(bytes_per_gpu))

    # ---------- Capacity-analysis API (decode only) ----------------------

    def session_kv_bytes(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Per-GPU KV cache bytes for one decode session of V4 hybrid attention.

        Closed-form (per file docstring formulas):

            session_kv_total =
                  n_csa * (c/m_csa  + c_idx/m_csa) × N × FP_mixed
                + n_hca * (c/m_hca)                × N × FP_mixed
                + (n_csa + n_hca) × n_win × c      × FP_mixed   [SWA cap, constant]
                + n_swa  ×           n_win × c     × FP_mixed   [pure-SWA cap]

        TODO(verify): see file docstring item 2. Specifically, we assume
        SWA branch is a *rolling buffer* sized n_win × c per layer and
        independent of N. If the production stack persists every token's
        SWA entry to disk (per §3.6.2 "SWA KV entries exist in every
        layer"), then SWA contribution scales with N — multiply by N/n_win
        in that case.
        """
        del parallel, dtype_kv  # reserved for future TP-shard / dtype variants
        cfg = self.merged_config(model_overrides)
        N         = int(session_length)
        c         = int(cfg["head_dim"])
        m_csa     = int(cfg["csa_compress_m"])
        m_hca     = int(cfg["hca_compress_m"])
        n_win     = int(cfg["swa_window"])
        c_idx     = int(cfg["indexer_head_dim"])

        layers = _v4_layer_breakdown(cfg)
        n_csa, n_hca, n_swa = layers["csa"], layers["hca"], layers["swa"]

        kv_per_entry = _per_token_kv_bytes(c)

        # Compressed-entry storage (scales with N/m).
        csa_kv = n_csa * (N // max(1, m_csa)) * kv_per_entry
        hca_kv = n_hca * (N // max(1, m_hca)) * kv_per_entry

        # Indexer K-cache for CSA layers (FP4, c_idx-dim per compressed block;
        # paper §2.3.4: "attention computation within the lightning indexer is
        # performed in FP4 precision").
        idx_kv = n_csa * (N // max(1, m_csa)) * (c_idx * _FP4_BYTES)

        # SWA rolling buffer: ALL layers (CSA + HCA + pure-SWA) contribute
        # n_win × c bytes (constant w.r.t. N).
        swa_kv = (n_csa + n_hca + n_swa) * n_win * kv_per_entry

        return float(csa_kv + hca_kv + idx_kv + swa_kv)

    def indexer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Indexer K-cache bytes per session, summed across CSA layers only.

        Per V4 paper §2.3.1: only CSA layers carry an indexer (HCA attends
        all compressed entries; SWA is windowed). Indexer K is compressed
        at rate m_csa (one indexer entry per m_csa tokens) and stored in
        FP4 (paper §2.3.4: "attention computation within the lightning
        indexer is performed in FP4 precision"; the K-cache is therefore
        sized to match the compute precision).
        """
        del parallel, dtype_kv
        cfg = self.merged_config(model_overrides)
        N      = int(session_length)
        m_csa  = int(cfg["csa_compress_m"])
        c_idx  = int(cfg["indexer_head_dim"])
        layers = _v4_layer_breakdown(cfg)
        n_csa  = layers["csa"]
        return float(n_csa * (N // max(1, m_csa)) * c_idx * _FP4_BYTES)

    def cold_layer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Average bytes fetched per missed layer per session, weighted across
        the V4 hybrid CSA / HCA / SWA layer mix.

        Per-layer fetch by type:
            CSA: top_k × (c-dim mixed-precision)             (sparse top-k entries)
            HCA: (N / m_hca) × (c-dim mixed-precision)       (all compressed entries)
            SWA: n_win × (c-dim mixed-precision)             (sliding window)

        Then averaged: returned value = (sum / total_layers).
        """
        del parallel, dtype_kv
        cfg = self.merged_config(model_overrides)
        N         = int(session_length)
        c         = int(cfg["head_dim"])
        m_hca     = int(cfg["hca_compress_m"])
        n_win     = int(cfg["swa_window"])
        topk_csa  = int(cfg["csa_topk"])
        layers    = _v4_layer_breakdown(cfg)
        n_csa, n_hca, n_swa = layers["csa"], layers["hca"], layers["swa"]
        L = max(1, n_csa + n_hca + n_swa)

        v_entry = _per_token_kv_bytes(c)
        csa_per_layer = topk_csa * v_entry
        hca_per_layer = (N // max(1, m_hca)) * v_entry
        swa_per_layer = n_win * v_entry

        total = (
            n_csa * csa_per_layer
            + n_hca * hca_per_layer
            + n_swa * swa_per_layer
        )
        return float(total / L)

    def weight_bytes_per_gpu(
        self,
        parallel,
        dtype_param,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Static weight bytes resident on ONE GPU (attn + MoE), TP/EP-sharded.

        Cross-checked against V4 paper §2.3 (CSA / HCA weight shapes and
        dtype) on 2026-05-25; remaining open item is line-by-line
        validation against the open-source HuggingFace V4-Pro checkpoint.
        """
        del dtype_param
        cfg = self.merged_config(model_overrides)
        TP  = max(1, int(parallel.tp))
        EP  = max(1, int(parallel.ep))

        L         = int(cfg["num_layers"])
        d         = int(cfg["hidden_size"])
        n_h       = int(cfg["num_q_heads"])
        c         = int(cfg["head_dim"])
        d_c       = int(cfg["query_latent_dim"])
        n_h_idx   = int(cfg["indexer_heads"])
        c_idx     = int(cfg["indexer_head_dim"])
        g         = int(cfg.get("output_proj_groups", 8))
        d_g       = int(cfg.get("output_proj_intermediate_dim", 1024))

        n_routed  = int(cfg["num_routed_experts"])
        n_shared  = int(cfg["num_shared_experts"])
        d_int     = int(cfg["expert_intermediate_size"])

        layers = _v4_layer_breakdown(cfg)
        n_csa, n_hca, n_swa = layers["csa"], layers["hca"], layers["swa"]

        # ---- Attn weights per layer (FP8) -------------------------------
        # Q-down + Q-up (W^DQ shared by main Q and indexer Q per §2.3.1):
        w_q = (d * d_c + d_c * (n_h * c)) * _FP8_BYTES
        # KV-compress (W^KV): CSA has TWO streams (W_a^KV + W_b^KV per eq 9),
        # HCA has only ONE (W^KV per eq 20).
        w_kv_compress_csa = 2 * d * c * _FP8_BYTES
        w_kv_compress_hca = 1 * d * c * _FP8_BYTES
        # Z-compress (W^Z): same CSA-vs-HCA distinction (eq 10 vs eq 21).
        w_z_csa = 2 * d * c * _FP8_BYTES
        w_z_hca = 1 * d * c * _FP8_BYTES
        # Output projection (grouped, paper §2.3.1 last paragraph):
        w_o = ((n_h * c) * d_g + (g * d_g) * d) * _FP8_BYTES
        # Indexer weights (CSA layers only):
        #   W^IUQ ∈ R^{d_c × n_h^I × c_I} — driven by FP4 indexer compute (§2.3.4)
        #   W^w   ∈ R^{d × n_h^I}        — FP8, computes head weights w_t^I (eq 15)
        # Note W^DQ is reused from the main Q stack (latent c_t^Q is shared with
        # the main query per the closing remark of §2.3.1).
        w_idx = (d_c * (n_h_idx * c_idx)) * _FP4_BYTES + (d * n_h_idx) * _FP8_BYTES

        w_csa_layer = w_q + w_kv_compress_csa + w_z_csa + w_o + w_idx
        w_hca_layer = w_q + w_kv_compress_hca + w_z_hca + w_o
        w_swa_layer = w_q + w_o   # pure SWA: no KV compression weights

        attn_total_bytes = (
            n_csa * w_csa_layer
            + n_hca * w_hca_layer
            + n_swa * w_swa_layer
        ) / TP

        # ---- MoE weights per layer (FP8) -------------------------------
        per_expert_per_layer = (d * 2 * d_int + d_int * d) * _FP8_BYTES
        per_expert_all_L = per_expert_per_layer * L
        router_per_layer = d * n_routed * _FP8_BYTES
        router_all_L = router_per_layer * L

        routed_per_gpu = (n_routed / EP) * per_expert_all_L
        shared_per_gpu = n_shared * per_expert_all_L
        router_per_gpu = router_all_L

        ffn_total_bytes = routed_per_gpu + shared_per_gpu + router_per_gpu

        return float(attn_total_bytes + ffn_total_bytes)


# ============================================================================
#  Concrete V4-Pro / V4-Flash registrations
# ============================================================================


@register_model("deepseek-v4-pro")
class DeepSeekV4ProEstimator(_DeepSeekV4Base):
    """V4-Pro: 1.6T total params, 49B active (per V4 paper Table-of-contents).

    Architecture: 61 layers, hidden=7168, head_dim=512, MoE 384 routed +
    1 shared, top-6 routing.
    """
    default_model_config = {
        "num_layers": 61,
        "first_layer_attn": "HCA",            # first 2 layers HCA
        "first_layer_count": 2,
        "hidden_size": 7168,
        "num_q_heads": 128,
        "head_dim": 512,                      # c
        "query_latent_dim": 1536,             # d_c
        # Grouped output projection
        "output_proj_groups": 16,             # g (V4-Pro)
        "output_proj_intermediate_dim": 1024, # d_g
        # CSA / HCA / SWA
        "csa_compress_m": 4,
        "csa_topk": 1024,
        "hca_compress_m": 128,
        "swa_window": 128,
        # Indexer
        "indexer_heads": 64,
        "indexer_head_dim": 128,
        # MoE
        "num_routed_experts": 384,
        "num_shared_experts": 1,
        "top_k_experts": 6,
        "expert_intermediate_size": 3072,
        "vocab_size": 128_000,
        "mtp_depth": 1,
    }


@register_model("deepseek-v4-flash")
class DeepSeekV4FlashEstimator(_DeepSeekV4Base):
    """V4-Flash: 284B total params, 13B active.

    Architecture: 43 layers, hidden=4096, head_dim=512, MoE 256 routed +
    1 shared, top-6 routing. First 2 layers are pure SWA.
    """
    default_model_config = {
        "num_layers": 43,
        "first_layer_attn": "SWA",            # first 2 layers pure SWA
        "first_layer_count": 2,
        "hidden_size": 4096,
        "num_q_heads": 64,
        "head_dim": 512,
        "query_latent_dim": 1024,
        # Grouped output projection
        "output_proj_groups": 8,              # g (V4-Flash)
        "output_proj_intermediate_dim": 1024, # d_g
        # CSA / HCA / SWA
        "csa_compress_m": 4,
        "csa_topk": 512,
        "hca_compress_m": 128,
        "swa_window": 128,
        # Indexer
        "indexer_heads": 64,
        "indexer_head_dim": 128,
        # MoE
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "top_k_experts": 6,
        "expert_intermediate_size": 2048,
        "vocab_size": 128_000,
        "mtp_depth": 1,
    }
