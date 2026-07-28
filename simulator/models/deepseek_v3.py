"""DeepSeek-V3 / V3.2 cost estimator: MLA + DeepSeekMoE.

Currently only the **decode attention** path is modelled. Prefill and FFN
are placeholders.

Attention decomposition (per layer, per token, per sample, before TP/EP/bs):

    Six projections (constant in N). NoPE and RoPE up-projections are
    merged into the same matmul (single weight matrix produces
    [NoPE; RoPE] concatenated output), so the projection widens by d_h^R:
        flops_q_down_proj   = 2 * d * d_q_latent
        flops_q_up_proj     = 2 * d_q_latent * n_h * (d_h + d_h_rope)
                                                              (W_UQ ∪ W_QR)
        flops_kv_down_proj  = 2 * d * (d_kv_latent + d_h_rope)
                                                              (W_DKV ∪ W_KR)
        flops_k_up_proj     = 2 * d_kv_latent * n_h * d_h
        flops_v_up_proj     = 2 * n_h * d_kv_latent * d_h          (V up-projection)
        flops_o_proj        = 2 * n_h * d_h * d                    (Output projection)

    W_QR (Q RoPE proj, ~1.54e9 / token / 61-layer model) and W_KR (K RoPE
    proj, ~5.6e7 / token) are now counted via the (d_h + d_h_rope) and
    (d_kv_latent + d_h_rope) widening above — consistent with the bytes
    side, which has always used the same widened dims. See the W_QR / W_KR
    section in the per-helper docstrings.

    Main attention core (always runs):
        flops_attn_qk_dot       = 2 * n_h * n_eff * (d_kv_latent + d_h_rope)
        flops_attn_weighted_v   = 2 * n_h * n_eff * d_kv_latent

    DSA path (only when DSA active; ADDITIVE, does NOT replace main attn):
        Indexer first scores all N tokens to produce top-k indices, then main
        attention runs over those top-k tokens. So when DSA is active, both
        the 4 DSA components AND main attn QK/V are counted (with n_eff=top_k
        for main attn).

        flops_dsa_q_idx_proj       = 2 * d_q_latent * q_idx_heads * d_idx
        flops_dsa_k_idx_proj       = 2 * d * d_idx
        flops_dsa_indexer_scoring  = 2 * q_idx_heads * d_idx * N
        flops_dsa_head_weight      = 2 * d * q_idx_heads

    n_eff selection:
        DSA active (dsa_topk is set AND N >= dsa_topk) -> n_eff = dsa_topk
        else                                            -> n_eff = N

    Per-GPU per-step total:
        F_attn_per_gpu = batch_size * num_layers * (sum of above) / TP
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ModelCostEstimator, StageCost, register_model


# ============================================================================
#  Per-layer per-token FLOPs functions (single sample, no parallel divisor).
#  Each function returns a plain int. Caller multiplies by L, batch, divides
#  by TP at the orchestration layer.
# ============================================================================


# ----- Six projections (constant in N) --------------------------------------
# W_QR (Q RoPE proj) and W_KR (K RoPE proj) are folded into W_UQ and W_DKV
# respectively via output-dim widening — same convention as the bytes side
# (see bytes_q_up_proj_weight / bytes_kv_down_proj_weight). The fused
# matmuls produce [NoPE; RoPE] concatenated outputs in one shot:
#     W_UQ ∪ W_QR : d_q_latent → n_h * (d_h + d_h_rope)
#     W_DKV ∪ W_KR: d          → (d_kv_latent + d_h_rope)
#
# Extra cost from folding (per layer per token, vs NoPE-only):
#     W_QR-equiv: 2 * d_q_latent * n_h * d_h_rope  ~= 25.2M  (× 61 = 1.54e9)
#     W_KR-equiv: 2 * d          *       d_h_rope  ~= 0.92M  (× 61 = 5.60e7)
# Together ~1.59e9, ~7.5% of the prior NoPE-only 5-projection total (2.12e10).
#
# Six-projection constant totals (V3 spec, × 61 layers):
#     before folding (NoPE only):                 ≈ 21.24e9
#     after  folding (NoPE + RoPE via W_QR/W_KR): ≈ 22.83e9
#
# Note: W_KR is single-head (MQA-style decoupled rotary key); folding it
# into W_DKV's output adds the d_h_rope output dim once (no n_h factor),
# matching the standalone single-head W_KR cost.


def flops_q_down_proj(d: int, d_q_latent: int) -> int:
    """W_DQ : R^d -> R^{d_c'}  (Q down-projection).

    Formula: 2 * d * d_q_latent

    Sanity (V3, d=7168, d_q_latent=1536, × 61 layers):
        2 * 7168 * 1536 * 61 = 1,343,225,856  ✓
    """
    return 2 * d * d_q_latent


def flops_q_up_proj(d_q_latent: int, n_h: int, d_h: int, d_h_rope: int) -> int:
    """W_UQ ∪ W_QR : R^{d_c'} -> R^{n_h * (d_h + d_h^R)}  (Q up-projection,
    NoPE + RoPE folded into one matmul).

    Formula: 2 * d_q_latent * n_h * (d_h + d_h_rope)

    Sanity (V3, d_q_latent=1536, n_h=128, d_h=128, d_h^R=64, × 61 layers):
        2 * 1536 * 128 * (128 + 64) * 61 = 4,605,345,792  ✓
        breakdown:
          NoPE (W_UQ): 2 * 1536 * 128 * 128 * 61 = 3,070,230,528
          RoPE (W_QR): 2 * 1536 * 128 *  64 * 61 = 1,535,115,264
    """
    return 2 * d_q_latent * n_h * (d_h + d_h_rope)


def flops_kv_down_proj(d: int, d_kv_latent: int, d_h_rope: int) -> int:
    """W_DKV ∪ W_KR : R^d -> R^{d_c + d_h^R}  (KV down-projection,
    KV latent + RoPE key folded into one matmul).

    Formula: 2 * d * (d_kv_latent + d_h_rope)

    Note: W_KR is MQA-style single-head — there is no n_h factor on the
    RoPE part. Folding it into W_DKV's output dim correctly adds only the
    d_h_rope columns once.

    Sanity (V3, d=7168, d_kv_latent=512, d_h^R=64, × 61 layers):
        2 * 7168 * (512 + 64) * 61 = 503,709,696  ✓
        breakdown:
          KV latent (W_DKV): 2 * 7168 * 512 * 61 = 447,741,952
          RoPE key  (W_KR ): 2 * 7168 *  64 * 61 =  55,967,744
    """
    return 2 * d * (d_kv_latent + d_h_rope)


def flops_k_up_proj(d_kv_latent: int, n_h: int, d_h: int) -> int:
    """W_UK : R^{d_c} -> R^{n_h * d_h}  (K up-projection).

    Formula: 2 * d_kv_latent * n_h * d_h

    Sanity (V3, d_kv_latent=512, n_h=128, d_h=128, × 61 layers):
        2 * 512 * 128 * 128 * 61 = 1,023,410,176  ✓

    NOTE: V2/V3 papers show this can be absorbed into W_UQ in inference
    (attention kernels that merge the two). We count it explicitly here
    (non-absorbed path), matching the public V3 / V3.2 weight-count
    references.
    """
    return 2 * d_kv_latent * n_h * d_h


def flops_v_up_proj(n_h: int, d_kv_latent: int, d_h: int) -> int:
    """W_UV : R^{d_c} -> R^{n_h * d_h}  (V up-projection).

    Formula: 2 * n_h * d_kv_latent * d_h

    Sanity (V3, n_h=128, d_kv_latent=512, d_h=128, × 61 layers):
        2 * 128 * 512 * 128 * 61 = 1,023,410,176  ✓
    """
    return 2 * n_h * d_kv_latent * d_h


def flops_o_proj(n_h: int, d_h: int, d: int) -> int:
    """W_O : R^{n_h * d_h} -> R^d  (Output projection).

    Formula: 2 * n_h * d_h * d

    Sanity (V3, n_h=128, d_h=128, d=7168, × 61 layers):
        2 * 128 * 128 * 7168 * 61 = 14,327,742,464  ✓
    """
    return 2 * n_h * d_h * d


# ----- Main attention core --------------------------------------------------


def flops_attn_qk_dot(n_h: int, d_kv_latent: int, d_h_rope: int, n_eff: int) -> int:
    """Main attn Q . K^T over n_eff tokens. Always runs.

    n_eff selection (handled by caller):
        DSA active (V3.2, N>=2048): n_eff = top_k = 2048   (sparse: attend to selected top-k only)
        else:                       n_eff = N              (dense: attend to all history)

    Formula: 2 * n_h * n_eff * (d_kv_latent + d_h_rope)

    Sanity (V3, n_h=128, d_c+d_h^R=576, ×61 layers):
        # Dense (DSA off, n_eff = N):
        N=1024:    2*128*576*1024*61 =   9,210,691,584  ✓
        N=9216:    2*128*576*9216*61 =  82,896,224,256  ✓
        # DSA on (n_eff pinned to top_k=2048, constant in N):
        n_eff=2048: 2*128*576*2048*61 = 18,421,383,168  ✓
    """
    return 2 * n_h * n_eff * (d_kv_latent + d_h_rope)


def flops_attn_weighted_v(n_h: int, d_kv_latent: int, n_eff: int) -> int:
    """(softmax) . V over n_eff tokens; V dim = d_kv_latent (512 in V3).

    Formula: 2 * n_h * d_kv_latent * n_eff

    Sanity (V3.2, n_h=128, d_kv_latent=512, ×61 layers):
        N=1024  (DSA off, n_eff=N=1024):       2*128*512*1024*61 =  8,187,281,408  ✓
        N>=2048 (DSA on,  n_eff=top_k=2048):   2*128*512*2048*61 = 16,374,562,816  ✓
    """
    return 2 * n_h * d_kv_latent * n_eff


# ----- DSA path (only active when dsa_topk is not None AND N >= dsa_topk) ----
# DSA replaces main attn Q.K^T with a lightning indexer + top-k selection.
# The following four components together constitute the DSA-added FLOPs.


def flops_dsa_q_idx_proj(d_q_latent: int, q_idx_heads: int, d_idx: int) -> int:
    """W_DSA_q_up: R^{d_c'} -> R^{q_idx_heads * d_idx}  (per layer per token).

    Formula: 2 * d_q_latent * q_idx_heads * d_idx

    Sanity (V3.2, d_c'=1536, q_idx_heads=64, d_idx=128, ×61 layers):
        2 * 1536 * 64 * 128 = 25,165,824  per layer
        × 61 = 1,535,115,264  ✓
    """
    return 2 * d_q_latent * q_idx_heads * d_idx


def flops_dsa_k_idx_proj(d: int, d_idx: int) -> int:
    """W_DSA_k: R^d -> R^{d_idx}  (shared across q_idx_heads).

    Formula: 2 * d * d_idx

    Sanity (V3.2, d=7168, d_idx=128, ×61 layers):
        2 * 7168 * 128 = 1,835,008  per layer
        × 61 = 111,935,488  ✓
    """
    return 2 * d * d_idx


def flops_dsa_indexer_scoring(q_idx_heads: int, d_idx: int, n_total: int) -> int:
    """Indexer Q . K^T scoring over all N historical tokens.

    Formula: 2 * q_idx_heads * d_idx * n_total

    This is the N-linear core of the DSA path; its output top-k indices are
    then fed to the main attention weighted-V step.

    Sanity (V3.2, q_idx_heads=64, d_idx=128, ×61 layers):
        slope (per N): 2 * 64 * 128 * 61 = 1,000,009
        N=9216  -> 9,210,691,584   ✓
        N=30720 -> 30,702,305,280  ✓
    """
    return 2 * q_idx_heads * d_idx * n_total


def flops_dsa_head_weight(d: int, q_idx_heads: int) -> int:
    """Per-indexer-head scalar weight W_DSA_w: R^d -> R^{q_idx_heads}.

    Corresponds to the `w_{t,j}` scalar in V3.2 indexer formula (1):
        I_{t,s} = sum_j w_{t,j} * ReLU(q^I_{t,j} . k^I_s)

    Formula: 2 * d * q_idx_heads

    Sanity (V3.2, d=7168, q_idx_heads=64, ×61 layers):
        2 * 7168 * 64 = 917,504  per layer
        × 61 = 55,967,744  ✓
    """
    return 2 * d * q_idx_heads


# ============================================================================
#  Per-layer bytes functions (HBM bytes, single sample, no parallel divisor).
#  Bytes are split into:
#     (a) weight bytes  -- shared across the batch (NOT multiplied by bs)
#     (b) KV bytes      -- per-sample (multiplied by bs in caller)
#     (c) activation    -- per-sample (multiplied by bs in caller)
#
#  All weight functions assume FP8 storage (1 byte/element). The dtype is
#  hard-coded to keep formulas readable; switch to dtype_param later if needed.
#
#  W_QR / W_KR (Q/K RoPE projections) are folded into W_UQ / W_DKV on BOTH
#  the FLOPs and bytes sides via output-dim widening — the up-dim is
#  (d_h + d_h^R) for Q and (d_c + d_h^R) for KV. See header comment above
#  the flops_* helpers for the breakdown.
# ============================================================================


_FP8_BYTES = 1   # FP8 = 1 byte/element (weights, indexer K cache for V3/V3.2)
_FP4_BYTES = 0.5 # FP4 = 0.5 byte/element (V4 indexer K-cache and indexer compute,
                 # per V4 paper §2.3.4: "attention computation within the lightning
                 # indexer is performed in FP4 precision"). Only used by V4; kept
                 # in this module so V3.2 / V4 share dtype constants.
_BF16_BYTES = 2  # BF16 = 2 byte/element (main MLA KV cache, per V3.2 deployment)


# ----- Six projection weight bytes (always present, constant in N) ----------


def bytes_q_down_proj_weight(d: int, d_q_latent: int) -> int:
    """W_DQ weight: d * d_q_latent  bytes (FP8).

    Sanity (V3.2, d=7168, d_q_latent=1536, ×61 layers):
        7168 * 1536 * 61 = 671,612,928  ✓
    """
    return d * d_q_latent * _FP8_BYTES


def bytes_q_up_proj_weight(d_q_latent: int, n_h: int, d_h: int, d_h_rope: int) -> int:
    """W_UQ weight (NoPE + RoPE parts): d_q_latent * n_h * (d_h + d_h^R)  bytes.

    Sanity (V3.2, d_q_latent=1536, n_h=128, d_h=128, d_h^R=64, ×61):
        1536 * 128 * (128 + 64) * 61 = 1536 * 128 * 192 * 61 = 2,302,672,896  ✓
    """
    return d_q_latent * n_h * (d_h + d_h_rope) * _FP8_BYTES


def bytes_kv_down_proj_weight(d: int, d_kv_latent: int, d_h_rope: int) -> int:
    """W_DKV weight (KV latent + RoPE parts): d * (d_kv_latent + d_h^R)  bytes.

    Sanity (V3.2, d=7168, d_c+d_h^R=576, ×61):
        7168 * 576 * 61 = 251,854,848  ✓
    """
    return d * (d_kv_latent + d_h_rope) * _FP8_BYTES


def bytes_k_up_proj_weight(d_kv_latent: int, n_h: int, d_h: int) -> int:
    """W_UK weight: d_kv_latent * n_h * d_h  bytes.

    Sanity (V3.2, d_kv_latent=512, n_h=128, d_h=128, ×61):
        512 * 128 * 128 * 61 = 511,705,088  ✓
    """
    return d_kv_latent * n_h * d_h * _FP8_BYTES


def bytes_v_up_o_proj_weight(n_h: int, d_kv_latent: int, d_h: int, d: int) -> int:
    """Combined W_UV + W_O weight bytes (V up-projection + output projection).

        W_UV: d_kv_latent * n_h * d_h
        W_O:  n_h * d_h * d

    Sanity (V3.2, ×61):
        W_UV: 512 * 128 * 128 * 61 = 511,705,088
        W_O:  128 * 128 * 7168 * 61 = 7,163,871,232
        sum  = 7,675,576,320  ✓
    """
    w_uv = d_kv_latent * n_h * d_h
    w_o  = n_h * d_h * d
    return (w_uv + w_o) * _FP8_BYTES


# ----- DSA additional weight bytes (only when DSA is active) ----------------


def bytes_dsa_q_idx_proj_weight(d_q_latent: int, q_idx_heads: int, d_idx: int) -> int:
    """W_DSA_q_up weight: d_q_latent * q_idx_heads * d_idx  bytes.

    Sanity (V3.2, d_c'=1536, q_idx_heads=64, d_idx=128, ×61):
        1536 * 64 * 128 * 61 = 767,557,632  ✓
    """
    return d_q_latent * q_idx_heads * d_idx * _FP8_BYTES


def bytes_dsa_k_idx_proj_weight(d: int, d_idx: int) -> int:
    """W_DSA_k weight: d * d_idx  bytes.

    Sanity (V3.2, d=7168, d_idx=128, ×61):
        7168 * 128 * 61 = 55,967,744  ✓
    """
    return d * d_idx * _FP8_BYTES


def bytes_dsa_head_weight(d: int, q_idx_heads: int) -> int:
    """W_DSA_w weight: d * q_idx_heads  bytes.

    Sanity (V3.2, d=7168, q_idx_heads=64, ×61):
        7168 * 64 * 61 = 27,983,872  ✓
    """
    return d * q_idx_heads * _FP8_BYTES


# ----- KV cache bytes (per-sample, multiplied by batch in caller) -----------
# These represent how much KV cache must be READ per decode step. Caller
# multiplies by batch_size and divides by TP.


def bytes_kv_cache_read(d_kv_latent: int, d_h_rope: int, n_eff: int) -> int:
    """Main attn KV cache read: n_eff tokens × (d_kv_latent + d_h_rope) × L  bytes.

    Mixed-precision storage convention (per V4 paper §2.3.4):
        * KV latent dimensions  → FP8  (1 B/elem)
        * RoPE-key dimensions   → BF16 (2 B/elem; rotary needs higher
                                        precision to preserve relative-
                                        position fidelity)

    n_eff = min(N, top_k) when DSA is active; n_eff = N otherwise.

    The FP8-latent + BF16-RoPE mixed-precision storage has been verified
    against the V3.2 HuggingFace reference implementation
    (https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference).
    V4 paper §2.3.4 documents the same scheme.

    Sanity (V3.2, d_c=512, d_h^R=64, ×61 layers, mixed FP8 latent + BF16 RoPE):
        per-token-per-layer KV = 512×1 + 64×2 = 640 B/token/layer
        N=2048 (DSA on, n_eff=2048):  2048 × 640 × 61 = 79,953,920 B
        N=1024 (DSA off, n_eff=1024): 1024 × 640 × 61 = 39,976,960 B
    """
    return n_eff * (d_kv_latent * _FP8_BYTES + d_h_rope * _BF16_BYTES)


def bytes_dsa_k_index_cache_read(d_idx: int, n_total: int) -> int:
    """DSA indexer K-cache read: N tokens × d_idx × L  bytes.

    Storage precision: FP8 (1 byte/element), per V3.2 indexer design.

    Unlike the main KV cache, this is NOT clamped by top_k -- the indexer
    must scan all N historical tokens to score them.

    Note: per user decision (2026-05-13), this is read EVEN WHEN DSA is
    inactive at runtime (i.e. N < top_k). The caller does not gate this.

    Sanity (V3.2, d_idx=128, ×61 layers, FP8):
        N=1024:  1024 * 128 * 61 = 7,995,392
        N=2048:  2048 * 128 * 61 = 15,990,784
        N=9216:  9216 * 128 * 61 = 71,958,528
        N=30720: 30720 * 128 * 61 = 239,861,760
    """
    return n_total * d_idx * _FP8_BYTES


# ============================================================================
#  FFN (DeepSeekMoE) helpers
#
#  DeepSeek-V3 FFN structure (real, per paper §2.1.2 / §4.2):
#    - First `num_dense_ffn_layers` layers (= 3): standard dense FFN with
#      intermediate=18432.
#    - Remaining 58 layers: MoE with 1 shared + 256 routed experts, top-K=8,
#      expert_intermediate=2048.
#
#  Current simplification (per user 2026-05-13):
#    - Treat ALL `num_layers` (= 61) as MoE layers with expert_intermediate=2048.
#    - Skip dense FFN modelling. This causes ~5% over-count on `expert_weight`
#      (because dense FFN weight (smaller, 1 instance) is replaced by 256
#      "fake experts" of size 2048). Negligible at first pass; revisit later
#      if needed.
#
#  Counting convention (cluster-wide):
#    Total tokens entering FFN per step = batch_per_GPU × EP
#       (because attn DP × TP = EP, so EP GPUs each feed `batch_per_GPU` tokens
#        into the EP plane.)
#    Total FLOPs   = total_tokens × (router + (top_k + n_shared) × per_expert)
#    Total Bytes   = (n_routed + EP × n_shared) × per_expert_weight + router_weight
#
#  Per-GPU is total / EP (mathematically equivalent for AI / MFU purposes).
# ============================================================================


# ----- FLOPs ------------------------------------------------------------------


def flops_moe_router_per_layer_per_token(d: int, n_routed: int) -> int:
    """Router scoring: project hidden state to per-expert affinity scalars.

    Formula: 2 * d * n_routed   (per layer per token)

    Sanity (V3, d=7168, n_routed=256, ×61 layers under the simplified
    'all MoE' assumption):
        2 * 7168 * 256 = 3,670,016 / layer
        × 61 = 223,870,976
    """
    return 2 * d * n_routed


def flops_moe_per_expert_per_layer_per_token(d: int, expert_intermediate: int) -> int:
    """Single expert SwiGLU forward (gate + up + down): per layer per token.

    Formula: 2 * 3 * d * expert_intermediate

    Sanity (V3, d=7168, expert_intermediate=2048, ×61 layers):
        2 * 3 * 7168 * 2048 = 88,080,384 / layer
        × 61 = 5,372,903,424
    """
    return 2 * 3 * d * expert_intermediate


# ----- Bytes ------------------------------------------------------------------


def bytes_moe_per_expert_weight_all_layers(
    d: int, expert_intermediate: int, n_layers: int
) -> int:
    """SwiGLU weights for ONE expert across ALL layers (FP8).

    Formula: 3 * d * expert_intermediate * n_layers * dtype

    Sanity (V3, d=7168, expert_intermediate=2048, n_layers=61, FP8):
        3 * 7168 * 2048 * 61 = 2,686,451,712  ✓
    """
    return 3 * d * expert_intermediate * n_layers * _FP8_BYTES


def bytes_moe_router_weight_all_layers(
    d: int, n_routed: int, n_layers: int
) -> int:
    """Router weights across ALL MoE layers (FP8).

    Formula: d * n_routed * n_layers * dtype

    Sanity (V3, d=7168, n_routed=256, n_layers=61 simplified, FP8):
        7168 * 256 * 61 = 111,935,488

    Note: real V3 has 58 MoE layers (not 61); using 61 keeps consistency with
    the simplified `expert_weight` accounting above. ~5% over-count.
    """
    return d * n_routed * n_layers * _FP8_BYTES


def bytes_moe_token_dispatch_combine_per_layer(
    d: int, top_k: int
) -> int:
    """Cluster-wide local-HBM bytes per token per layer for the EP all-to-all
    dispatch + combine of routed-expert hidden activations (BF16).

    Physics — every token must traverse the EP boundary twice per MoE layer
    (once dispatched to its top-k expert owners, once combined back):

        dispatch:
            send-side local read   :    1 × d × bf16    (token from HBM into
                                                         dispatch kernel)
            recv-side land write   : top_k × d × bf16   (top_k receiver GPUs
                                                         each land a copy)
        combine:
            recv-side local read   : top_k × d × bf16   (each receiver reads
                                                         its expert output)
            send-side land write   :    1 × d × bf16    (sender lands the
                                                         combined sum)

      Per token per layer cluster-wide = 2 * (1 + top_k) * d * bf16.

    Conservative model assumptions:
      - The top_k experts of a token land on top_k distinct GPUs (typical
        for fine-grained routing with EP=32, n_routed=256). If experts
        cluster on fewer GPUs, the recv-side bytes drop linearly — this is
        an upper bound.
      - Shared experts are NOT included: every GPU hosts a copy locally
        (replicated), so they incur no all-to-all traffic.
      - Hidden state is BF16 in V3 / V3.2 / V4 deployments (FP8 is used
        only for weights and KV cache).

    Sanity (V3, d=7168, top_k=8, BF16, ×61 layers):
        2 * (1 + 8) * 7168 * 2 = 258,048 / token / layer / cluster
        × 61 layers = 15,740,928 / token / cluster
    """
    return 2 * (1 + top_k) * d * _BF16_BYTES


# ============================================================================
#  Session-level KV cache: closed-form formula (V3 / V3.2)
#
#  Used by ModelCostEstimator.session_kv_bytes(). Returns the KV cache bytes
#  consumed by ONE decode session of `session_length` tokens, summed across
#  all attention layers and (currently) NOT TP-sharded — MLA's KV latent is
#  replicated across TP ranks because cross-rank attention shuffling is too
#  expensive.
#
#  Mixed-precision storage convention (verified against V3.2 HuggingFace
#  reference implementation; also documented in V4 paper §2.3.4):
#    * KV latent dimensions    → FP8  (1 B/elem)
#    * RoPE-key dimensions     → BF16 (2 B/elem; preserves rotary
#                                      positional encoding fidelity)
#    * DSA indexer K-cache     → FP8  (1 B/elem; V3.2 paper line 146)
#
#  Per-token KV bytes per layer:
#      V3 base   = d_kv_latent × FP8 + d_h_rope × BF16
#      V3.2      = V3_base + d_idx × FP8                 (DSA indexer K-cache)
#
#  Total session KV (per GPU, MLA-replicated, not TP-sharded):
#      session_kv = num_layers × ctx × per_token_kv
#
#  Mixed-precision verification (V3.2): the FP8 KV-latent + BF16 RoPE-key +
#  FP8 indexer-K scheme has been confirmed against the open-source V3.2
#  HuggingFace reference implementation. (V3 / V3.2 papers do not state
#  inference KV dtype explicitly; V3 §3.3.3 covers FP8 for activations /
#  optimizer / communication only, and V3.2 line 146 only mentions
#  "indexer ... can be implemented in FP8".) The convention is encoded
#  via _FP8_BYTES / _BF16_BYTES below; if a future model variant ever
#  uses a different scheme, change them in one place.
#
#  ----- Sanity-check vs user estimate (NOT a real measurement) -----
#  per_token_v32 = 61 × (512 × 1 + 64 × 2 + 128 × 1) = 61 × 768 = 46,848 B/token
#
#    ctx     | closed-form  | user estimate | ratio (closed/estimate)
#    --------|--------------|---------------|------------------------
#    128 K   |   5.72 GiB   |   5.24 GiB    |   1.09x
#    1   M   |  45.75 GiB   |  41.92 GiB    |   1.09x
#    4   M   | 183.00 GiB   | 167.68 GiB    |   1.09x
#
#  The right-hand "user estimate" column is NOT a production measurement
#  — it is a user-provided rough estimate. The closed-form is exactly 9%
#  above it across all ctx, which is suspicious (constant ratio suggests
#  the estimate may itself be a slightly different closed-form, e.g.
#  using a 9%-smaller per_token constant). Do not draw conclusions from
#  this gap until a real production trace is available.
#
#  TODO(measure): obtain real V3.2 HBM occupancy measurements from a
#  deployed inference engine (e.g. SGLang / vLLM with HF reference
#  weights) and replace the "user estimate" column. Only then can we
#  decide whether the closed-form needs a correction factor.
# ============================================================================


def latent_kv_bytes_for_dtype(dtype_kv) -> float:
    """Map a KV-cache dtype to the per-element byte size of the MLA *latent*.

    Only the KV latent switches precision; the RoPE-key sub-vector always
    stays BF16 (rotary needs the precision) and the DSA indexer K stays FP8:
        * Dtype.BF16                 -> 2 B/elem  (full-precision BF16 latent)
        * Dtype.FP8 / MIXED_BF16_FP8 -> 1 B/elem  (default FP8 latent)

    Accepts either a Dtype enum or a raw string, avoiding a `core` import
    (which would create a circular dependency).
    """
    kv_val = getattr(dtype_kv, "value", dtype_kv)
    return _BF16_BYTES if kv_val == "bf16" else _FP8_BYTES


def session_kv_bytes_closed_form(
    session_length: int,
    n_layers: int,
    d_kv_latent: int,
    d_h_rope: int,
    d_idx: int | None,
    latent_bytes: float = _FP8_BYTES,
) -> int:
    """Closed-form session KV bytes for V3-family models.

    Args:
        session_length: ctx tokens in the session (S).
        n_layers: number of transformer layers (L).
        d_kv_latent: MLA KV-latent dim (d_c, e.g. 512).
        d_h_rope: MLA RoPE-key dim per token (d_h^R, e.g. 64).
        d_idx: DSA indexer head dim, or None if no DSA (V3 base).
        latent_bytes: per-element bytes for the KV latent. Default
            `_FP8_BYTES` (1) preserves the historical FP8-latent behaviour;
            pass `_BF16_BYTES` (2) for BF16-latent deployments. Use
            `latent_kv_bytes_for_dtype(dtype_kv)` to derive it.

    Returns: total bytes (per GPU, replicated across TP).
    """
    per_token_per_layer = (
        d_kv_latent * latent_bytes      # MLA latent  (FP8 or BF16)
        + d_h_rope  * _BF16_BYTES       # RoPE key    (BF16, always)
    )
    if d_idx is not None:
        per_token_per_layer += d_idx * _FP8_BYTES   # DSA indexer K (FP8)
    return int(n_layers * session_length * per_token_per_layer)


# ============================================================================


# ============================================================================
#  V3 estimator (dense MLA, no DSA)
# ============================================================================


@register_model("deepseek-v3")
class DeepSeekV3Estimator(ModelCostEstimator):
    """DeepSeek-V3 base: dense MLA, no DSA."""

    default_model_config = {
        "num_layers": 61,
        "num_dense_ffn_layers": 3,
        "hidden_size": 7168,                      # d
        "num_q_heads": 128,                       # n_h
        "head_dim": 128,                          # d_h (NoPE)
        "mla_kv_latent_dim": 512,                 # d_c
        "mla_query_latent_dim": 1536,             # d_c'
        "mla_rope_head_dim": 64,                  # d_h^R
        # MoE (placeholder; not used in attn cost)
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "top_k_experts": 8,
        "expert_intermediate_size": 2048,
        "vocab_size": 128_000,
        "mtp_depth": 0,
        # DSA off for V3 base.
        "dsa_topk": None,
        "dsa_q_idx_heads": 64,
        "dsa_idx_head_dim": 128,
    }

    # ---------- attention -------------------------------------------------

    def compute_attn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        cfg = self.merged_config(model_overrides)

        # Only the decode path is currently modelled.
        if workload.phase.value != "decode":
            return StageCost(flops=0.0, bytes_=0.0)

        # ---- gather config ----
        N = int(workload.context_length)
        bs = int(workload.batch_size)
        L = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))

        d = int(cfg["hidden_size"])
        n_h = int(cfg["num_q_heads"])
        d_h = int(cfg["head_dim"])
        d_q_latent = int(cfg["mla_query_latent_dim"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope = int(cfg["mla_rope_head_dim"])

        dsa_topk = cfg.get("dsa_topk")
        dsa_active = dsa_topk is not None and N >= int(dsa_topk)
        n_eff = int(dsa_topk) if dsa_active else N
        q_idx_heads = int(cfg["dsa_q_idx_heads"])
        d_idx = int(cfg["dsa_idx_head_dim"])

        # ---- per-layer per-token FLOPs (single sample, no parallel divisor) ----
        # 6 projections (always present, constant in N). NoPE and RoPE
        # are folded: flops_q_up_proj includes W_QR, flops_kv_down_proj
        # includes W_KR (matches the bytes side's widened dims).
        per_layer_per_token = (
            flops_q_down_proj(d, d_q_latent)
            + flops_q_up_proj(d_q_latent, n_h, d_h, d_h_rope)
            + flops_kv_down_proj(d, d_kv_latent, d_h_rope)
            + flops_k_up_proj(d_kv_latent, n_h, d_h)
            + flops_v_up_proj(n_h, d_kv_latent, d_h)
            + flops_o_proj(n_h, d_h, d)
        )
        # Main attention: Q.K^T and (softmax).V always run, with n_eff tokens.
        # When DSA is active, n_eff = top_k (the indexer selects top_k tokens
        # which are then attended to via main attention). When DSA is off,
        # n_eff = N (dense attention over the full history).
        # The 4 DSA components below are ADDITIONAL when DSA is active --
        # they do not replace main attn QK, they precede it (indexer scoring
        # produces the top-k indices that main attn uses).
        per_layer_per_token += flops_attn_qk_dot(n_h, d_kv_latent, d_h_rope, n_eff)
        per_layer_per_token += flops_attn_weighted_v(n_h, d_kv_latent, n_eff)

        # DSA path (only when active): adds 4 components that produce top-k indices.
        if dsa_active:
            per_layer_per_token += (
                flops_dsa_q_idx_proj(d_q_latent, q_idx_heads, d_idx)
                + flops_dsa_k_idx_proj(d, d_idx)
                + flops_dsa_indexer_scoring(q_idx_heads, d_idx, N)
                + flops_dsa_head_weight(d, q_idx_heads)
            )

        # ---- per-layer bytes ----
        # (a) Weight bytes: shared across the batch -> NOT multiplied by bs.
        #     Sharded by TP across heads (Q_up, K_up, V_up+O).
        # (b) KV cache bytes: per-sample -> multiplied by bs.
        #     Sharded by TP across heads (KV is read by per-TP-shard heads).
        # NOTE: For simplicity, we apply the same TP-divisor to all weight and
        #       KV bytes. This is a reasonable approximation but slightly
        #       over-estimates the savings on Q_down / KV_down (which are
        #       column-split rather than head-split). Refine later if needed.

        per_layer_weight_bytes = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h, d)
        )

        # KV cache reads (per sample). n_eff captures the DSA top-k clamp.
        # Indexer K-cache is read even when DSA is inactive at runtime
        # (per user decision 2026-05-13: keep the modelling simple).
        per_layer_per_sample_kv_bytes = (
            bytes_kv_cache_read(d_kv_latent, d_h_rope, n_eff)
            + bytes_dsa_k_index_cache_read(d_idx, N)
        )

        if dsa_active:
            per_layer_weight_bytes += (
                bytes_dsa_q_idx_proj_weight(d_q_latent, q_idx_heads, d_idx)
                + bytes_dsa_k_idx_proj_weight(d, d_idx)
                + bytes_dsa_head_weight(d, q_idx_heads)
            )

        # ---- aggregate: per-GPU per-step FLOPs and bytes ----
        flops = per_layer_per_token * L * bs / TP
        bytes_ = (
            per_layer_weight_bytes * L                  # shared across batch
            + per_layer_per_sample_kv_bytes * L * bs    # per-sample
        ) / TP

        return StageCost(flops=float(flops), bytes_=float(bytes_))

    # ---------- FFN (DeepSeekMoE) -----------------------------------------

    def compute_ffn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        """DeepSeekMoE FFN cost.

        Counting convention (cluster-wide, divided by EP at the end to get
        per-GPU; mathematically equivalent for AI / MFU purposes):

          total_tokens (per step) = batch_per_GPU × EP
            (because attn DP × TP = EP, so every EP-rank's GPU feeds
             batch_per_GPU tokens into the EP plane.)

          total_FLOPs = total_tokens × [
              n_layers × router_per_token              (each token routed once / layer)
              + n_layers × (top_k + n_shared) × per_expert_per_token
                                                       (each token activates 9 experts / layer)
          ]

          total_weight_bytes =
              (n_routed + EP × n_shared) × per_expert_weight_all_layers
              + EP × router_weight_all_layers
              (router is replicated on every GPU, same as shared experts)

          total_act_bytes =
              total_tokens × n_layers × dispatch_combine_per_layer
              (every token must traverse the EP all-to-all twice per MoE
               layer — once dispatched to its top-k owners, once combined
               back. See bytes_moe_token_dispatch_combine_per_layer for
               the per-token-per-layer = 2*(1+top_k)*d*bf16 derivation.
               Shared experts incur no all-to-all and are excluded.)

          total_bytes = total_weight_bytes + total_act_bytes
          per_GPU_FLOPs = total_FLOPs / EP
          per_GPU_bytes = total_bytes   / EP
          AI = total_FLOPs / total_bytes  (= per_GPU equivalent)

        Simplifications (TODO):
          - All `num_layers` (61) treated as MoE layers, including the first
            `num_dense_ffn_layers` (3). Real V3 has dense FFN there, but we
            simplify to MoE expert_intermediate for now (~5% over-count on
            expert_weight).
          - Token activation bytes count only the EP all-to-all dispatch +
            combine traffic (the conservative "must-go-through-HBM" floor).
            We do NOT count the per-mat-vec activation re-reads inside an
            expert's gate/up/down sequence — those are typically fused in
            the kernel and live in registers / SMEM, not HBM.
        """
        cfg = self.merged_config(model_overrides)

        if workload.phase.value != "decode":
            return StageCost(flops=0.0, bytes_=0.0)

        bs = int(workload.batch_size)
        L = int(cfg["num_layers"])
        EP = max(1, int(parallel.ep))
        DP = max(1, int(parallel.dp))

        d = int(cfg["hidden_size"])
        n_routed = int(cfg["num_routed_experts"])
        n_shared = int(cfg["num_shared_experts"])
        top_k = int(cfg["top_k_experts"])
        expert_intermediate = int(cfg["expert_intermediate_size"])

        # ---- FLOPs (cluster-wide) ----
        # workload.batch_size is ATTN per-GPU bs (per-DP sessions, MLA replicated
        # across TP).  FFN per-GPU token count = bs * DP / EP = bs / TP, because
        # TP ranks hold replicated sessions and contribute no extra unique tokens.
        ffn_bs = bs * DP / EP
        total_tokens = ffn_bs * EP
        per_layer_flops_per_token = (
            flops_moe_router_per_layer_per_token(d, n_routed)
            + (top_k + n_shared) * flops_moe_per_expert_per_layer_per_token(d, expert_intermediate)
        )
        total_flops = total_tokens * L * per_layer_flops_per_token

        # ---- Bytes (cluster-wide) ----
        # (a) Weight bytes:
        #   Routed experts: sharded across EP, so cluster-wide = n_routed copies.
        #   Shared experts: replicated on every GPU, so cluster-wide = EP × n_shared copies.
        #   Router: replicated on every GPU (each GPU needs full d×n_routed to score),
        #           so cluster-wide = EP × router_w.
        per_expert_w = bytes_moe_per_expert_weight_all_layers(d, expert_intermediate, L)
        router_w = bytes_moe_router_weight_all_layers(d, n_routed, L)
        total_weight_bytes = (n_routed + EP * n_shared) * per_expert_w + EP * router_w

        # (b) Token activation bytes (EP all-to-all dispatch + combine).
        #     Conservative "must-go-through-HBM" floor; see helper docstring.
        per_token_per_layer_act = bytes_moe_token_dispatch_combine_per_layer(d, top_k)
        total_act_bytes = total_tokens * L * per_token_per_layer_act

        total_bytes = total_weight_bytes + total_act_bytes

        # ---- Per-GPU (divide by EP; the math is equivalent for AI / time) ----
        flops_per_gpu = total_flops / EP
        bytes_per_gpu = total_bytes / EP

        return StageCost(flops=float(flops_per_gpu), bytes_=float(bytes_per_gpu))

    # ---------- Capacity-analysis API (decode only) -----------------------

    def session_kv_bytes(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Per-GPU KV cache bytes for ONE decode session of `session_length` tokens.

        Closed-form (2026-05-20): mixed-precision per the V3.2/V4 spec —
        KV latent FP8, RoPE BF16, DSA indexer FP8. KV is replicated across
        TP ranks (MLA does not TP-shard KV), so `parallel` is unused. Future
        deployments could TP-shard, in which case divide by TP here.

        Detailed formula and the 9% closed-form vs user-estimate gap
        discussion live in the module block-comment above
        `session_kv_bytes_closed_form`.

        TODO(measure): we currently have NO real production measurement to
        validate the closed-form. The reference column in the block-comment
        is a user-provided rough estimate, ~9% below the closed-form across
        all ctx. Obtain a real V3.2 HBM-occupancy trace (SGLang / vLLM with
        HF reference weights) before applying any correction factor.
        """
        del parallel  # KV replicated across TP (MLA); TP-shard reserved
        cfg = self.merged_config(model_overrides)
        n_layers    = int(cfg["num_layers"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])
        d_idx_raw   = cfg.get("dsa_idx_head_dim")
        dsa_present = cfg.get("dsa_topk") is not None
        d_idx = int(d_idx_raw) if (dsa_present and d_idx_raw is not None) else None
        return float(session_kv_bytes_closed_form(
            session_length=int(session_length),
            n_layers=n_layers,
            d_kv_latent=d_kv_latent,
            d_h_rope=d_h_rope,
            d_idx=d_idx,
            latent_bytes=latent_kv_bytes_for_dtype(dtype_kv),
        ))

    def indexer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """DSA indexer K-cache bytes per session, summed across all layers.

        For V3 base (no DSA) this returns 0. For V3.2 (DSA active) every
        layer carries an indexer:
            indexer_kv = num_layers × session_length × d_idx × FP8

        The FP8 storage dtype is consistent with the V3.2 HuggingFace
        reference implementation (V3.2 paper §2 line 146 confirms
        "can be implemented in FP8" for indexer compute; HF impl also
        stores the K-cache in FP8).
        """
        del parallel, dtype_kv
        cfg = self.merged_config(model_overrides)
        if cfg.get("dsa_topk") is None:
            return 0.0
        n_layers = int(cfg["num_layers"])
        d_idx    = int(cfg["dsa_idx_head_dim"])
        return float(n_layers * int(session_length) * d_idx * _FP8_BYTES)

    def cold_layer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Bytes that must be fetched per missed layer per session.

        For V3 base (no DSA): full per-layer KV (S × v_token).
        For V3.2 (DSA active): main attn only attends top_k tokens, so
        the cold layer fetch is capped at top_k tokens (or session_length
        if S < top_k, the dense fallback regime).

        v_token = d_kv_latent × FP8 + d_h_rope × BF16  (mixed-precision
        per V4 paper §2.3.4, also assumed for V3 / V3.2).
        """
        del parallel, dtype_kv
        cfg = self.merged_config(model_overrides)
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])
        v_token = d_kv_latent * _FP8_BYTES + d_h_rope * _BF16_BYTES

        topk = cfg.get("dsa_topk")
        if topk is None or session_length < int(topk):
            n_eff = int(session_length)
        else:
            n_eff = int(topk)
        return float(n_eff * v_token)

    def weight_bytes_per_gpu(
        self,
        parallel,
        dtype_param,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Static weight bytes resident on ONE GPU (attn + FFN), TP/EP-sharded.

        Reuses the existing per-component bytes_*_weight() helpers. dtype_param
        is currently unused: those helpers hard-code FP8. TODO: plumb dtype_param
        through the helpers when we add BF16 / FP4 weight support.

        Sharding model:
            - Attn weights: divided by TP (head-sharded approximation; see
              compute_attn_cost for the same simplification).
            - MoE routed-expert weights: divided by EP (each rank owns
              n_routed / EP experts).
            - MoE shared-expert weights + router: replicated on every GPU.
        """
        del dtype_param  # see TODO above
        cfg = self.merged_config(model_overrides)

        L = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))
        EP = max(1, int(parallel.ep))

        d = int(cfg["hidden_size"])
        n_h = int(cfg["num_q_heads"])
        d_h = int(cfg["head_dim"])
        d_q_latent = int(cfg["mla_query_latent_dim"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope = int(cfg["mla_rope_head_dim"])

        dsa_topk = cfg.get("dsa_topk")
        dsa_present = dsa_topk is not None
        q_idx_heads = int(cfg["dsa_q_idx_heads"])
        d_idx = int(cfg["dsa_idx_head_dim"])

        n_routed = int(cfg["num_routed_experts"])
        n_shared = int(cfg["num_shared_experts"])
        expert_intermediate = int(cfg["expert_intermediate_size"])

        # ---- Attn weights (per-layer, summed across L, then TP-sharded) ----
        attn_per_layer_bytes = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h, d)
        )
        if dsa_present:
            attn_per_layer_bytes += (
                bytes_dsa_q_idx_proj_weight(d_q_latent, q_idx_heads, d_idx)
                + bytes_dsa_k_idx_proj_weight(d, d_idx)
                + bytes_dsa_head_weight(d, q_idx_heads)
            )
        attn_total_bytes = attn_per_layer_bytes * L / TP

        # ---- MoE weights ----
        # Routed experts: n_routed total, partitioned across EP ranks.
        # Shared experts: replicated on every GPU.
        # Router: replicated on every GPU.
        per_expert_w = bytes_moe_per_expert_weight_all_layers(
            d, expert_intermediate, L
        )
        router_w = bytes_moe_router_weight_all_layers(d, n_routed, L)

        routed_per_gpu = (n_routed / EP) * per_expert_w
        shared_per_gpu = n_shared * per_expert_w
        router_per_gpu = router_w

        ffn_total_bytes = routed_per_gpu + shared_per_gpu + router_per_gpu

        return float(attn_total_bytes + ffn_total_bytes)


# ============================================================================
#  V3.2 estimator (V3 architecture + DSA)
# ============================================================================


@register_model("deepseek-v3.2")
class DeepSeekV32Estimator(DeepSeekV3Estimator):
    """V3.2 = V3 architecture + DSA (lightning indexer + top-k sparse attention).

    Inherits compute_attn_cost / compute_ffn_cost from V3; only the default
    config differs (dsa_topk is set, so the DSA path activates for N >= 2048).
    """

    default_model_config = {
        **DeepSeekV3Estimator.default_model_config,
        "dsa_topk": 2048,
    }

