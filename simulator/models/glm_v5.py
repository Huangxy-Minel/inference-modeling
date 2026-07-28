"""GLM-5 cost estimator: asymmetric MLA (V head dim ≠ K head dim) + DSA + DeepSeekMoE.

GLM-5 sits in the DeepSeek-V3 / V3.2 architectural lineage but introduces
**MLA-256** — a deliberately asymmetric Multi-head Latent Attention
where the **V per-head dim (256) is twice the K per-head dim (128)**.
The effect is that V-up-proj and O-proj operate on a doubled per-head
width, while the rest of the attention stack (Q-up-proj K-side, K-up-proj,
indexer) keeps DSv3's `head_dim=128`.

All FLOPs / bytes helpers from `deepseek_v3` are reused **as-is**; the
asymmetric design only changes which scalar (`head_dim` vs `value_head_dim`)
is passed into the V-side helpers.

Companion design doc: `docs/glm_v5.md`.

Source: GLM-5 technical report (Zhipu AI / THUDM, 2025), Table 10
("Architecture comparison vs GLM-4.5") + §2.1 + §RL DSA insights + §4
Mixed-Precision deployment.

Quantisation note (intentional simplification, see `docs/glm_v5.md` §1):
    * Real deployment is W8A8 standard attn + W4A8 MoE expert.
    * Cost model uses FP8 placeholder (1 B/elem) so cross-model AI/MFU
      comparisons stay apples-to-apples with DSv3 / V3.2.
    * Production-faithful weight-bytes: multiply MoE-expert byte totals
      by 0.5 (W4 = 0.5 B/elem).

MTP note: GLM-5 has 1 MTP head (training shares 3 layers, inference runs
4 spec-decoding steps with ~2.76 mean accept length). Excluded from the
main forward path here, same convention as DSv3 estimator.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import ModelCostEstimator, StageCost, register_model

# Reuse DSv3 helpers verbatim. The asymmetric MLA design only changes
# *which* per-head dim is passed into each helper.
from .deepseek_v3 import (
    # FLOPs
    flops_q_down_proj,
    flops_q_up_proj,
    flops_kv_down_proj,
    flops_k_up_proj,
    flops_v_up_proj,
    flops_o_proj,
    flops_attn_qk_dot,
    flops_attn_weighted_v,
    flops_dsa_q_idx_proj,
    flops_dsa_k_idx_proj,
    flops_dsa_indexer_scoring,
    flops_dsa_head_weight,
    flops_moe_router_per_layer_per_token,
    flops_moe_per_expert_per_layer_per_token,
    # Bytes — weight
    bytes_q_down_proj_weight,
    bytes_q_up_proj_weight,
    bytes_kv_down_proj_weight,
    bytes_k_up_proj_weight,
    bytes_v_up_o_proj_weight,
    bytes_dsa_q_idx_proj_weight,
    bytes_dsa_k_idx_proj_weight,
    bytes_dsa_head_weight,
    bytes_moe_per_expert_weight_all_layers,
    bytes_moe_router_weight_all_layers,
    bytes_moe_token_dispatch_combine_per_layer,
    # Bytes — KV
    bytes_kv_cache_read,
    bytes_dsa_k_index_cache_read,
    # Capacity closed-form
    session_kv_bytes_closed_form,
    latent_kv_bytes_for_dtype,
    # Dtype constants (FP8 placeholder for W8A8/W4A8; BF16 for RoPE)
    _FP8_BYTES,
    _BF16_BYTES,
)


# ============================================================================
#  GLM-5 estimator
# ============================================================================


@register_model("glm-5")
class GLM5Estimator(ModelCostEstimator):
    """GLM-5 (744B / 40B) — asymmetric MLA-256 + DSA + DeepSeekMoE.

    Architecture summary (see Table 10 / §2.1 of the GLM-5 tech report):
      * 3 dense FFN + 75 MoE = 78 transformer layers (MTP excluded)
      * d = 6144, n_h = 64
      * Asymmetric MLA: K-side head_dim = 128, **V-side value_head_dim = 256**
      * d_q_latent = 2048, d_kv_latent = 512, d_h_rope = 64
      * MoE: 256 routed × top-8 + 1 shared, expert_intermediate = 2048
      * DSA: top_k = 2048, indexer 32 heads × 128 dim
      * Vocab 154 880, max ctx 131 072

    Modelling simplifications (see `docs/glm_v5.md` §4):
      * 3 dense FFN layers folded into MoE counting (same as DSv3)
      * MTP head excluded from forward
      * FP8 placeholder for W8A8 attn + W4A8 MoE (1 B/elem)
      * MLA KV stays replicated across TP (no TP-shard)
    """

    # Calibrated fixed HBM overhead (GiB), keyed by GPU preset. Measured
    # from an SGLang HBM-profiling run of GLM-5.1-FP8 (2-node 16×H20,
    # TP1/DP16/EP16, DeepEP, BF16 KV, mem-fraction-static=0.8, 2026-07-23):
    #   Total 95 GiB = model 59.71 + KV 12.79 + [fixed 22.5]
    #   where the model's own weight estimate is 58.64 GiB, so to make the
    #   capacity engine's KV budget match the measured KV pool (12.79 GiB):
    #     fixed = 96×0.95 − 58.64 − 12.79 ≈ 19.8 GiB
    #   This 19.8 lumps together true fixed infra (PyTorch active 2.54 +
    #   native/DeepEP 6.32 + CUDA graph 5.24 ≈ 14.1) AND the framework's
    #   runtime-dynamic reserve (PyTorch inactive cache etc.) that SGLang
    #   never releases to the KV pool. Validated: predicted max_total_tokens
    #   138325 vs measured 137024 (0.9%).
    overhead_fixed_gb_by_gpu = {
        "H20": 19.8,
        # H100 (== H800, same HBM/compute spec). Measured from a 4-node
        # 32×H800 SGLang run of GLM-5.1-FP8 (TP1/DP32/EP32, DeepEP, BF16 KV,
        # 4K in + 128 out, 2026-07-26): Total 79.11 GiB usable = model 38.54
        # + KV pool 21.0 (data 18.84 + indexer 2.16) + [fixed+available].
        # With the model's own weight estimate 36.70 GiB and the 80 GiB
        # nominal H100 capacity used by the preset:
        #   fixed = 80×0.95 − 36.70 − 21.0 ≈ 18.3 GiB
        # Validated: predicted admitted decode BS = KV_pool / session_kv =
        # 21.0 / 0.3928 = 53, matching the measured admit BS=53 (device_pool).
        "H100": 18.3,
    }

    default_model_config = {
        # ---- Layer counts ----
        "num_layers": 78,                         # 3 dense + 75 MoE
        "num_dense_ffn_layers": 3,                # informational; merged into MoE
        "mtp_depth": 1,                           # tracked but not on forward path
        # ---- Hidden / attention ----
        "hidden_size": 6144,                      # d
        "num_q_heads": 64,                        # n_h  (DSv3 is 128)
        "head_dim": 128,                          # d_h  — K-side / NoPE per-head
        "value_head_dim": 256,                    # ★ asymmetric V-side per-head
        "mla_kv_latent_dim": 512,                 # d_c
        "mla_query_latent_dim": 2048,             # d_c'  (DSv3 is 1536)
        "mla_rope_head_dim": 64,                  # d_h^R
        # ---- MoE ----
        "num_routed_experts": 256,
        "num_shared_experts": 1,
        "top_k_experts": 8,
        "expert_intermediate_size": 2048,
        # Dense intermediate is informational; not actually invoked because
        # we treat all `num_layers` as MoE for simplicity (see DSv3 convention).
        "dense_ffn_intermediate": 12288,
        # ---- Misc ----
        "vocab_size": 154_880,
        # ---- DSA (always on by default; matches V3.2 deployment) ----
        "dsa_topk": 2048,
        "dsa_q_idx_heads": 32,                    # DSv3.2 is 64
        "dsa_idx_head_dim": 128,
    }

    # ------------------------------------------------------------------
    #  Attention cost (decode only)
    # ------------------------------------------------------------------

    def compute_attn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        cfg = self.merged_config(model_overrides)

        if workload.phase.value != "decode":
            return StageCost(flops=0.0, bytes_=0.0)

        # ---- gather config ----
        N  = int(workload.context_length)
        bs = int(workload.batch_size)
        L  = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))

        d           = int(cfg["hidden_size"])
        n_h         = int(cfg["num_q_heads"])
        d_h         = int(cfg["head_dim"])           # K-side / NoPE per head
        d_h_v       = int(cfg["value_head_dim"])     # V-side per head (★ 256)
        d_q_latent  = int(cfg["mla_query_latent_dim"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])

        dsa_topk    = cfg.get("dsa_topk")
        dsa_active  = dsa_topk is not None and N >= int(dsa_topk)
        n_eff       = int(dsa_topk) if dsa_active else N
        q_idx_heads = int(cfg["dsa_q_idx_heads"])
        d_idx       = int(cfg["dsa_idx_head_dim"])

        # ---- per-layer per-token FLOPs (single sample, no parallel divisor) ----
        # K-side helpers use head_dim (=128), V-side helpers use value_head_dim
        # (=256). Helpers themselves are unchanged from DSv3 — the asymmetric
        # design is expressed purely by which scalar is passed in.
        per_layer_per_token = (
            flops_q_down_proj(d, d_q_latent)
            + flops_q_up_proj(d_q_latent, n_h, d_h, d_h_rope)        # K-side d_h=128
            + flops_kv_down_proj(d, d_kv_latent, d_h_rope)
            + flops_k_up_proj(d_kv_latent, n_h, d_h)                 # K-side d_h=128
            + flops_v_up_proj(n_h, d_kv_latent, d_h_v)               # V-side d_h=256 ★
            + flops_o_proj(n_h, d_h_v, d)                            # V-side d_h=256 ★
        )
        # Main attention runs in latent space (W_UK absorbed) — independent
        # of value_head_dim. Form identical to DSv3.2.
        per_layer_per_token += flops_attn_qk_dot(n_h, d_kv_latent, d_h_rope, n_eff)
        per_layer_per_token += flops_attn_weighted_v(n_h, d_kv_latent, n_eff)

        # DSA additional FLOPs (only when active)
        if dsa_active:
            per_layer_per_token += (
                flops_dsa_q_idx_proj(d_q_latent, q_idx_heads, d_idx)
                + flops_dsa_k_idx_proj(d, d_idx)
                + flops_dsa_indexer_scoring(q_idx_heads, d_idx, N)
                + flops_dsa_head_weight(d, q_idx_heads)
            )

        # ---- per-layer bytes ----
        # Weight bytes (shared across batch). V-side helper takes value_head_dim.
        per_layer_weight_bytes = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h_v, d)   # V-side d_h=256 ★
        )

        # KV-cache reads (per sample). n_eff captures the DSA top-k clamp for
        # main KV; indexer K-cache is read every step regardless (per V3.2
        # convention also followed in DSv3 estimator).
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
        # Same TP simplification as DSv3 (head-sharded approximation).
        flops = per_layer_per_token * L * bs / TP
        bytes_ = (
            per_layer_weight_bytes * L                  # shared across batch
            + per_layer_per_sample_kv_bytes * L * bs    # per-sample
        ) / TP

        return StageCost(flops=float(flops), bytes_=float(bytes_))

    # ------------------------------------------------------------------
    #  FFN (DeepSeekMoE) — identical to DSv3 convention; 3 dense layers
    #  folded into MoE for simplicity.
    # ------------------------------------------------------------------

    def compute_ffn_cost(
        self,
        parallel,
        workload,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> StageCost:
        cfg = self.merged_config(model_overrides)

        if workload.phase.value != "decode":
            return StageCost(flops=0.0, bytes_=0.0)

        bs = int(workload.batch_size)
        L  = int(cfg["num_layers"])
        EP = max(1, int(parallel.ep))
        DP = max(1, int(parallel.dp))

        d                   = int(cfg["hidden_size"])
        n_routed            = int(cfg["num_routed_experts"])
        n_shared            = int(cfg["num_shared_experts"])
        top_k               = int(cfg["top_k_experts"])
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

        # ---- Weight bytes (cluster-wide) ----
        per_expert_w = bytes_moe_per_expert_weight_all_layers(d, expert_intermediate, L)
        router_w     = bytes_moe_router_weight_all_layers(d, n_routed, L)
        total_weight_bytes = (n_routed + EP * n_shared) * per_expert_w + EP * router_w

        # ---- Token activation bytes (EP all-to-all dispatch + combine) ----
        per_token_per_layer_act = bytes_moe_token_dispatch_combine_per_layer(d, top_k)
        total_act_bytes = total_tokens * L * per_token_per_layer_act

        total_bytes = total_weight_bytes + total_act_bytes

        # ---- Per-GPU (divide cluster total by EP; equivalent for AI / time) ----
        flops_per_gpu = total_flops / EP
        bytes_per_gpu = total_bytes / EP

        return StageCost(flops=float(flops_per_gpu), bytes_=float(bytes_per_gpu))

    # ------------------------------------------------------------------
    #  Capacity-analysis API (decode only)
    # ------------------------------------------------------------------

    def session_kv_bytes(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Per-GPU KV cache bytes for ONE decode session.

        Mixed-precision storage identical to V3.2:
            per_token_per_layer = d_kv_latent × FP8 + d_h_rope × BF16
                                + d_idx × FP8        (DSA always on)
                                = 512 + 128 + 128 = 768 B
        MLA KV is replicated across TP — `parallel` is unused.
        """
        del parallel  # KV replicated across TP (MLA)
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

            indexer_kv = num_layers × session_length × d_idx × FP8

        Returns 0 if DSA is disabled via override (defensive; default is on).
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
        """Bytes that must be fetched per missed (cold) layer per session.

        DSA active path: main attn only attends top_k tokens, so cold-layer
        fetch is capped at top_k (or session_length if S < top_k).
        Per-token cost = d_kv_latent × FP8 + d_h_rope × BF16 (NOT including
        indexer-K, which is a separate API above).
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
        """Static weight bytes resident on ONE GPU (attn + MoE), TP/EP-sharded.

        FP8 placeholder (1 B/elem) for both W8A8 attn and W4A8 MoE — for
        production-faithful numbers, halve the MoE expert contribution.
        Sharding: attn divided by TP, routed experts divided by EP, shared
        experts and router replicated.
        """
        del dtype_param
        cfg = self.merged_config(model_overrides)

        L  = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))
        EP = max(1, int(parallel.ep))

        d           = int(cfg["hidden_size"])
        n_h         = int(cfg["num_q_heads"])
        d_h         = int(cfg["head_dim"])           # K-side
        d_h_v       = int(cfg["value_head_dim"])     # V-side ★
        d_q_latent  = int(cfg["mla_query_latent_dim"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])

        dsa_topk    = cfg.get("dsa_topk")
        dsa_present = dsa_topk is not None
        q_idx_heads = int(cfg["dsa_q_idx_heads"])
        d_idx       = int(cfg["dsa_idx_head_dim"])

        n_routed            = int(cfg["num_routed_experts"])
        n_shared            = int(cfg["num_shared_experts"])
        expert_intermediate = int(cfg["expert_intermediate_size"])

        # ---- Attn weights (per-layer, summed across L, then TP-sharded) ----
        attn_per_layer_bytes = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h_v, d)   # V-side ★
        )
        if dsa_present:
            attn_per_layer_bytes += (
                bytes_dsa_q_idx_proj_weight(d_q_latent, q_idx_heads, d_idx)
                + bytes_dsa_k_idx_proj_weight(d, d_idx)
                + bytes_dsa_head_weight(d, q_idx_heads)
            )
        attn_total_bytes = attn_per_layer_bytes * L / TP

        # ---- MoE weights ----
        per_expert_w = bytes_moe_per_expert_weight_all_layers(d, expert_intermediate, L)
        router_w     = bytes_moe_router_weight_all_layers(d, n_routed, L)

        routed_per_gpu = (n_routed / EP) * per_expert_w
        shared_per_gpu = n_shared * per_expert_w
        router_per_gpu = router_w

        ffn_total_bytes = routed_per_gpu + shared_per_gpu + router_per_gpu

        return float(attn_total_bytes + ffn_total_bytes)
