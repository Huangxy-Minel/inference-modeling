"""GLM-5.2 cost estimator: GLM-5 backbone + IndexShare + wider NoPE head.

GLM-5.2 is a near-identical successor to GLM-5 (asymmetric MLA + DSA +
DeepSeekMoE, 78 layers, d=6144). Only three things differ, two of which
touch the cost model:

  1. head_dim (qk NoPE per head) : 128 -> **192**  (widens Q-up / K-up)
  2. max context                 : 131072 -> **1_048_576 (1M)** (config only)
  3. **IndexShare**              : the DSA lightning indexer is computed
     once per group of `g = 4` layers and its top-k indices are reused by
     the other 3 layers. This collapses the indexer FLOPs / K-cache /
     weights from `L` layers to `n_full = ceil(L / g)` layers.

Everything else (n_h=64, v_head_dim=256, q_lora=2048, kv_lora=512,
rope=64, MoE 256×top-8+1 @ 2048, DSA topk=2048 / 32 heads / 128 dim,
vocab 154880, mtp=1) is byte-for-byte identical to GLM-5, so this class
subclasses GLM5Estimator and inherits `compute_ffn_cost` and
`cold_layer_kv_bytes_per_session` unchanged.

Companion design doc: `docs/glm_v52.md`.

Sources:
  * zai-org/GLM-5.2 config.json (HuggingFace)   — concrete parameters
  * zai-org/glm-52-blog (HuggingFace, 2026-06-17) — IndexShare, 1M, FP8 KV
  * GLM-5 technical report                        — shared background

Validation anchor: the closed-form per-token FLOPs with IndexShare (g=4)
reproduces the blog's published "~2.9× FLOP reduction at 1M context" vs
g=1 (see `docs/glm_v52.md` §2).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .base import StageCost, register_model
from .glm_v5 import GLM5Estimator

# Reuse DSv3 helpers verbatim (same set GLM-5 uses).
from .deepseek_v3 import (
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
    bytes_kv_cache_read,
    bytes_dsa_k_index_cache_read,
    session_kv_bytes_closed_form,
    latent_kv_bytes_for_dtype,
    _FP8_BYTES,
)


def _num_full_indexer_layers(num_layers: int, group: int) -> int:
    """Number of layers that run the FULL DSA indexer under IndexShare.

    IndexShare places 1 `full` indexer per group of `group` layers and
    reuses its top-k indices for the other (group-1) `shared` layers.
    n_full = ceil(L / group). group=1 disables IndexShare (n_full = L).
    """
    g = max(1, int(group))
    return math.ceil(int(num_layers) / g)


@register_model("glm-5.2")
class GLM52Estimator(GLM5Estimator):
    """GLM-5.2 — GLM-5 backbone + IndexShare + NoPE head_dim=192.

    Inherits `compute_ffn_cost` and `cold_layer_kv_bytes_per_session`
    from GLM5Estimator (identical MoE / main-KV behaviour). Overrides the
    attention path and the indexer-touching capacity APIs to apply the
    IndexShare `n_full = ceil(L / index_share_group)` scaling.
    """

    # GLM-5.2 has a different backbone (IndexShare, wider NoPE); do NOT
    # inherit GLM-5.1's H20 fixed-overhead calibration. Left empty until a
    # GLM-5.2 HBM-profiling run is measured (falls back to CLI / 0).
    overhead_fixed_gb_by_gpu: dict = {}

    default_model_config = {
        **GLM5Estimator.default_model_config,
        # ---- The two cost-model deltas vs GLM-5 ----
        "head_dim": 192,                # qk NoPE per head (GLM-5 was 128)
        "index_share_group": 4,         # IndexShare: 1 full indexer / 4 layers
        # ---- Config-only delta ----
        "max_context_length": 1_048_576,  # 1M (GLM-5 was 131072)
    }

    # ------------------------------------------------------------------
    #  Attention cost (decode only) — GLM-5 path + IndexShare
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

        N  = int(workload.context_length)
        bs = int(workload.batch_size)
        L  = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))

        d           = int(cfg["hidden_size"])
        n_h         = int(cfg["num_q_heads"])
        d_h         = int(cfg["head_dim"])           # K-side / NoPE per head (192)
        d_h_v       = int(cfg["value_head_dim"])     # V-side per head (256)
        d_q_latent  = int(cfg["mla_query_latent_dim"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])

        dsa_topk    = cfg.get("dsa_topk")
        dsa_active  = dsa_topk is not None and N >= int(dsa_topk)
        n_eff       = int(dsa_topk) if dsa_active else N
        q_idx_heads = int(cfg["dsa_q_idx_heads"])
        d_idx       = int(cfg["dsa_idx_head_dim"])

        # IndexShare: how many layers actually run the full indexer.
        n_full = _num_full_indexer_layers(L, cfg.get("index_share_group", 1))

        # ---- Per-layer per-token FLOPs that run in EVERY layer ----
        # 6 MLA projections (K-side d_h=192, V-side d_h=256) + main attn.
        per_layer_per_token = (
            flops_q_down_proj(d, d_q_latent)
            + flops_q_up_proj(d_q_latent, n_h, d_h, d_h_rope)        # NoPE 192
            + flops_kv_down_proj(d, d_kv_latent, d_h_rope)
            + flops_k_up_proj(d_kv_latent, n_h, d_h)                 # NoPE 192
            + flops_v_up_proj(n_h, d_kv_latent, d_h_v)               # V 256
            + flops_o_proj(n_h, d_h_v, d)                            # V 256
            + flops_attn_qk_dot(n_h, d_kv_latent, d_h_rope, n_eff)   # latent-space
            + flops_attn_weighted_v(n_h, d_kv_latent, n_eff)
        )

        # ---- DSA indexer FLOPs — only in n_full layers (IndexShare) ----
        dsa_per_full_layer = 0
        if dsa_active:
            dsa_per_full_layer = (
                flops_dsa_q_idx_proj(d_q_latent, q_idx_heads, d_idx)
                + flops_dsa_k_idx_proj(d, d_idx)
                + flops_dsa_indexer_scoring(q_idx_heads, d_idx, N)   # N-linear
                + flops_dsa_head_weight(d, q_idx_heads)
            )

        # ---- Weight bytes ----
        per_layer_weight_bytes = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h_v, d)   # V 256
        )
        # Indexer weights live only in full-indexer layers.
        dsa_weight_per_full_layer = 0
        if dsa_active:
            dsa_weight_per_full_layer = (
                bytes_dsa_q_idx_proj_weight(d_q_latent, q_idx_heads, d_idx)
                + bytes_dsa_k_idx_proj_weight(d, d_idx)
                + bytes_dsa_head_weight(d, q_idx_heads)
            )

        # ---- KV-cache reads (per sample) ----
        # Main latent KV: every layer. Indexer K-cache: only full layers.
        main_kv_per_layer   = bytes_kv_cache_read(d_kv_latent, d_h_rope, n_eff)
        idx_kv_per_full     = bytes_dsa_k_index_cache_read(d_idx, N) if dsa_active else 0

        # ---- Aggregate (per-GPU per-step), TP-sharded ----
        flops = (
            per_layer_per_token * L
            + dsa_per_full_layer * n_full
        ) * bs / TP

        bytes_ = (
            per_layer_weight_bytes * L                       # weights, every layer
            + dsa_weight_per_full_layer * n_full             # indexer weights
            + main_kv_per_layer * L * bs                     # main KV, every layer
            + idx_kv_per_full * n_full * bs                  # indexer K, full layers
        ) / TP

        return StageCost(flops=float(flops), bytes_=float(bytes_))

    # ------------------------------------------------------------------
    #  Capacity-analysis API — indexer terms scale with n_full
    # ------------------------------------------------------------------

    def session_kv_bytes(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Per-GPU session KV: main latent KV (every layer) + indexer
        K-cache (only n_full layers under IndexShare)."""
        del parallel  # KV replicated across TP (MLA)
        cfg = self.merged_config(model_overrides)
        L           = int(cfg["num_layers"])
        d_kv_latent = int(cfg["mla_kv_latent_dim"])
        d_h_rope    = int(cfg["mla_rope_head_dim"])
        S           = int(session_length)

        # Main latent KV over all layers (d_idx=None -> no indexer term).
        main = session_kv_bytes_closed_form(
            session_length=S,
            n_layers=L,
            d_kv_latent=d_kv_latent,
            d_h_rope=d_h_rope,
            d_idx=None,
            latent_bytes=latent_kv_bytes_for_dtype(dtype_kv),
        )
        # Indexer K-cache only in full-indexer layers.
        indexer = 0.0
        if cfg.get("dsa_topk") is not None:
            n_full = _num_full_indexer_layers(L, cfg.get("index_share_group", 1))
            d_idx  = int(cfg["dsa_idx_head_dim"])
            indexer = n_full * S * d_idx * _FP8_BYTES
        return float(main + indexer)

    def indexer_kv_bytes_per_session(
        self,
        session_length: int,
        parallel,
        dtype_kv,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """DSA indexer K-cache per session — only n_full layers (IndexShare).

            indexer_kv = n_full × S × d_idx × FP8,   n_full = ceil(L / g)
        """
        del parallel, dtype_kv
        cfg = self.merged_config(model_overrides)
        if cfg.get("dsa_topk") is None:
            return 0.0
        L      = int(cfg["num_layers"])
        n_full = _num_full_indexer_layers(L, cfg.get("index_share_group", 1))
        d_idx  = int(cfg["dsa_idx_head_dim"])
        return float(n_full * int(session_length) * d_idx * _FP8_BYTES)

    def weight_bytes_per_gpu(
        self,
        parallel,
        dtype_param,
        model_overrides: Mapping[str, Any] | None = None,
    ) -> float:
        """Static per-GPU weight bytes. Same as GLM-5 except the indexer
        weights live only in n_full layers (IndexShare)."""
        del dtype_param
        cfg = self.merged_config(model_overrides)

        L  = int(cfg["num_layers"])
        TP = max(1, int(parallel.tp))
        EP = max(1, int(parallel.ep))

        d           = int(cfg["hidden_size"])
        n_h         = int(cfg["num_q_heads"])
        d_h         = int(cfg["head_dim"])           # NoPE 192
        d_h_v       = int(cfg["value_head_dim"])     # V 256
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

        # ---- Attn weights: MLA projections in every layer ----
        mla_per_layer = (
            bytes_q_down_proj_weight(d, d_q_latent)
            + bytes_q_up_proj_weight(d_q_latent, n_h, d_h, d_h_rope)
            + bytes_kv_down_proj_weight(d, d_kv_latent, d_h_rope)
            + bytes_k_up_proj_weight(d_kv_latent, n_h, d_h)
            + bytes_v_up_o_proj_weight(n_h, d_kv_latent, d_h_v, d)
        )
        attn_total_bytes = mla_per_layer * L
        if dsa_present:
            n_full = _num_full_indexer_layers(L, cfg.get("index_share_group", 1))
            dsa_per_full = (
                bytes_dsa_q_idx_proj_weight(d_q_latent, q_idx_heads, d_idx)
                + bytes_dsa_k_idx_proj_weight(d, d_idx)
                + bytes_dsa_head_weight(d, q_idx_heads)
            )
            attn_total_bytes += dsa_per_full * n_full
        attn_total_bytes = attn_total_bytes / TP

        # ---- MoE weights (identical to GLM-5) ----
        per_expert_w = bytes_moe_per_expert_weight_all_layers(d, expert_intermediate, L)
        router_w     = bytes_moe_router_weight_all_layers(d, n_routed, L)
        routed_per_gpu = (n_routed / EP) * per_expert_w
        shared_per_gpu = n_shared * per_expert_w
        router_per_gpu = router_w
        ffn_total_bytes = routed_per_gpu + shared_per_gpu + router_per_gpu

        return float(attn_total_bytes + ffn_total_bytes)
