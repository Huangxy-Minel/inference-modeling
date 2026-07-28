# GLM-5 — Architecture & Cost-Model Spec

> Companion design doc for `simulator/models/glm_v5.py`. Mirrors the
> docstring style of `simulator/models/deepseek_v3.py`.
> All numbers are sourced from the GLM-5 technical report:
>
> * `GPU:LLM Background/technical report/GLM-5- from Vibe Coding to
>   Agentic Engineering.pdf` (Zhipu AI / THUDM, 2025) — Table 10
>   (architecture comparison vs GLM-4.5), §2.1 (Architecture), §RL DSA
>   insights (top-k disclosure), §4 (Mixed-Precision deployment).

---

## 1. Architecture parameters

GLM-5 is a **MLA + DSA + DeepSeekMoE** architecture in the DeepSeek-V3 /
V3.2 lineage, with a key novelty: **asymmetric MLA — V head dim is wider
than K head dim** (the "MLA-256" variant).

| Symbol | Value | Description |
| --- | --- | --- |
| Total params | **744 B** | Table 10 |
| Active params | **40 B** | Table 10 |
| `num_layers` (L) | **78** | 3 dense FFN + 75 MoE (we exclude the 1 MTP layer from forward; see §4) |
| `num_dense_ffn_layers` | 3 | Same as DSv3; modelled as MoE for simplicity |
| `mtp_depth` | 1 | Inference: 1 MTP head, runs as 4 spec-decoding steps; **not** included in main forward path |
| `hidden_size` (d) | **6144** | DSv3 is 7168 |
| `num_q_heads` (n_h) | **64** | **Half of DSv3's 128** |
| `head_dim` (d_h) | **128** | K NoPE per-head dim (same as DSv3) |
| `mla_rope_head_dim` (d_h^R) | **64** | RoPE per-head dim (same as DSv3, inferred from QK Head Dim 192 = 128 + 64) |
| `value_head_dim` (d_h^V) | **256** | **★ V per-head dim — the asymmetric piece, 2× of K** |
| `mla_kv_latent_dim` (d_c) | **512** | Same as DSv3 |
| `mla_query_latent_dim` (d_c') | **2048** | DSv3 is 1536 |
| `expert_intermediate_size` | **2048** | Same as DSv3 |
| `num_routed_experts` | **256** | Same as DSv3 |
| `top_k_experts` | **8** | Same as DSv3 |
| `num_shared_experts` | 1 | Same as DSv3 |
| Dense intermediate (raw) | 12288 | Only used by 3 dense layers; **not** invoked because we treat all 78 layers as MoE (see §4) |
| `vocab_size` | **154 880** | Table 10 |
| `max_context_length` | **131 072** | 128K (training); mid-training extended to 200K |
| `dsa_topk` | **2048** | §RL "k = 2048 used by the indexer" |
| `dsa_q_idx_heads` | **32** | Indexer query heads; DSv3.2 is 64 |
| `dsa_idx_head_dim` | 128 | Same as DSv3.2 |

### Quantisation (real deployment vs cost-model default)

| Component | Real deployment (§4) | Cost-model default |
| --- | --- | --- |
| Standard attn + dense MLP `Linear` | **W8A8** (INT8) | FP8 placeholder (1 B/elem; semantically identical) |
| MoE expert `Linear` | **W4A8** (INT4 weight + INT8 act) | FP8 placeholder (1 B/elem) — **2× over-count** vs production |
| KV Cache | FP8 latent + BF16 RoPE + FP8 indexer | Same (matches DSv3.2) |
| Embedding / RMSNorm / SDPA / Router | BF16 | Same |

The cost model intentionally inherits DSv3's `_FP8_BYTES = 1` constant
so cross-model AI/MFU comparisons stay apples-to-apples. If someone
wants a production-faithful weight-bytes number, divide all expert
weights by 2 (W4 vs W8).

---

## 2. Differences vs DeepSeek-V3 / V3.2 / V4

| Aspect | DeepSeek-V3 | DeepSeek-V3.2 | DeepSeek-V4-Pro | **GLM-5** |
| --- | --- | --- | --- | --- |
| Attention type | MLA (dense) | MLA + DSA (sparse) | MLA + CSA (compressed sparse) | **MLA + DSA (sparse)** |
| K-side per-head dim | 128 | 128 | varies | 128 |
| **V-side per-head dim** | 128 | 128 | varies | **256 ★** |
| n_h | 128 | 128 | 128 | **64** |
| d_q_latent | 1536 | 1536 | 1536 | **2048** |
| d_kv_latent | 512 | 512 | 512 | 512 |
| Indexer Q heads | n/a | 64 | n/a (CSA) | **32** |
| DSA top-k | n/a | 2048 | n/a (CSA top-k different) | **2048** |
| MoE 256 × top-8 + 1 shared | ✓ | ✓ | 384 × top-6 (Pro) | **✓** |
| Hidden d | 7168 | 7168 | 5120 | **6144** |
| Layers (dense + MoE) | 3 + 58 (= 61) | 3 + 58 | 0 + 64 | **3 + 75 (= 78)** |
| Dense FFN handling in code | merge into MoE | merge into MoE | n/a | **merge into MoE (~2% over-count)** |

### Modelling implications of the asymmetric MLA

The DSv3 attention helpers parameterise the per-head dim as a single
`d_h`. GLM-5 has **two** per-head dims:

* **K-side `d_h = head_dim = 128`** — used by `flops_q_up_proj`,
  `flops_k_up_proj`, `bytes_q_up_proj_weight`, `bytes_k_up_proj_weight`.
* **V-side `d_h = value_head_dim = 256`** — used by `flops_v_up_proj`,
  `flops_o_proj`, `bytes_v_up_o_proj_weight`.

Both the FLOPs and bytes helpers in `deepseek_v3.py` already accept
`d_h` as a function parameter, so **no helper code change is required**;
the GLM-5 estimator simply calls the V-side helpers with
`value_head_dim` and the K-side helpers with `head_dim`.

The main attention core (`flops_attn_qk_dot`, `flops_attn_weighted_v`)
runs in **latent space** (W_UK absorbed) and depends only on
`d_kv_latent + d_h_rope` and `d_kv_latent` respectively — neither
depends on the V head dim. So those helpers stay verbatim.

### KV cache format is identical to V3.2

The V up-projection is materialised on-the-fly from the latent; **V is
not stored in KV cache** — only the latent + RoPE + indexer-K are.
Therefore:

```
per_token_per_layer_kv = d_kv_latent × FP8 + d_h_rope × BF16 + d_idx × FP8
                       = 512 × 1   + 64 × 2  + 128 × 1
                       = 768 B/token/layer
```

Identical to DeepSeek-V3.2. **The asymmetric MLA does not affect KV.**

---

## 3. Cost-model formulas

Per layer per token, single sample, before TP/EP/batch divisor.
Phase: **decode** (consistent with DSv3 estimator scope).

### 3.1 MLA attention (asymmetric)

```
# Six projections — K-side uses head_dim=128, V-side uses value_head_dim=256:
flops_q_down_proj  = 2 × d × d_q_latent
flops_q_up_proj    = 2 × d_q_latent × n_h × (head_dim + d_h_rope)        # K-side
flops_kv_down_proj = 2 × d × (d_kv_latent + d_h_rope)
flops_k_up_proj    = 2 × d_kv_latent × n_h × head_dim                    # K-side
flops_v_up_proj    = 2 × n_h × d_kv_latent × value_head_dim              # V-side ★
flops_o_proj       = 2 × n_h × value_head_dim × d                         # V-side ★

# Main attention (latent-space, V head dim irrelevant):
flops_attn_qk_dot      = 2 × n_h × n_eff × (d_kv_latent + d_h_rope)
flops_attn_weighted_v  = 2 × n_h × n_eff × d_kv_latent
```

### 3.2 DSA (always active under default deployment)

```
flops_dsa_q_idx_proj      = 2 × d_q_latent × q_idx_heads × d_idx
flops_dsa_k_idx_proj      = 2 × d × d_idx
flops_dsa_indexer_scoring = 2 × q_idx_heads × d_idx × N
flops_dsa_head_weight     = 2 × d × q_idx_heads
```

`n_eff` selection (matches DSv3.2):
* `dsa_topk = 2048` set AND `N >= 2048` → `n_eff = 2048` (sparse)
* otherwise → `n_eff = N` (dense fallback at very short prompts)

### 3.3 Attention bytes (same convention as DSv3)

```
bytes_qkvo_weight_per_layer = (
      d × d_q_latent                         # W_DQ
    + d_q_latent × n_h × (head_dim + d_h_rope)  # W_UQ ∪ W_QR (K-side)
    + d × (d_kv_latent + d_h_rope)           # W_DKV ∪ W_KR
    + d_kv_latent × n_h × head_dim           # W_UK (K-side)
    + d_kv_latent × n_h × value_head_dim     # W_UV (V-side, 256!)
    + n_h × value_head_dim × d               # W_O  (V-side, 256!)
) × FP8
+ DSA weight bytes (same as DSv3.2 form)
```

KV-cache read per token per layer (decoder step):
```
bytes_kv_cache_per_token_per_layer
    = d_kv_latent × FP8 + d_h_rope × BF16 + d_idx × FP8
    = 512 + 128 + 128 = 768 B
```

After TP-sharding the attention weights (head-sharded approximation,
same as DSv3): each GPU's attn-weight bytes = `total_attn_weight_bytes / TP`.
**MLA's KV cache stays replicated across TP** (same convention as DSv3
estimator — cross-TP shuffling is too expensive).

### 3.4 MoE (identical convention to DSv3)

```
total_tokens          = batch_per_GPU × EP
flops_router_per_tok  = 2 × d × n_routed
flops_per_expert_per_tok = 2 × 3 × d × expert_intermediate
flops_moe_per_layer_per_token = (
      flops_router_per_tok
    + (top_k + n_shared) × flops_per_expert_per_tok
)
total_flops = total_tokens × L × flops_moe_per_layer_per_token
```

Bytes (cluster-wide, divided by EP at the end):

```
per_expert_w_all_L = 3 × d × expert_intermediate × L × FP8
router_w_all_L     = d × n_routed × L × FP8
total_weight_bytes = (n_routed + EP × n_shared) × per_expert_w_all_L
                   + EP × router_w_all_L
total_act_bytes    = total_tokens × L
                   × bytes_moe_token_dispatch_combine_per_layer(d, top_k)
```

`bytes_moe_token_dispatch_combine_per_layer` reused **as-is** from
`deepseek_v3.py` (`2 × (1 + top_k) × d × BF16`).

### 3.5 Session-level KV (closed-form, capacity-analysis API)

```
per_token_per_layer = d_kv_latent × FP8 + d_h_rope × BF16 + d_idx × FP8
                    = 512 + 128 + 128 = 768 B
session_kv_bytes    = num_layers × S × 768
                    = 78 × S × 768
```

At `S = 131 072`:
```
78 × 131 072 × 768 = 7.85 GiB per session, MLA-replicated across TP.
```

For comparison, DSv3.2 at the same ctx is `61 × 131 072 × 768 = 6.14 GiB`
— GLM-5 is **~28% larger per session** (purely from `78/61` more layers).

### 3.6 Indexer K-cache & cold-layer KV (DRAM-pool capacity API)

```
indexer_kv_bytes_per_session   = num_layers × S × d_idx × FP8
                               = 78 × S × 128

cold_layer_kv_bytes_per_session = n_eff × (d_kv_latent × FP8 + d_h_rope × BF16)
                                = n_eff × 640                     # only main-attn KV, NOT indexer
where n_eff = min(S, dsa_topk)
```

This matches the DSv3.2 convention exactly.

### 3.7 Per-GPU weight bytes (capacity input)

```
attn_per_layer = bytes_q_down + bytes_q_up + bytes_kv_down + bytes_k_up
               + bytes_v_up_o + DSA weights                   # asymmetric: V-side uses 256
attn_total     = attn_per_layer × L / TP

routed_per_gpu = (n_routed / EP) × per_expert_w_all_L
shared_per_gpu = n_shared × per_expert_w_all_L                # replicated
router_per_gpu = router_w_all_L                                # replicated
ffn_total      = routed_per_gpu + shared_per_gpu + router_per_gpu

weight_bytes_per_gpu = attn_total + ffn_total
```

Reference numbers under default deployment (TP=2, EP=16, FP8 placeholder):

```
attn per layer (GLM-5): ~25.4 MB              # bigger than DSv3 due to wider Q-up + V-side 256
attn total (78 layers, TP=2):  ~989 MB
expert weight (per expert × 78 layers): ~2.94 GB
FFN per GPU = (256/16)*2.94 + 1*2.94 + 0.122 ≈ 50.0 GB
weight per GPU ≈ 51.0 GB                       # FP8 placeholder; W4A8 deployment ≈ 25.5 GB
```

(Exact numbers will be printed by the sanity-check script during
verification; treat the above as the target order-of-magnitude.)

---

## 4. Modelling simplifications & known gaps

These mirror the DeepSeek-V3 estimator's accuracy bar (~5% absolute):

* **Dense FFN (3 layers) merged into MoE.** Per user 2026-06-03 design
  decision (matching DSv3 simplification). Real dense-FFN intermediate
  is 12288 (~9 expert-equivalents); we treat each of those 3 layers as
  if it were MoE with `expert_intermediate=2048 × (8+1)=18432` per layer.
  Net effect: ~2% over-count on MoE FLOPs and bytes. Acceptable.

* **MTP (Multi-Token Prediction) excluded from forward.** GLM-5 has 1
  MTP head at inference time (running as 4 spec-decoding steps with
  ~2.76 mean accept length). We track `mtp_depth=1` in the config but
  do **not** include MTP-head FLOPs/bytes in `compute_attn_cost` /
  `compute_ffn_cost`. Same convention as DSv3 estimator.

* **TP simplification**: same as DSv3 — attn weights and KV bytes both
  divided by TP (head-sharded approximation).

* **No embedding / norm bytes** in `weight_bytes_per_gpu`.

* **No TPSP / async shared-expert overlap** — runtime optimisations
  that don't change FLOPs/bytes totals.

* **Activation bytes for MoE** count only EP all-to-all dispatch +
  combine (same conservative HBM floor as DSv3).

* **Prefill not modelled** — `compute_attn_cost` / `compute_ffn_cost`
  return zero cost when `workload.phase != decode`.

* **W4A8 MoE precision not modelled.** All weights treated as FP8
  (1 B/elem) for cross-model consistency. Real GLM-5 deployment halves
  the MoE expert weight bytes; production-faithful capacity numbers
  should multiply expert byte totals by 0.5.

* **Muon Split optimisation not modelled** — training-time only.

---

## 5. DRAM-pool scenario applicability

| Scenario | Applies to GLM-5? | Reason |
| --- | --- | --- |
| Scenario 1 — `sparse_on_demand` | **Yes** | DSA is active by default (`dsa_topk=2048`); cold-layer fetch can be capped at top_k tokens, exactly the sparse-attention regime targeted by Scenario 1 |
| Scenario 2 — `shared_prefix` | **Yes** | Cross-session shared prefix in DRAM with layer-stride prefetch is orthogonal to attention type; closed-form KV per layer (768 B/token, MLA-replicated) is what the analysis needs |

Both scenarios apply; no driver-side block needed (GLM-5 is sparse
attention with DSA active).

---

## 6. Reference deployment YAML

`simulator/configs/glm_v5_decode_gb200.yaml` mirrors the DSv3.2 sample
deployment for cross-model comparability:

* `tp=2, ep=16, pp=1, dp=8` (= 16 GPUs, EP = TP × DP)
* `phase=decode, batch_size=128, context_length=131072, new_tokens=1`
* `dsa_topk=2048` (DSA always active)
* `dtype_param=fp8` (W8A8/W4A8 placeholder), `dtype_kv=mixed_bf16_fp8`

Override `model_overrides` to sweep alternate configurations
(e.g. `value_head_dim`, `dsa_topk`, expert-bytes precision).
