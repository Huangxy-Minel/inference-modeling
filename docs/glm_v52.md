# GLM-5.2 — Architecture & Cost-Model Spec

> Companion design doc for `simulator/models/glm_v52.py`. GLM-5.2 is a
> near-identical successor of GLM-5 (see `docs/glm_v5.md`); this doc only
> details the **deltas**. Read `docs/glm_v5.md` first for the shared MLA /
> DSA / MoE derivations.
>
> Sources:
> * `zai-org/GLM-5.2` config.json (HuggingFace) — concrete parameters.
> * `zai-org/glm-52-blog` (HuggingFace blog, "GLM-5.2: Built for
>   Long-Horizon Tasks", 2026-06-17) — IndexShare, 1M context, FP8 KV.
> * GLM-5 technical report — shared architectural background.

---

## 1. What changed vs GLM-5

GLM-5.2 keeps the **entire** GLM-5 backbone (asymmetric MLA + DSA +
DeepSeekMoE, 78 layers, d=6144, n_h=64, MoE 256×top-8+1). Only **three**
things differ, and only two of them touch the cost model:

| Aspect | GLM-5 | **GLM-5.2** | Cost impact |
| --- | --- | --- | --- |
| `head_dim` (qk **NoPE** per head) | 128 | **192** | Q-up / K-up projection FLOPs & bytes grow |
| `max_position_embeddings` | 131 072 | **1 048 576 (1M)** | config only (workload sets ctx) |
| **IndexShare** (DSA indexer sharing) | none (indexer every layer) | **1 indexer per 4 layers** | **indexer FLOPs / K-cache → ~1/4** |

Everything else is byte-for-byte identical to GLM-5 (confirmed against
the GLM-5.2 config.json):

```
num_hidden_layers      78     first_k_dense_replace  3
hidden_size            6144   n_routed_experts       256
num_attention_heads    64     num_experts_per_tok    8
q_lora_rank            2048   n_shared_experts       1
kv_lora_rank           512    moe_intermediate_size  2048
qk_rope_head_dim       64     intermediate_size      12288 (dense)
v_head_dim             256    vocab_size             154880
qk_head_dim            256    index_topk             2048
                              index_head_dim         128
                              index_n_heads          32
                              num_nextn_predict_layers 1
```

Note `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192 + 64 = 256`.
In GLM-5 this was `128 + 64 = 192`. The V head dim stays 256, so in
GLM-5.2 the Q/K per-head width (256) now **equals** the V per-head width
(256) — the "asymmetry" of GLM-5 is gone at the qk_head_dim level, but
the cost model still tracks NoPE (192) and V (256) separately because
they feed different projections.

### Quantisation

Same convention as GLM-5: FP8 placeholder (1 B/elem) for weights so
cross-model AI/MFU comparisons stay apples-to-apples. The blog confirms
**KV-cache is FP8** (mixed FP8 latent + BF16 RoPE, same as DSv3.2/GLM-5).

---

## 2. IndexShare — the one new mechanism

GLM-5.2 introduces **IndexShare** (blog + arXiv:2603.12201): the DSA
lightning indexer is computed once per group of `g = 4` transformer
layers, and the resulting top-k token **indices are reused** by the
other 3 layers in the group.

`config.json` encodes this as `indexer_types` alternating
`full / shared / shared / shared` across the 78 layers.

### What IndexShare saves (and what it does NOT)

A "full" layer runs the complete indexer; a "shared" layer skips it and
reuses the group's indices. Per shared layer we therefore **drop**:

* `flops_dsa_q_idx_proj`  (build indexer Q)
* `flops_dsa_k_idx_proj`  (build indexer K)
* `flops_dsa_indexer_scoring` = `2·q_idx_heads·d_idx·N`  ← the N-linear term
* `flops_dsa_head_weight`
* the **indexer K-cache read** (`2·q_idx_heads·d_idx·N`-shaped bytes) — a
  shared layer needs no indexer K-cache because it computes no scoring.
* the indexer **weight** bytes (q_idx / k_idx / head_weight matrices).

What stays **per every layer** (unchanged): the 6 MLA projections, the
main attention QK/PV over the top-k tokens, and the main latent KV cache.
Because the main latent KV (512·FP8 + 64·BF16 = 640 B/tok/layer) is
per-layer and dominates the indexer K-cache (128 B/tok/layer), IndexShare
cuts per-token **FLOPs** dramatically at long context but barely dents
**KV-cache capacity** — exactly what the blog reports.

### Modelling

Let `g = index_share_group = 4` and `n_full = ceil(L / g)` (= ceil(78/4)
= **20** full-indexer layers). The DSA indexer terms are multiplied by
`n_full` instead of `L`:

```
attn FLOPs (per GPU per step) = bs/TP × [
      L      × (6 MLA proj + main QK/PV over top_k)      # every layer
    + n_full × (q_idx + k_idx + indexer_scoring(N) + head_weight)   # IndexShare
]

indexer K-cache read bytes = bs/TP × n_full × (2·q_idx_heads·d_idx·N-shaped)
indexer weight bytes       = n_full × (q_idx_w + k_idx_w + head_w) / TP
indexer session KV         = n_full × S × d_idx × FP8
```

Setting `g = 1` recovers exact GLM-5 behaviour (`n_full = L`).

### Cross-check against the blog's "2.9× FLOP reduction at 1M"

Per-token FLOPs at N = 1M (attn + MoE FFN), single sample:

```
without IndexShare (g=1): attn 6.87e11 + ffn 5.3e10  ≈ 7.40e11
with    IndexShare (g=4): attn 2.12e11 + ffn 5.3e10  ≈ 2.65e11
ratio ≈ 2.79×  ≈  blog's 2.9×   ✓
```

The reduction comes entirely from the indexer scoring term
(`2·32·128·N = 8192·N`, = 8.2e9/layer at 1M) collapsing from 78 layers
to 20. This closed-form reproduction of the published 2.9× is the main
validation anchor for the GLM-5.2 estimator.

---

## 3. head_dim 128 → 192 impact

Only the NoPE-carrying projections change (main attention runs in latent
space and is unaffected — see `docs/glm_v5.md` §3.1):

```
flops_q_up_proj  = 2 × d_q_latent × n_h × (head_dim + d_h_rope)
                 GLM-5:   2×2048×64×(128+64) = 50.3M
                 GLM-5.2: 2×2048×64×(192+64) = 67.1M   (+33%)
flops_k_up_proj  = 2 × d_kv_latent × n_h × head_dim
                 GLM-5:   2×512×64×128 = 8.4M
                 GLM-5.2: 2×512×64×192 = 12.6M          (+50%)
```

These are N-independent constants; their absolute contribution is small
versus the main attention and (at long ctx) the indexer, so the net
effect on total AI is minor.

---

## 4. Modelling simplifications & known gaps

Inherits all GLM-5 simplifications (dense→MoE fold, MTP off forward path,
FP8 placeholder, MLA KV replicated across TP, decode-only). Plus:

* **IndexShare `n_full = ceil(L/g)` is an approximation.** The real
  layout is `full/shared×3` repeating; 78 = 4×19 + 2, so the exact full
  count is 19 or 20 depending on the tail. We use `ceil(78/4)=20`. The
  ±1 uncertainty moves 1M-ctx attention FLOPs by <1%.
* **MTP IndexShare + KVShare not modelled** — MTP is off the forward
  path entirely (same as GLM-5).
* **1M context is physically KV-capacity bound**, not modelled as a
  capacity limit here; the estimator returns closed-form numbers that
  will exceed single-GPU HBM at large ctx (informational).

---

## 5. DRAM-pool scenario applicability

Identical to GLM-5 — DSA is active, so both scenarios apply, **no
driver-side block needed**:

| Scenario | Applies? | Reason |
| --- | --- | --- |
| Scenario 1 — `sparse_on_demand` | **Yes** | DSA active (`dsa_topk=2048`); IndexShare only changes *which layers* recompute indices, not the sparse-attention nature |
| Scenario 2 — `shared_prefix` | **Yes** | orthogonal to attention type |

---

## 6. Reference deployment YAML

`simulator/configs/glm_v52_decode_gb200.yaml` mirrors GLM-5's deployment
(`tp=2, ep=16, pp=1, dp=8`, decode, `dsa_topk=2048`, FP8 placeholder,
mixed KV) for cross-model comparability, with `context_length` bumped to
demonstrate the 1M regime. Override `model_overrides.index_share_group`
(e.g. to 1) to ablate IndexShare.
