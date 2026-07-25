# Arch-Descriptor Survey: colibri(GLM) / inkling / laguna / olmoe

Input for the Phase-1 descriptor design (see `moe-runtime-plan.md`). Maps how the
four MoE engines each handle attention / router / norm / quant, so the shared core
knows what is a safe default vs. what must be a descriptor field vs. what needs an
explicit escape hatch. Engine files: GLM = `colibri.c` (multi-arch mothership,
GLM paths only), `inkling.c`, `laguna.c`, `olmoe.c`.

All four share the same substrate (`st.h` safetensors, `tok.h`, `json.h`,
streaming-expert LRU cache, `openai_server.py` serve protocol) and diverge almost
entirely in the attention + router math.

## Safe shared defaults (all four agree)

Pre-norm topology (input_layernorm → attn → post_attention_layernorm → mlp →
model.norm), RMSNorm everywhere, causal softmax attention, top-k gated MoE, per-row
weight scales as baseline, SwiGLU (silu) expert activation, streaming-expert LRU.

## Must be descriptor fields (real variation between engines)

**Attention**
- RoPE type per layer-class: `{none (inkling), plain-partial (GLM), plain-full
  (olmoe), yarn-partial (laguna global)}` + θ / factor / beta_fast/slow /
  partial-fraction / yarn-mscale.
- Head-asymmetry model: `{uniform (GLM-MLA, olmoe), global-vs-sliding classes
  (inkling), per-layer-Q-count (laguna)}`.
- Sliding schedule: window + selector (`layer_types[]` string, or `(i+1)%6` rule).
- QK-norm scope: `{none, latent (GLM), per-head (inkling, laguna), whole-vector (olmoe)}`.
- Attention output gate: `{none, per-head-softplus (laguna)}`.
- **Attention scale: `1/d` (inkling, deliberate w/ qk-norm) vs `1/√d` (others)`.**
- head_dim source: explicit vs derived (`hidden/n_heads` in olmoe; `nope+rope` in GLM).

**Router / MoE**
- Activation: `{sigmoid-lossfree (GLM, inkling, laguna), softmax (olmoe)}`.
- Correction bias `e_score_correction_bias`: present (GLM/inkling/laguna) — used for
  *selection only*, combine weight is the unbiased sigmoid — vs absent (olmoe).
- Shared-expert semantics: `{none (olmoe), always-on-unscaled (laguna),
  always-on (GLM), gated-in-pool (inkling — shared experts normalized jointly with
  routed pool, NOT always-on)}`; shared count 0/1/2.
- gate_up layout: `{separate gate_proj/up_proj (GLM, olmoe), fused block-concat
  [E·2I,D] gate-rows-then-up-rows (inkling, laguna)}`.
- Routed scaling source: `routed_scaling_factor` (GLM 1.0?, laguna 2.5) vs
  `route_scale × per-layer gate.global_scale` (inkling 8.0) vs none (olmoe).
- norm_topk default: false (GLM, olmoe) vs true (laguna) vs folded-in (inkling).
- Dense-vs-MoE plan: `first_k_dense_replace` (GLM), `mlp_layer_types[]` (inkling),
  `mlp_only_layers[]` + layer-0-dense (laguna), all-MoE (olmoe).
- Group routing: GLM forces n_group==1; others none.

**Norm**
- eps default: 1e-5 (GLM, olmoe) vs 1e-6 (inkling, laguna).
- Optional embed_norm (inkling); non-RMS LayerNorm only in GLM's DSA indexer k_norm.

**Quant**
- Scale granularity: `{per-row (all baseline), grouped(gs) (GLM int4/int3)}`.
- Format enum: f32/bf16/int8/int4/int3-g64/E8-lattice (GLM richest; olmoe f32-resident
  + int8-expert only; inkling bf16-resident + int4/int8 expert; laguna bf16/int8
  resident + int4 expert).

## Escape hatches (do NOT reduce to a clean field)

- **inkling**: short-convolutions (depthwise causal conv1d k=4 at 4 sites: K, V,
  attn-out, mlp-out, with per-layer conv state across decode); learned
  relative-position bias replacing RoPE (`r_proj` D→H·d_rel vs per-layer bank);
  log-length attention scaling (`τ = 1 + α·ln((pos+1)/floor)`); muP logit scaling
  (÷24.0) + per-layer `mlp.global_scale`; nested `text_config`; MTP draft heads.
- **GLM**: MLA latent KV compression (q_lora/kv_lora, nope/rope/v head-dim split,
  cache stores latent only); DSA "lightning indexer" (sparse-key selection, own
  LayerNorm k_norm + RoPE); DeepSeek-V3 MTP layer.
- **laguna**: per-head softplus attention output gate; tied embeddings (lm_head
  falls back to embed_tokens).
- **olmoe**: whole-vector QK-norm (over all heads, not per-head).
- **cross-cutting**: multi-EOS lists (GLM 3 / laguna 4 / others 1); `intermediate_size`
  name-flip (dense vs MoE meaning depends on which sibling key is present);
  per-engine quant-bits CLI convention differs (inkling 0=f32; colibri ≥16=f32).

## Implication for extraction order (Phase 1, extract-first)

The math/attention/kv/serve substrate is *drifted copy-paste* — convergent but not
byte-identical (e.g. `rss_gb` encodes Linux-KB in laguna vs macOS-bytes in
inkling/olmoe; `softmax` uses double-accum in laguna vs float in inkling;
`rmsnorm`/`softmax` were renamed `_row`). So extraction reconciles per-function,
each step gated on all touched engines' token-exact oracles staying green. Start
with byte-identical, oracle-neutral helpers (rmsnorm, siluf, scalar utils); handle
the drifted ones (softmax float-vs-double, rss_gb platform) as deliberate
reconciliations verified by the oracle.
