# GLM-5.3-Flash on sabrewing

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is 321B total / ~18B active:
45 layers, 288 routed experts (top-8) + 1 shared, vocab 154,880, natively multimodal (a 24-block
ViT this engine does not run). Engine: `c/glm53.c`. Reference: transformers' `glm5_next`
(5.16+), against which it is token-exact on a tiny fixture.

Despite the name it shares almost nothing with GLM-5.2 (`c/colibri.c`). Its skeleton is Kimi's:
a hybrid of linear-attention layers and sparse full attention, over a DeepSeek-V4-style mHC
residual.

## What is in a layer

| | GLM-5.3-Flash | note |
|---|---|---|
| residual | mHC, 4 streams, 20 Sinkhorn iterations (`moe_hc.h`, shared with `deepseek.c`) | the final collapse is a plain **mean** of the streams; V4 learns one |
| 34 layers | **KDA** (Kimi Delta Attention): linear, per-**channel** decay | O(1) state per sequence: 64 heads x 128x128 floats |
| 11 layers | **DSA**: NoPE MLA (`qk_rope_head_dim` 0) behind a pooled lightning indexer | position lives only in KDA's decay and convs |
| FFN | 3 dense layers, then sigmoid-routed MoE + shared expert | clamped SwiGLU everywhere |

**KDA**, per head, per token:

```
q,k,v = SiLU(depthwise causal conv4(W x))        q, k L2-normed (eps INSIDE the sqrt), q *= 128^-0.5
g     = -5 * sigmoid(exp(A_log[h]) * (f_b(f_a(x)) + dt_bias))     per channel ("safe" lower bound)
beta  = sigmoid(b_proj x)
S     = diag(exp(g)) S;   S += k (x) ((v - k^T S) * beta);   o = q^T S
out   = o_proj( RMSNorm_head(o) * sigmoid(g_b(g_a(x))) )
```

The decay is per key channel, not per head as in Qwen3.5's gated delta net. That is the one
difference from `moe_linattn.h`'s kernel, and the fixture is checked to fail if it is collapsed
to one scalar per head.

**The pooled indexer.** Keys are grouped into pools of `index_kpool` (16) tokens. A pool's index
key is a per-channel softmax over its slots of `compress_gate . x + ape[slot]`, mixing the
tokens' LayerNormed index keys, so a pool is not forced to describe itself by its mean. A query
at position P scores every pool that ends at or before P by `sum_h w_h relu(q_h . key / sqrt(d))`,
keeps the top `index_topk / index_kpool` (128 at the real config, i.e. 2048 tokens), and attends
to those pools' tokens **plus the incomplete visible tail** (`(P+1) % 16` tokens). Below 128
complete pools nothing is dropped. `indexer_types: "shared"` layers reuse the previous DSA
layer's selection. The shipped config runs the indexer on every DSA layer, but it is implemented.

**MLA is absorbed.** The cache holds the 512-wide kv latent, about 3 KB per token per DSA layer
including the indexer's keys: 33 KB/token for the model, so 1.1 GB at 32k context. Expanded K/V
would be 1.39 MB/token. `kv_b` is split at load into its key block transposed per head (q to
latent space) and its value rows (latent to output). Each is one GEMM per head over a prefill.

**Routing.** Sigmoid scores, a selection-only `e_score_correction_bias` with optional group
gating, combine weights from the unbiased score renormalized and multiplied by 2.5, and experts
accumulated in ascending id (the reference's `index_add_` order). The SwiGLU clamp is
asymmetric: `gate <= 10`, `|up| <= 10`, on routed, shared and dense MLPs alike.

## Validation

```sh
cd c && make glm53 ARCH=native
make glm53-test          # needs torch + transformers >= 5.16 to build the fixture
make glm53-container     # FP8 checkpoint shape -> converter -> engine
```

**The oracle.** `tools/make_tiny_glm53.py` builds the fixture from transformers'
`Glm5NextForConditionalGeneration`. It has 3 KDA and 3 DSA layers (one with a shared indexer),
a dense layer and MoE, and a 21-token prompt: 5 complete pools against a 2-pool budget, so the
indexer really drops pools, plus a live tail.

```
logits: max|diff| 1.818e-06 over 256 (rel 9.81e-07 vs ref scale 1.853) — ok
split prefill (11+10): max|diff| 0.000e+00 — ok
teacher forcing: 31/31   greedy: 10/10   ppl 177.6596 (ref 177.6597, -0.00%)
ORACLE PASS
```

The three gates check different things:
- The teacher-forced pass is checked against HF's **chunked** KDA form.
- Greedy decode is checked against HF's **recurrent** form and every cache.
- The split prefill must land where one shot does.

**Mutation-tested.** Each of these 15 single-line bugs turns the oracle red:
- indexer: keeps every pool; keeps the worst pools; no tail; pool key without the gate or
  without the ape; k-norm without its bias; a shared layer selecting nothing
- KDA: decay per head; no output gate; no k l2norm
- the hc head as stream 0
- router: weights from the biased score; ignoring the bias
- the wrong MLA scale
- no SwiGLU clamp

Two of them passed a first draft of the fixture. The real `swiglu_limit` (10) never fires at
tiny activations, so the fixture uses 0.5. A small k-norm bias only shifts pool scores
uniformly, so the fixture's is larger.

**The container.** `make glm53-container` runs the real converter on an FP8-checkpoint-shaped
fixture: 185 e4m3 tensors in 16x16 blocks, with a fresh transformers reference on the exact
dequantized weights.
- `--xbits 0` is lossless by construction and passes that oracle exactly (logits rel 9.5e-07,
  31/31, 10/10).
- The int4 container is line-identical to the f32 engine on its own int4 round-trip twin. That
  checks the plumbing; how much int4 costs is a separate number.
- On a random tiny model int4 costs +5.3% ppl. The real model's cost needs the real
  checkpoint.

## Running the real checkpoint

```sh
hf download zai-org/GLM-5.3-Flash --local-dir ~/models/glm53_fp8     # ~328 GB, FP8
python3 c/tools/convert_glm53_int4.py --indir ~/models/glm53_fp8 --outdir ~/models/glm53_i4
cd c && make glm53 ARCH=native            # CUDA=1 puts the bf16 residents in VRAM
SNAP=~/models/glm53_i4 ./glm53 -p "..." -n 128
python3 openai_server.py --engine ./glm53 --model ~/models/glm53_i4    # arch auto-detected
```

The container holds bf16 residents (~18 GB) and per-row int4 routed experts: 288 x 42 layers x
12.6 MB is about 152 GB. All of it loads into RAM, so a 187 GB box holds it with little to spare.

The gateway's `glm53` renderer is ported from upstream colibri's. Upstream pins that renderer
byte for byte against the checkpoint's `chat_template.jinja`, and the port matched it on 54 of
54 renders. GLM-5.3 always opens `<think>`; there is no thinking-off prompt. The gateway always
splits the reasoning out of `content` and surfaces it as `reasoning_content` (or an Anthropic
thinking block when one was asked for).

## Not done yet

None of this has run on the real checkpoint. All of it is validated on fixtures.

- **The int4 cost.** Per-row scales are this repo's container; upstream uses 64-wide groups. On
  a 3-mantissa-bit FP8 source, per-row int4 may cost more than it did for Laguna's bf16 source.
  Measure perplexity on real text after converting, before trusting it.
- **CUDA.** Only the bf16 residents move to VRAM. There is no expert cache and no batched decode
  or continuous batching (`KV_SLOTS`). Expected single-stream speed, bandwidth-bound: ~2 tok/s
  CPU-only, ~5–7 with the A6000 holding residents and some experts. The long-context, many-agent
  case is where this model shines: 33 KB/token of KV vs Laguna's ~140 KB.
- **The indexer is quadratic in prefill.** Every query scores every visible pool. That is fine
  at agent-turn lengths and slow at 100k.
- **Not implemented:** the MTP layer (index 45) and vision.
