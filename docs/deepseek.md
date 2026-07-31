# DeepSeek-V4-Flash on sabrewing

[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) is 284B total /
13B active, 43 layers, 256 experts (top-6) + 1 shared, 1M context. It ships natively as
FP4 routed experts + FP8 dense.

Use **`DeepSeek-V4-Flash-0731`**, the official release; the unsuffixed repo is the earlier
preview. 0731 ships a DSpark speculative-decoding module as an extra `mtp.*` stack (config
`dspark_*`, 3 layers, 46 `compress_ratios` for 43 layers). It is opt-in even in vLLM, this
engine does not implement it, and the converter skips it — the 43 main layers stand alone.

Engine: `c/deepseek.c`. **Stage B: the shipped fp4 checkpoint runs on CPU.** Oracle-green
against transformers, with an fp4 expert container and a bf16 resident tier. No CUDA tier,
no batching yet — the same order laguna and qwen35 landed in.

This is the first architecture here that shares almost nothing with the others. It is worth
reading `c/deepseek.c`'s header comment before touching it; the summary below is the map.

## What is actually different

| | V4-Flash | what the other engines assume |
|---|---|---|
| residual | `hc_mult`=4 **parallel streams** per token, mixed by mHC | one vector, `x += f(norm(x))` |
| attention | shared-KV MQA: **one** KV head for 64 query heads, and **K is V** | GQA, separate K and V |
| head_dim | 512, with partial **interleaved** RoPE on the trailing 64 | 64–256, half-split RoPE |
| output proj | grouped low-rank: 8 groups → 1024 each → mixed to hidden | one `[D, H*hd]` matmul |
| context | 128 sliding window **plus** a compressed KV series (CSA 4:1 / HCA 128:1) | one KV cache |
| routing | `sqrt(softplus(logits))`, and the first layers route by a frozen hash table | sigmoid / softmax |
| sinks | per-head learnable attention sink | none (laguna has an output *gate*, different thing) |

### mHC (Manifold-Constrained Hyper-Connections)

Each token carries 4 residual streams. At both sublayer sites a small learned map produces
`(pre, post, comb)` per token: `pre` collapses the 4 streams into the single vector the
sublayer consumes, `post` (range `[0,2]`) places the sublayer output back across the 4, and
`comb` — a 4×4 matrix projected onto the **doubly-stochastic manifold** by 20 Sinkhorn-Knopp
iterations — mixes the old streams forward:

```
streams[k] = post[k] * sublayer_out + sum_j comb[j][k] * streams[j]
```

`comb` is consumed **transposed** (the sum is over the *first* stream axis). Sinkhorn
produces a doubly-stochastic but *non-symmetric* matrix, so getting that direction wrong is
silent — it still normalizes, it just mixes the wrong way. A final `hc_head` collapses the
4 streams once before the output norm.

### Three attention types

Every layer has a 128-token sliding window. On top of that:

- **`sliding_attention`** — window only. Uses the *main* rope (θ=10000, no scaling).
- **`heavily_compressed_attention`** — plus a 128:1 compressed series. Every closed window
  of 128 tokens is pooled into one KV entry by a per-channel softmax over the window slots
  (gated by a learned `position_bias`), RMS-normed, and RoPE'd at the window's absolute
  start. A query sees every entry whose window closed before it: `e < (pos+1)/m`.
- **`compressed_sparse_attention`** — plus a 4:1 compressed series, of which each query
  attends to only the **top 512** entries chosen by a Lightning Indexer.

CSA's compressor emits **two** series per token in one `2*head_dim` tensor: `Ca` (the token's
contribution to the *next* window's entry) and `Cb` (to the *current* one). Entry `w` pools
window `w-1`'s Ca with window `w`'s Cb over `2m` slots — effective width `2m`, stride `m`.
Window 0 of a forward call therefore needs the *previous call's* last-window Ca, which is
the only cross-call state besides the running entry list and the partial-window buffer.

The compressed entries are concatenated onto the sliding keys and masked per query, so the
attention is one softmax over `[window ++ selected compressed]` with the per-head sink as an
extra logit that is normalized in and then dropped.

Both CSA/HCA layers and every compressor use the *compress* rope (θ=160000, YaRN factor 16
with `attention_factor` pinned to 1.0 — the reference does not rescale cos/sin).

### The traps

Each of these was read off `transformers/models/deepseek_v4/modeling_deepseek_v4.py`, not
guessed, and each produces plausible-looking activations when wrong:

1. **RoPE is interleaved** (pairs consecutive channels, GPT-J style), applied to the
   **trailing** `rope_dim` channels with the leading channels passed through.
2. **K == V, so the value carries RoPE.** The attention output is un-rotated by the
   **conjugate** rotation (`-sin`) at the *query* position before the output projection.
3. The compressor's window softmax is **per channel** across the window slots, and the gate
   has `position_bias` added *before* the softmax. The saved overlap slice is the
   already-biased gate.
4. Router combine weights come from the **unbiased** score; `e_score_correction_bias` only
   decides which experts win. Then renormalize, then scale by `routed_scaling_factor`.
5. The SwiGLU clamp is **asymmetric**: `gate.clamp(max=limit)`, `up.clamp(-limit, limit)`.
6. `hash_moe` layers take their expert ids from `tid2eid[input_ids]` — the *token id*, not
   the hidden state. The learned gate still supplies the combine weights.

## Validation

Three harnesses. All must be green; they answer different questions.

### 1. Dependency-free cross-check — runs anywhere, in under a second

```sh
cd c && make deepseek && make deepseek-ref
```

`tools/dsv4_ref.py` is a **pure-Python reimplementation of the whole tiny model** (no torch,
no numpy, no GPU, no weights to download). It writes a random-weight snapshot plus the
reference continuation, and `deepseek.c` must reproduce it. Current status on a 4-core CPU
container:

```
logits: max|diff| 6.528e-07 over 64 (rel 3.85e-07 vs ref scale 1.696) — ok
split prefill (6+7): max|diff| 0.000e+00 — ok
teacher forcing: 19/19   greedy: 6/6   ppl 45.4051 (ref 45.4051, -0.00%)
ORACLE PASS
```

Three gates, and the third is not covered by transformers' own oracle either:
- **logits** — a numeric comparison of the final logit row, so drift that argmax would round
  away still fails.
- **greedy** — `full_ids` is produced in Python by re-running the *whole prefix from scratch*
  each step, so the C engine's incremental compressor buffers, sliding ring and indexer
  state must reproduce a stateless recompute.
- **split prefill** — the same prompt in two chunks must land on identical logits. This is
  the only check on the compressor buffer and the CSA overlap slice crossing a call
  boundary with `n > 1`, which is the easiest thing in this architecture to get wrong.

The prompt length (13) is coprime with both compress rates, so both compressors carry a
partial window across the prefill → decode boundary.

**What it does not prove:** agreement with transformers. The two implementations were
written independently against the same reference source, so a shared *misreading* would
satisfy both. It catches transcription bugs, not comprehension bugs.

Mutation-tested, so the harness is known not to be vacuous — each of these fails it:
consuming `comb` untransposed, dropping the conjugate output rotation, and running CSA
layers with HCA-style non-overlapping windows.

### 2. The transformers oracle — the real gate

```sh
cd c && make deepseek-test        # or: python3 tools/make_tiny_deepseek.py /tmp/tds
SNAP=/tmp/tds ./deepseek /tmp/tds/ref_deepseek.json
```

Needs torch and a transformers carrying `deepseek_v4` (5.14.1 here; 5.3 has no such model).
The tiny config puts all three attention types and both router kinds in one 4-layer stack,
with YaRN on the compress rope and plain RoPE on the main rope.

```
logits: max|diff| 2.980e-08 over 64 (rel 1.50e-07 vs ref scale 0.199) — ok
split prefill (6+7): max|diff| 0.000e+00 — ok
teacher forcing: 21/21   greedy: 8/8   ppl 61.4384 (ref 61.4384, -0.00%)
ORACLE PASS
```

**Its first run failed, and found two bugs the dependency-free harness could not see.**
Both would have made the 284B checkpoint emit fluent-looking garbage:

1. **The output projection is `head.weight`, not `lm_head.weight`.** That is its name in
   the shipped checkpoint *and* in what `save_pretrained` writes for this arch. The loader
   looked only for `lm_head.weight`, found nothing, and silently fell back to the tied
   embedding — on a model with `tie_word_embeddings: false`. Every layer matched
   transformers activation-for-activation and only the logits were wrong, which is exactly
   how a silent fallback fails. The fallback is now conditional on the config actually
   tying, and a hard error otherwise.

2. **YaRN's `attention_factor` was dropped.** A comment here asserted V4 pins it to 1.0.
   It does not: transformers multiplies **both** cos and sin by `0.1*ln(factor)+1`, which
   is **1.2773** for the compress rope (factor 16). Scaling cos and sin together makes it a
   per-rotation *gain*, so RoPE stops being norm-preserving. Sliding layers use the main
   rope (not YaRN, factor 1.0) and stayed exact while every CSA/HCA layer drifted — that
   asymmetry is what localised it, after a per-layer activation diff showed the compressed
   entries matching in their leading channels but not in norm.

Why the pure-Python harness missed them: it writes its own snapshot, so it never exercised
the checkpoint's tensor names, and it **pinned `"attention_factor": 1.0`** in the config it
emitted, making the correct and incorrect behaviours coincide. It now computes the real
mscale. This is the "shared misreading" failure the harness's own docs warn about.

A third fixture bug surfaced on the way: `make_tiny_deepseek.py` used `index_n_heads=2`.
The indexer score is `sum_h w_h * relu(q_h . k_e)`; with two heads a random model relus
both to zero for most (query, entry) pairs, so whole score rows tie at exactly 0.0 and the
fixture measured **`torch.topk`'s tie-break** rather than the architecture. torch does not
define one (it returned the highest tied index; a stable index-ascending scan returns the
lowest), so that comparison is unwinnable and meaningless. Real V4-Flash has 64 indexer
heads, where an all-head tie is a 2^-64 event; the fixture now uses 8.

### 3. fp4 container equivalence — `make deepseek-fp4`

`tools/convert_deepseek_fp4.py`'s output can only be exercised by the 167 GB checkpoint, so
this builds the same container shape from a tiny snapshot **plus a twin holding the
dequantized values as plain f32**. Both describe identical weights, so the fp4 expert path
and bf16 resident tier must agree with the f32 path exactly:

```
fp4 container : logits: max|diff| 1.933e+00 ... teacher forcing: 14/19 ... got: 22 51 45 32 37 60
f32 twin      : logits: max|diff| 1.933e+00 ... teacher forcing: 14/19 ... got: 22 51 45 32 37 60
FP4-CONTAINER PASS
```

Comparing against the *unquantized* model instead would only measure quantization loss on
random weights (hence the large `max|diff|` both sides share, which is not the point). Any
*difference between the two columns* is a real plumbing bug: fused gate/up row order,
nibble order, block-scale stride, or the kernel's decode. Widths must be multiples of 32,
so the fixture is built with `DSV4_D=64 DSV4_MOE_I=32`.

`make kernel-check` separately proves `matmul_fp4_k` decodes correctly: the exact path is
bit-identical to a double-precision dequant reference, the AVX-512 path matches it to
2.1e-07, and the batched kernel is bit-identical to the S=1 kernel.

## Running the real checkpoint

```sh
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir /mnt/AZURA/DeepSeek-V4-Flash-0731
cd c && python3 tools/convert_deepseek_fp4.py \
    --indir /mnt/AZURA/DeepSeek-V4-Flash-0731 --outdir /mnt/AZURA/dsv4_fp4
SNAP=/mnt/AZURA/dsv4_fp4 ./deepseek -f chat.txt -n 64
```

Download 167 GB, convert ~40 min (I/O bound, ~55 s/layer), container ~161 GB.

**The conversion is lossless**, which is the whole point of the container choice:

- **Routed experts are copied VERBATIM.** The checkpoint is already fp4 (e2m1) with a
  per-32-block ue8m0 scale. sabrewing's existing int4 container is uniform 4-bit with one
  scale per row, and re-gridding onto it is *not* free — measured on real layer-0 experts,
  taking the shipped fp4 weights as ground truth:

  | container | relative error |
  |---|---|
  | int4, per-row scale | 17.0% |
  | int4, group-128 | 11.8% |
  | int4, group-32 (same granularity!) | 9.7% |
  | fp4 verbatim | 0% |

  The loss is not dynamic range — the per-row ue8m0 exponent spread is only ~1 power of 2.
  It is that e2m1's levels are non-uniform (`{0,.5,1,1.5,2,3,4,6}`), so mapping them onto
  int4's uniform ladder quantizes a second time. Keeping the source format also makes
  conversion a byte copy rather than a dequant/requant pass over 277B parameters. The
  nibble convention already matches (low = even column), so `moe_quant.h` only needed a
  new decode, not a new layout.
- **Dense fp8 → bf16 is exact too.** e4m3 has 3 mantissa bits and the block scale is a
  power of two, so bf16's 8 bits hold it exactly (verified: `rel=0.00e+00`).

Memory, measured: **~161 GB resident**, which is why the bf16 tier exists. Projections are
kept bf16 in RAM instead of expanded to f32 (`Wt` in `deepseek.c`, activations stay f32),
halving residents from ~25 GB to ~12.5 GB. Loading is `pread` + `fadvise(DONTNEED)`, so
page cache is not doubled. On the 187 GB box here that fits with ~25 GB spare — but only
with nothing else running.

The name mapping lives in `MAP`/`TOP_MAP` in the converter: the HF repo ships DeepSeek's
own inference layout (`layers.0.attn.wq_a`), not the transformers layout the engine was
written against (`model.layers.0.self_attn.q_a_proj`). Note the indexer nests the other
way round in the two schemes — native `attn.indexer.compressor.*` vs transformers
`compressor.indexer.*`.

## Known limits

- **CPU only, single stream.** No CUDA tier, no batching, no serve mode. Decode is a
  per-token, per-expert loop; `matmul_fp4_kb` (the batched, read-the-weight-row-once
  kernel) exists and is validated but nothing calls it yet — that is the group-by-expert
  lever, and it is what a batched decode path would use first.
- **Compressed KV grows without bound.** CSA keeps one f32 `head_dim` entry per 4 tokens
  per CSA layer — inherent to the architecture (it *is* the long-range memory), but at 1M
  context that is hundreds of GB in f32. This, not the weights, is what decides whether
  long context is reachable on one box.
- **Attention is O(window + compressed) per query, computed densely.** Correct, not fast:
  the indexer scores every compressed entry per query rather than exploiting sparsity.
- **Prefill is slow.** ~5 tok/s on 24 AVX-512 cores at this stage; the expert GEMV
  dominates and every routed expert is re-read per token.
- **The DSpark / MTP module is not implemented.** `mtp.*` is skipped at conversion. The
  main stack does not depend on it; it is speculative decoding, i.e. throughput, not
  quality.
- **The indexer's top-k tie-break is ours, not the reference's.** Ties go to the lowest
  entry index; `torch.topk` defines no order. Unreachable in practice at 64 indexer heads
  (see Validation), but it is a real, documented divergence rather than a proof of
  equivalence.
- `st.h` gained an `I64` dtype for the hash router's `tid2eid` table. Existing dtypes are
  untouched; the fp4 container reuses the existing `U8`/`I8` raw path.

## Next

1. **CUDA tier + batched decode**, reusing the grouped-expert GEMM from
   `backend_cuda_laguna.{cu,h}` (arch-generic already) and `matmul_fp4_kb` on the CPU
   side. This is where the throughput is: 256 experts at top-6 is exactly the shape the
   group-by-expert path was built for.
2. **Quantize the compressed KV series** — the long-context blocker above.
3. **Serve mode** (`KV_SLOTS`) once batching lands.
4. A GPU fp4 kernel: the experts are already in a format Blackwell decodes natively, so
   the container does not need re-converting for it.
