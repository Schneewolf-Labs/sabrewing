# DeepSeek-V4-Flash on sabrewing

[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is 284B total /
13B active, 43 layers, 256 experts (top-6) + 1 shared, 1M context. It ships natively as
FP4 routed experts + FP8 dense.

Engine: `c/deepseek.c`. **Stage A: f32 CPU forward pass, dependency-free.** No quantized
container and no CUDA tier yet — the same order laguna and qwen35 landed in (oracle first,
then the container, then the GPU tier).

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

Two harnesses. Both must be green; they answer different questions.

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
satisfy both — and it writes its own snapshot, so it cannot catch a wrong assumption about
how a real checkpoint is *named* or laid out. It caught none of the three bugs listed under
the transformers oracle below. It catches transcription bugs, not comprehension bugs.

Mutation-tested, so the harness is known not to be vacuous — each of these fails it:
consuming `comb` untransposed, dropping the conjugate output rotation, and running CSA
layers with HCA-style non-overlapping windows.

### 2. The transformers oracle — the real gate

```sh
cd c && make deepseek-test        # or: python3 tools/make_tiny_deepseek.py /tmp/tds
SNAP=/tmp/tds ./deepseek /tmp/tds/ref_deepseek.json
```

Needs torch and a transformers carrying `deepseek_v4` (5.14.1 has it). The tiny config puts
all three attention types and both router kinds in one 4-layer stack, with YaRN on the
compress rope and plain RoPE on the main rope.

**Green**, against transformers 5.14.1 / torch 2.13:

```
logits: max|diff| 2.980e-08 over 64 (rel 1.50e-07 vs ref scale 0.199) — ok
split prefill (6+7): max|diff| 0.000e+00 — ok
teacher forcing: 21/21   greedy: 8/8   ppl 61.4384 (ref 61.4384, -0.00%)
ORACLE PASS
```

2.98e-08 is f32 round-off: the reference itself moves 3.2e-08 between float32 and float64
on this fixture, so the engine agrees with it to the precision the reference has.

Both CI jobs run this: `engines` runs the dependency-free check (no torch needed), and
`deepseek-oracle` runs this one. It is a hard gate — `transformers` is unpinned, so an
upstream change to `deepseek_v4` should turn CI red rather than skip.

### What the oracle caught that the Python reference could not

Both were invisible to the dependency-free harness, because that harness writes its own
snapshot and so shares the engine's assumptions about naming and about YaRN:

1. **Expert weights are not stored the way the module declares them.** The module has fused
   `gate_up_proj [E,2I,D]` / `down_proj [E,D,I]`, but `save_pretrained` writes per-expert
   `w1` / `w3` / `w2`. The mapping (`w1`=gate rows, `w3`=up rows, `w2`=down) was resolved by
   comparing tensors *by value* against the fused parameter, not inferred from the names.
   The loader now accepts all three layouts in the wild, including per-expert
   `gate_proj`/`up_proj`/`down_proj`.
2. **The output projection is saved as `head.weight`, not `lm_head.weight`.** The engine's
   original "missing lm_head → fall back to the embedding" was a *silent* tie, which on this
   untied model quietly produced a wrong output layer. That fallback is now an error unless
   `tie_word_embeddings` is actually set.
3. **RoPE `attention_scaling` was missing entirely.** YaRN returns a scaling that multiplies
   cos *and* sin; V4 pins it to 1.0 only when the config is written flat, and a nested
   `rope_parameters` without `attention_factor` gets the derived `0.1·ln(factor)+1` instead.
   Ignoring it cost 8.7e-03 on the logits. Note this makes the "rotation" a scaled rotation,
   and the conjugate un-rotation on the attention output scales by it a *second* time rather
   than dividing it out — that is the reference's behaviour, faithfully reproduced.

### One known, principled divergence: indexer tie-breaks

The Lightning Indexer scores an entry as `sum_h w_h · relu(q_h · k)`, so a score is exactly
`0.0` whenever relu zeroes every head. At the top-k boundary the reference calls
`torch.topk`, whose order among **equal** scores is an artifact of its partial-sort rather
than a defined semantic — `[0]*7` with `k=2` returns `[5, 4]`, and the answer changes with
both `n` and `k`. That is not reproducible by any rule, and not worth reproducing.

This engine uses a deterministic rule instead: highest score first, ties to the lower entry
id. Where scores tie at the boundary the two implementations may keep different — and
equally unranked — entries.

Measured, so the claim is not hand-waving: with `index_n_heads=2` (where ~25% of entries
score exactly 0) the two disagree and the logits differ by 2.4e-02; at 4, 8 and 16 heads
they agree to 3.2e-08, 2.2e-08 and 1.5e-08. Everything else in CSA is bit-exact — forcing
`index_topk` above the entry count, so selection is unambiguous, gives 6.0e-08. The fixtures
therefore use 8 index heads (the real model has 64), so the gate tests the indexer's
arithmetic rather than torch's internal ordering.

`DS_IDX=1` dumps the per-query selections and scores if this ever needs revisiting.

## Known limits of Stage A

- **f32 only.** The real checkpoint is FP4 experts + FP8 dense, neither of which `st.h`
  reads; a converter is the next piece of work. Today's engine loads F32/BF16 tensors, so
  it runs tiny models and any bf16 re-export, not the shipped 284B container.
- **Compressed KV grows without bound.** CSA keeps one f32 `head_dim` entry per 4 tokens per
  CSA layer — inherent to the architecture (it *is* the long-range memory), but at 1M
  context that is hundreds of GB in f32 and needs the quantized container to be practical.
- **No batching, no serve mode, no CUDA tier.** Single stream only.
- **Attention is O(window + compressed) per query, computed densely.** Correct, not fast:
  the indexer scores every compressed entry per query rather than exploiting sparsity.
- `st.h` gained an `I64` dtype for the hash router's `tid2eid` table. Existing dtypes are
  untouched.

## Next

1. An int4/fp4 converter (`tools/convert_deepseek_int4.py`) so the shipped 284B container
   loads. The loader already accepts all three expert layouts, so this is about dtypes, not
   naming.
2. Quantize the compressed KV series, which is the thing that decides whether long context
   is reachable at all on one box.
3. CUDA tier + batched decode, reusing the grouped-expert GEMM from
   `backend_cuda_laguna.{cu,h}` (arch-generic already).
4. Run the oracle once against a *real* config.json rather than a synthesized one, to
   confirm the layer-type schedule and rope parameters of the shipped checkpoint parse as
   expected — the tiny fixture writes its own config, so it cannot cover that.
