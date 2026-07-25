# What llama.cpp does that we should (and shouldn't) copy

Notes from reading [`poolsideai/llama.cpp`](https://github.com/poolsideai/llama.cpp)
branch `laguna` (04b2b72) — the fork Poolside ships their Laguna GGUFs for. Upstream
llama.cpp has **no** `laguna` arch as of master d6a1e18c (July 2026); support is in
review as [ggml-org/llama.cpp#25165](https://github.com/ggml-org/llama.cpp/pull/25165),
and the GGUF declares `general.architecture = laguna`, so mainline can't load it yet.

We did not benchmark against it (the 68 GB Q4_K_M was downloaded and deleted). This is
a code read: where it says *they* are faster, that is an argument from what the kernels
do, not a measurement. Where it reports numbers for **our** engine, those are measured
here — including one idea taken from them that turned out not to help (2a).

Their fork also doesn't build on gcc without a one-line fix (`std::isfinite` in
`common/speculative.cpp` with no `<cmath>`), which is a fair indicator of how fresh
this support is.

## Independent confirmations of our design

- **Their expert GEMM is our grouped GEMM.** `ggml_cuda_mul_mat_id` sorts the
  (token, expert) pairs by expert, gathers rows into a contiguous buffer, runs one
  GEMM per distinct expert over its rows, and scatters back through an inverse index.
  Same algorithm as `lag_moe_group`, arrived at independently. Good sign for the
  approach; less good as a differentiator.
- **SWA cache sizing.** `llama_kv_cache_iswa` sizes the sliding cache as
  `min(size_base, n_swa + n_ubatch)` padded to 256 — i.e. window **plus one
  micro-batch's span**, the exact invariant our tiny oracle taught us the hard way
  (a window-sized ring lets a pass's early rows overwrite history they still need).
  They bound the span with `n_ubatch`; we bound it with `kv_span()`. Two independent
  arrivals at the same rule is reassuring.
- **MoE block semantics.** Their `build_moe_ffn` takes sigmoid routing + score
  correction bias + sum-norm + `expert_weights_scale` as parameters, and the shared
  expert is summed in parallel — the same factoring as our `MoeDesc`/`moe_block`.
- **The GGUF metadata matches our converter's reading of the arch**: 48 layers, 256
  experts, top-10, sliding window 512, `expert_weights_scale 2.5`, sigmoid gating,
  1 leading dense layer.

## Where they are ahead, and what to take

**1. Per-expert GEMM quality (their biggest kernel edge).** Each expert's matmul goes
through MMQ: int8 tensor-core dot products, shared-memory tiling, per-block scales,
with per-architecture tuning tables (`mmq-config-ampere.cuh` etc.). Ours is one warp
per output row with f32 accumulate over unpacked nibbles. For prefill and large
batches — where arithmetic intensity is high — theirs should win comfortably. Taking
the *shape* of it (quantize activations to int8 per block, dp4a/tensor-core inner
product, tile in shared memory) is a real project, not a port, and it is the highest
kernel-level ceiling we have left on the GPU tier.

**2a. Online-softmax attention — TRIED, measured NEUTRAL.** Implemented as
`MOE_QK_ONLINE`: one pass over the K/V window keeping a running max/sum, no score
buffer. The reasoning for it was *wrong*. The win was supposed to be halved memory
traffic, but a sliding window is ~2 MB per stream per layer and stays in cache between
the two passes, so there was no DRAM traffic to save. The first version came out 11%
**worse** — it rescaled a 128-float accumulator on every position, doubling the
value-accumulation work. Branching so the rescale is paid only when the running max
actually rises (O(log n) times, not n) recovered that, and then it was a wash:

| B=8 @2k ctx, interleaved A/B/A/B | attention ms/step | aggregate tok/s |
|---|---|---|
| two-pass SIMD | 176.1, 170.1 | 20.42, 20.48 |
| online, branched rescale | 178.8, 164.2 | 20.23, 21.25 |

Flash attention's win is about not materializing an S×N score matrix in HBM, and about
tiling many query rows — neither applies to single-query CPU decode, where the score row
is a few KB and cache-resident. **Two-pass stays the default.** `LAG_ATTN_ONLINE=1`
keeps the path, because it is the right structure once K/V is quantized (one pass = one
dequant instead of two) and for a future GPU kernel.

**2b. Quantized KV — the lever that is still open.** int8 K/V with a per-row scale cuts
the bytes the attention phase touches and stacks with the sliding-window ring:
1.11 GB/slot → ~0.28 at 8k context. Unlike 2a this is not a guess about where the cost
is — it reduces bytes moved, and attention is 41-45% of a 2k-context step. It is lossy,
so it needs a perplexity gate rather than just a speed number.

**3. Quant recipe by tensor role.** Their Q4_K_M is not uniform: routed experts are
Q4_K **with an imatrix**, while the "signal path" (attention, shared experts,
embeddings) stays Q8_0 — 68 GB total. Ours is per-row symmetric int4 experts (no group
scales, no importance weighting) with bf16/int8 residents. Their recipe is precisely
pillar 5 of `moe-runtime-plan.md`, already validated in a shipping artifact: group
scales for the experts, high precision for everything the signal flows through. Worth
copying as a *policy* even before we implement K-quant-style group scales.

**4. Drafter capture points.** The graph stores every layer's input
(`res->t_layer_inp[il]`) and can keep pre-norm hidden states for all tokens
(`embeddings_nextn_masked`), so EAGLE3/DFlash drafters can consume mid-network
features; their DFlash drafter is a separate 2.2 GB model driving
`--spec-draft-n-max 15`. Our MTP is a single depth-0 head. If we revive it, copy the
capture-point design rather than inventing one.

## Where we are ahead

**Grouping costs them two host syncs per MoE layer; it costs us none.** Because
llama.cpp routes on the GPU, `mul_mat_id` copies the expert ids back to the host,
syncs, builds the sort on the CPU, copies the index arrays back, and syncs again —
twice per MoE layer, ~94 syncs per token for Laguna. We route on the CPU, so the sort
is already in host memory: the whole layer is one submission with **one** sync. At
decode batch sizes, where launch and sync latency dominate over arithmetic, that is a
structural advantage of doing routing where the scheduler lives — and it is an
argument for keeping routing host-side even when the rest moves to the GPU.

They also launch one GEMM per expert; we launch one grouped kernel with `grid.y` over
groups.

## A real gap: Laguna M.1

Their loader supports **two** Laguna shapes, distinguished at load time by the g_proj
output dimension:

| | S-2.1 (what we support) | M.1 |
|---|---|---|
| attention | hybrid 512-sliding / full | all full attention |
| output gate | per-**head** (`g_proj → n_head`) | per-**element** (`g_proj → n_head*head_dim`) |
| leading dense layers | 1 | 3 |

Checking `laguna.c` against each row rather than assuming:

- **Gate width — was a real gap, now FIXED.** `resmm(gate, ...)` hardcoded the output
  dim to `n_head`, so a `[n_head*head_dim, D]` g_proj would have been **silently
  mis-read** (first `n_head` rows, broadcast over head_dim) rather than rejected. The
  width now comes from the checkpoint (`Layer.wg_out = st_numel/D`), with per-head
  broadcast when it equals `n_head`, elementwise multiply when it equals
  `n_head*head_dim`, and a loud exit for anything else.
- **All-full attention — already fine.** `is_sliding[]` comes from `layer_types`, so a
  config with no `sliding_attention` entries yields an empty sliding schedule and the
  KV ring simply doesn't engage.
- **Leading dense layers — fine, now less fragile.** We derive dense layers from
  `mlp_only_layers` (`[0]` in S-2.1). `cfg_load` now falls back to `mlp_layer_types`
  (`['dense','sparse',...]`) when that key is absent, so a checkpoint shipping only the
  more explicit key no longer loads every layer as MoE.

Both are closed. The oracle covers the per-head path; the per-element path has no
checkpoint to test against yet, so it is written to be obviously right rather than
validated — and anything that is neither shape now exits with a message instead of
guessing.
