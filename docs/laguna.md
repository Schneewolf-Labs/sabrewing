# Laguna (Poolside Laguna-S-2.1, 118B/8B MoE) on sabrewing

[Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) is Poolside's agentic
coding model — 118B total, ~8B active, 48 layers, 256 experts (top-10) + 1 shared.
Unlike Inkling's 975B (streamed from NVMe), Laguna's int4 weights **fit a single
48 GB GPU**, so it's the flagship for the A6000 tier and batched serving.

Engine: `c/laguna.c` over the shared MoE runtime (`c/moe_*.h`). Architecture: a
Qwen2-MoE / GLM / DeepSeek-V3 hybrid — GQA with interleaved 512-sliding / global
attention, per-head QK-RMSNorm, a per-head **softplus attention output gate**, RoPE
(YaRN partial-rotary on global layers, plain full rotary on sliding), a dense layer
0 + sigmoid loss-free-routed MoE (routed ×2.5, shared unscaled). No short-convs
(those are Inkling's).

## Quickstart

```sh
hf download nbeerbower/Laguna-S-2.1-colibri-int4 --local-dir ~/models/laguna_i4
cd c
make laguna CUDA=1 ARCH=native        # GPU tier; `make laguna` for CPU-only
SNAP=~/models/laguna_i4 ./laguna -p "def is_prime(n):" -n 128
```

Laguna is an instruct/RL model — raw base-completion prompts degenerate. Use the
chat template (`-f` with a rendered prompt, or the gateway):

```
〈|EOS|〉<system>…</system>\n<user>…</user>\n<assistant></think>
```

`</think>` right after `<assistant>` disables reasoning; `<think>` enables it. The
gateway (`openai_server.py --arch laguna`) applies this automatically.

## Container format

Produced by `c/tools/convert_laguna_int4.py` from the bf16 HF checkpoint:
- **Routed experts → int4**, per-layer fused: `experts.gate_up_proj` packed
  `[E·2I, D/2]` (gate rows then up rows — block-concat, not interleaved) + `.qs`
  f32 per-row scales; `experts.down_proj` `[E·D, I/2]` + `.qs`.
- **Residents → bf16** (attn q/k/v/o/g, qk-norm, dense layer-0 MLP, shared expert,
  embed, lm_head); norms / router / `e_score_correction_bias` stay f32.
- `laguna.c` auto-detects the int4 container (`experts.gate_up_proj.qs` present).

## A6000 GPU tier

`make laguna CUDA=1` brings up the A6000 (`backend_cuda_laguna.cu`):
- **bf16 residents → VRAM** (~8 GB); resident matmuls run on-device.
- **Expert VRAM cache**: each MoE layer's int4 expert blob (~1.2 GB) uploads
  greedily until VRAM fills — 35–37 of 47 layers fit; the rest use the fast CPU
  int4 path. Fused GPU expert kernel + batched per-layer submission (one sync).
- **`RES8=1`** stores residents int8 in VRAM (~4 GB) → ~2 more expert layers fit
  and half the resident VRAM bandwidth (~lossless).

The routed experts total ~57 GB int4, which does not fit 48 GB — so some layers
stay on the CPU. That capacity gap is the single-stream bottleneck.

## Speed levers (all opt-in; the oracle stays exact)

| Knob | Effect |
|---|---|
| `CUDA=1` build | residents + expert cache on the A6000 |
| `RES8=1` | int8 residents in VRAM (fits more experts, +bandwidth) |
| `LAG_IDOT=1` | CPU experts via int8-VNNI. The old ~+19% no longer reproduces: after grouped batching it measures **neutral** end to end, so it only costs ~0.4% quant noise. Off by default. |
| `BATCH=N` | batched-decode throughput bench (N copies of the prompt) |
| `BATCH_VARY=1` | bench with *divergent* streams (sampled, per-stream seed) — the honest multi-tenant number |
| `BATCH_AB=1` | bench both MoE paths in one process (+ asserts they are token-identical) |
| `LAG_NOGROUP=1` | disable group-by-expert batching (A/B knob; output is identical either way) |
| `LAG_GROUP_CHUNK=N` | rows per grouping chunk, default 128 (prefill is chunked; bounds scratch) |
| `LAG_PREFILL_CHUNK=N` | serve mode: prompt tokens prefilled per step, default 128 (the decoders' worst-case stall when a long prompt arrives) |
| `LAG_PROF=1` | phase breakdown of a decode step (where the wall time actually goes) |
| `LAG_KVFULL=1` | store full-length KV on sliding layers too (A/B; same output, 2.9x the memory) |
| `LAG_KV_SPAN=N` | positions per forward pass, default 128 — also sets the sliding ring size |
| `LAG_ATTN_EXACT=1` | attention QK in scalar double (the oracle contract) instead of AVX-512 |
| `LAG_ATTN_ONLINE=1` | single-pass online-softmax attention (measured neutral — see `docs/llamacpp-notes.md`) |
| `LAG_ATTN_GROUPED=0/1` | force GQA-grouped attention off/on (default: on when there are ≥24 work items) |
| `KV8=1` | int8 KV cache (1.11 → 0.29 GB/slot at 8k). **Lossy and not yet validated on the real model** — see below |
| `LAG_NOREUSE=1` | serve mode: disable KV prefix reuse across requests (it is ON by default) |
| `LAG_PREFIX_SNAPS=N` | shared prefix snapshots to keep (default 4 when reuse is on); each costs one slot's worth of KV |
| `TOKENIZE=1` | print the engine's own token ids for `-p`/`-f` and exit (build a trustworthy reference) |
| `TF_DUMP=<file>` | oracle harness: dump this run's teacher-forced argmax stream, to score another config against it |
| `CUDA_HEADROOM_MB`, `CUDA_EXPERT_GB` | tune the expert-cache VRAM budget |
| `LAG_GPU_MINEL` | min weight size to offload a resident matmul (default 0 = all) |
| `NOGPU=1`, `GPU_DEV=n` | disable GPU / pick device |

## Batched serving (the throughput win)

Single-stream decode is bandwidth/latency-bound (~12–16 tok/s) — one token has to
walk the whole model, so it is a latency problem no amount of compute fixes. The
axis that *does* scale is **throughput**, and it has two levers, both landed and
both bit-exact:

1. **Batched residents** (`forward_batch`) — decode B streams in lockstep so the
   resident weights (attention, dense, shared expert, lm_head) are read **once for
   the whole batch** instead of once per stream.
2. **Group-by-expert MoE** (`lag_moe_group`, Megablocks-style) — sort the B·top_k
   (row, expert) pairs by expert and run one GEMM per **distinct** expert, so an
   expert's ~1.2 MB of int4 weights is read once per step no matter how many
   streams routed to it. This is what carries past the batch-16 wall, and it also
   speeds up **prefill**, where a chunk of prompt tokens groups the same way.

```
BATCH=16 SNAP=~/models/laguna_i4 ./laguna -f chat.txt -n 48               # identical streams
BATCH=16 BATCH_VARY=1 SNAP=~/models/laguna_i4 ./laguna -f chat.txt -n 48  # divergent streams
LAG_NOGROUP=1 ...                                                          # lever 2 off (A/B)
```

Measured on the A6000 tier (`RES8=1`, 37/47 expert layers in VRAM, 40-token prompt,
48 tokens/stream). `BATCH_AB=1` times both MoE paths **in one process** — same
loaded model, same page cache, KV rewound between passes — because cross-process
A/B on a 57 GB model is far too noisy to trust. Aggregate tok/s:

| Batch | resident-batched only | + group-by-expert | gain | distinct experts / layer / step |
|---|---|---|---|---|
| 1 | 15.7 | 16.4 | — (S=1 doesn't group) | — |
| **identical streams** (`BATCH=N`) ||||
| 8 | 30.5 | **40.9** | +34% | 10 of 80 pairs (8.0× fewer reads) |
| 16 | 34.0 | **43.2** | +27% | 10 of 160 (16.0×) |
| 32 | 30.3 | **47.2** | +56% | 10 of 320 (32.0×) |
| **divergent streams** (`BATCH_VARY=1`) ||||
| 8 | 29.3 | **36.0** | +23% | 29.6 of 80 (2.7×) |
| 16 | 30.3 | **38.4** | +27% | 41.1 of 160 (3.9×) |
| 32 | 31.8 | **44.3** | +39% | 64.2 of 320 (5.0×) |

The shape is the point: **resident-only batching saturates at ~30–34 tok/s** and
B=32 is no better than B=16 (it is slightly *worse*) because every stream still paid
for its own expert reads. With grouping the curve keeps climbing — 40.9 → 43.2 →
47.2 — so the batch-16 wall moved. Real divergent traffic keeps most of it: even
with 32 independently sampled streams, 320 routed pairs collapse onto ~64 distinct
experts, because expert popularity is skewed and streams share context.

Prefill benefits too (a chunk of prompt tokens groups exactly like a batch of
streams) — on a 2139-token agentic prompt, **22.0 → 28.6 tok/s (+30%)**.

**Read the two columns honestly.** `BATCH=N` decodes N *copies* of one prompt, so
every stream routes identically and the distinct-expert count collapses to top_k
(10) — a best case for grouping, and the number a demo would quote. `BATCH_VARY=1`
samples each stream with its own seed so the streams genuinely diverge; the bench
prints the measured amortization (`grouping: … distinct experts per MoE layer per
step`) so the mechanism is visible rather than assumed.

Every stream stays **bit-identical** to single-stream decode: each dot keeps the
ungrouped reduction order and the per-token combine sums the selected experts in
top-k rank order, so grouping changes *when a weight is read*, never the arithmetic.
`LAG_NOGROUP=1` is the A/B knob and must produce identical tokens.

## Chat serving (OpenAI + Anthropic gateway)

```sh
python3 openai_server.py --arch laguna --engine ./laguna --model ~/models/laguna_i4
# http://127.0.0.1:8000 — /v1/chat/completions, /v1/messages, streaming, web UI
```

`--arch laguna` is auto-detected from `config.json` (`model_type: laguna`). The
gateway renders Laguna's chat template and splits reasoning between
`<think>`/`</think>`; model id `laguna-s-2.1-colibri`.

### Continuous batching (`--kv-slots N`)

```sh
python3 openai_server.py --arch laguna --engine ./laguna --model ~/models/laguna_i4 --kv-slots 8
```

Serve mode used to finish one request before starting the next, so the throughput
above was unreachable through the gateway: a second client simply waited. With
`--kv-slots N` (passed to the engine as `KV_SLOTS`) up to N requests stay in flight
— each slot owns a KV cache and its own sampling stream, and every decode step runs
all active slots through `forward_batch`, so they share one pass over the residents
and one grouped read per distinct expert.

Measured with 8 requests submitted at once (`tools/`-free driver over the raw serve
protocol, A6000 tier, temp 0.7 / top-p 0.95):

| 8 concurrent requests | serial (`KV_SLOTS=1`) | continuous (`KV_SLOTS=8`) |
|---|---|---|
| 48 tokens each — aggregate | 11.72 tok/s | **17.66 tok/s** (+51%) |
| 192 tokens each — aggregate | 11.21 tok/s | **19.73 tok/s** (+76%) |
| worst first-token latency (192) | 105.9 s | **7.3 s** (14.6× better) |

The tail-latency number is the one that matters for agents: serially, the eighth
request waits out the other seven before it emits anything.

`KV_SLOTS=1` keeps the old serial loop (lowest single-request latency). A slot's
tokens are what the serial path would produce; only the RNG stream differs, since
each slot samples from its own seed. `CANCEL` is honored at the next step boundary.

**Chunked prefill.** A newly admitted request used to prefill in one blocking call,
so an agentic 2k-token prompt arriving mid-flight froze every stream already decoding
for its whole prefill. The prompt is now fed `LAG_PREFILL_CHUNK` tokens per outer
iteration, interleaved with the decode steps of the running slots. Measured with 6
streams mid-generation when a 2139-token prompt arrives:

| prefill chunk | worst decoder stall | new request's TTFT | total wall |
|---|---|---|---|
| one-shot (`999999`) | 70.8 s | 70.5 s | 153.4 s |
| **128** (default) | **4.9 s** | 73.4 s | 154.0 s |
| 32 | 1.8 s | 96.8 s | 168.1 s |

128 buys a 14× smaller stall for ~4% of the newcomer's time-to-first-token and no
change in total wall; 32 halves the stall again but starts charging the newcomer
(+37% TTFT) because smaller chunks group worse. Chunking is **bit-identical** to
one-shot prefill — positions are absolute, attention reads the same cached K/V, and
every kernel reduces per row — so `LAG_PREFILL_CHUNK=999999` is an exact A/B knob for
the old behavior (verified: 4 requests × 40 greedy tokens byte-identical at chunk 17
vs one-shot).
- ~~The shared sampler qsorts the whole 100k-token vocabulary per token for
  top-p~~ **fixed**: `sample_logits` now takes a top-64 head by partial selection
  and only widens (finally deferring to the full sort) when the nucleus needs more,
  which is exact by construction and validated pick-for-pick in `make kernel-check`.
  Recovered the whole 13%: 8 × 96 tokens at temp 0.7 went **19.7 → 22.4 tok/s**,
  now level with greedy (22.7), so sampling is no longer a serving tax.

## Validation

- **Token-exact oracle**: `c/tools/make_tiny_laguna.py` builds a tiny random-init
  `LagunaForCausalLM` (needs `transformers >= 5.7`) and a reference; the engine
  matches it teacher-forced **36/36**, greedy **24/24**, perplexity within 0.00%.

  ```sh
  SNAP=/path/to/tiny_laguna ./laguna ref_laguna.json
  ```

- **Quant kernels** (not oracle-reachable — the oracle runs f32): `make kernel-check`
  validates the int4/bf16/int8 kernels against a double-precision dequant reference.
- The real 118B int4 model reproduces its CPU output byte-identically under the GPU
  tier and the batched path (residual float noise only in the mixed CPU/GPU expert
  case, within quant tolerance).
- **Group-by-expert batching** is covered at three levels:
  - the teacher-forced oracle prefills S=36 rows, so it runs *through* the grouped
    path — 36/36 and ppl 0.00% still hold, and `LAG_NOGROUP=1` gives identical
    output on every oracle config (f32 / int4 / `LAG_IDOT`);
  - `LAG_CUDA_TEST=1 ./laguna` compares the grouped CUDA kernels against the
    per-token ones on random weights: **max|abs| = 0** (bit-identical), no model
    needed;
  - `BATCH_AB=1` re-decodes the real 118B model both ways from a rewound KV and
    asserts the streams are token-identical (verified at B=8/16/32, greedy and
    sampled).

## Where a decode step's time goes (`LAG_PROF=1`)

Batched decode and grouped experts changed the shape of the problem, so the costs
that matter now are not the ones single-stream profiling found. All three columns are
one build, one measurement pass, nothing else on the machine (divergent streams,
A6000, `RES8=1`):

| phase | B=8, 88-tok ctx | B=32, 88-tok ctx | B=8, 2187-tok ctx |
|---|---|---|---|
| **experts CPU** (10 of 47 MoE layers) | **35.3%** | **32.2%** | 27.3% |
| **attention** (per-sequence) | 8.9% | 11.3% | **41.3%** |
| qkv+rope projections | 16.5% | 16.4% | 9.2% |
| experts VRAM (37 layers) | 12.5% | 13.1% | 7.1% |
| gate + o_proj | 9.7% | 10.5% | 5.5% |
| routing + group sort | 9.6% | 9.0% | 5.4% |
| shared expert | 4.6% | 4.0% | 2.6% |
| dense MLP / lm_head / sampling | 2.8% | 3.4% | 1.5% |
| ms per step | 211 | 741 | 402 |
| aggregate tok/s | 37.9 | 43.2 | 19.9 |

Two costs dominate, and which one depends on context length:

1. **The capacity gap.** The ~12 GB of routed experts that don't fit VRAM are 10 of
   47 MoE layers but ~⅓ of a short-context step — **7.3 ms per CPU layer vs 0.71 ms
   per VRAM layer, 10×**. The levers: a per-*expert* VRAM cache instead of per-layer
   all-or-nothing (so hot experts of the overflow layers are resident too), heat-tiered
   quant to shrink the 57 GB, and pillar 4's split-bandwidth dispatch (run a layer's
   CPU and VRAM experts concurrently).
2. **Attention scales with context, not batch.** Its share is ~9-11% at short context
   but **41% at 2k** — and it is the one part of `forward_batch` still per-sequence.
   Quantized KV would cut its bandwidth; a flash-attention-style tiled kernel would
   cut its passes.

> **A note on these numbers.** An earlier revision of this table reported 54.5% for
> CPU experts and 35% for attention at 2k context. That column was measured while a
> compile and two oracle runs shared the machine, which inflated it — the numbers
> above replace it. Two published claims about `LAG_IDOT` were also wrong for related
> reasons (a cross-build comparison, and comparing a B=8 run's eight prefills against
> a B=1 run's one). The benchmarking rules in `CLAUDE.md` exist because of these.

### `LAG_IDOT` after grouping: no longer a win

Grouping changed which int4 expert kernel is better, because the f32 path dequantizes
each weight row **once per group** and reuses it across the group's rows, while
`MOE_Q4_IDOT` skips dequant but pays a horizontal reduction per 32-element block.
`make kernel-check` measures it directly (GMAC/s, f32 / idot):

| shape | S=1 | S=2 | S=4 | S=8 | S=16 |
|---|---|---|---|---|---|
| gate_up (D→2I) | 77.8 / **90.3** | 157.7 / **170.8** | 181.7 / 187.2 | **196.1** / 153.0 | **206.7** / 196.6 |
| down (I→D) | **205.1** / 146.8 | 163.2 / **177.5** | **215.4** / 209.9 | **243.4** / 181.8 | **278.3** / 214.9 |

IDOT keeps its ~+16% only for `gate_up` at one or two rows — the regime it was
measured in originally — and loses from four rows up and on `down` almost everywhere.
End to end the differences cancel: B=32 short measured 43.56 tok/s with `LAG_IDOT=1`
vs 43.19 without, i.e. noise. It stays **off by default**: no speed to buy, and it
costs ~0.4% quant noise. The same table quantifies the batched kernel itself —
`gate_up` f32 goes 77.8 → 206.7 GMAC/s (2.7×) purely from having rows to amortize
dequant over.

### Attention: SIMD for generation, and threads over heads

Attention computed QK in **scalar double** (laguna's oracle contract) and
parallelized over streams only — at B=8 that fed 8 threads on a 24-core box.
`MOE_QK_SIMD` (AVX-512 FMA dot + vectorized value accumulation) now runs for
generation while `g_exact` keeps the double path, and decode parallelizes over
(stream, head). Measured within one build:

| | attention ms/step | aggregate |
|---|---|---|
| B=8 @2k, scalar double + per-stream threads | 238.8 | 17.6 tok/s |
| B=8 @2k, SIMD + per-head threads | **166** (−30%) | **19.9** (+13%) |
| B=1 short, scalar double | 5.68 | 20.8 tok/s |
| B=1 short, SIMD + per-head | **2.47** (2.3×) | 21.9 |
| B=32 short (attention is only ~11% there) | 86.5 → 83.4 | 43.4 → 43.2 (noise) |

Long-prompt prefill also improves (583 → 464 s for 8 × 2139 tokens, −20%). The win is
where the profile said it would be — long context — and invisible at short context,
which is why the table above matters more than the change itself.
`LAG_ATTN_EXACT=1` restores the double path. The oracle still validates that exact
path, so the SIMD path is checked in `make kernel-check` against it (windows of
1/17/512/2187, agreeing to ~1e-7).

## KV cache: sliding layers on a ring

36 of laguna's 48 layers attend only to the last 512 positions, but the cache used to
store the full context for every layer — and KV, not compute, is what caps how many
agents share a box. Sliding layers now use a ring of `next_pow2(window + span)` slots:

| per slot @ 8k ctx | KV |
|---|---|
| full-length (`LAG_KVFULL=1`) | 3.22 GB |
| sliding layers on a 1024-slot ring | **1.11 GB** (2.9×) |

The ring must hold the window *plus* one forward pass's span, because a pass writes
every position's K/V before it attends — a ring of exactly `window` lets the pass's
early rows overwrite the history those same rows need. The tiny oracle caught that
instantly (36/36 → 4/36), which is why `forward()` now runs in `kv_span()`-sized
passes; that chunking is bit-identical to one pass.

Validated: oracle 36/36 + ppl 0.00% with the ring, including `LAG_KV_SPAN=4`, which
shrinks the ring to 16 slots so a 36-token prefill **wraps it twice**; and on the real
118B, a 2139-token prompt (wrapping the 1024-slot ring) generates identical text with
and without trimming.

## Prefix caching for agent loops (`LAG_KV_REUSE=1`)

Decode throughput is not what limits an agent loop — **re-prefill** is. An agent
re-sends its whole transcript every turn, so at ~29 tok/s prefill a 20k-token context
costs roughly **12 minutes per turn** against ~12 s of decode for 200 output tokens.
Essentially the entire turn is spent recomputing a context that barely changed.

Two mechanisms, both **on by default** (`LAG_NOREUSE=1` disables):

1. **Per-slot reuse.** Each slot's KV cache is now persistent and carries the token ids
   it represents. On admission the engine takes the longest common prefix with the new
   prompt and prefills only the tail. Zero copying — the cache is already right.
2. **Shared prefix snapshots.** Per-slot reuse only helps if a request returns to *its*
   slot. An agent fleet shares a preamble (system prompt, tool definitions, repo
   context) across many conversations, so snapshots of that preamble are pooled and any
   slot can restore one with a `memcpy`. Lengths round down to a 256-token granule so
   requests sharing a preamble hit the same snapshot.

This is what the spare RAM is for. The weights are already fully resident (57 GB of
187), so there is nothing left to promote to RAM; **KV is the thing worth hoarding**, and
at int8/8k a cache is 0.29 GB, so ~100 GB of headroom holds hundreds of contexts.

### Pin `cache_slot` per conversation

The gateway assigns `min(free_slots)` when a request does not name one, so a
conversation's turns can land on **different** slots and per-slot reuse never fires
(the shared pool still catches the preamble). An agent harness should send a stable
slot per conversation:

```json
{"model": "laguna-s-2.1-colibri", "messages": [...], "cache_slot": 3}
```

### The sliding-ring constraint on rewinding

Reuse is refused when the divergence point is more than `ring - window` (~512) tokens
back, because the sliding layers' ring no longer holds that history — reusing it would
silently attend to K/V that later positions overwrote. Normal agent turns diverge at the
end of the transcript and always qualify; a client that *edits* earlier history falls
back to a full prefill. Refusals are counted in the `[kv-reuse]` line on exit rather
than being invisible.

## Status of the unvalidated pieces

Two things in this document are implemented but **not** yet backed by a measurement, and
are off by default for that reason:

- **int8 KV (`KV8=1`) — measured, modest damage, stays opt-in.**

  | on text the model is confident about (ppl 1.91) | prediction agreement |
  |---|---|
  | f32 vs itself | 1024/1024 |
  | benign change (exact double QK vs SIMD QK) | 1024/1024 |
  | **int8 KV, 32-element block scales** | **958/1024 — 66 flips (6.4%)** |

  The benign control is what makes this readable: pure float-rounding differences flip
  *nothing* on confident text, so all 66 flips are attributable to int8. Perplexity is
  blind to it — it moved 1.91 → 1.81, i.e. "improved" — which is why the gate scores
  prediction agreement instead.

  Two earlier claims here were wrong and are worth recording. First, an initial gate on
  hard out-of-distribution prose reported 706/1024 and I called int8 broken; on that text
  a *benign* change scores 931/1024, because the model is near-tied everywhere and 1024
  was never reachable. Judge a perturbation against a benign control, not against 100%.
  Second, I attributed the damage to outlier channels in V (unnormalized, unlike K which
  passes qk-RMSNorm) and predicted 32-element block scales would largely fix it: they
  moved the hard-text score 706 → 733. The blocks are still the right call on principle
  (1.125 B/value vs 1.03, and kernel-check carries an outlier stress case), but they were
  not the explanation.

  So: 3.8× less KV memory (1.11 → 0.29 GB/slot at 8k) for ~6% of next-token predictions
  changing. Worth it when memory is the binding constraint, not a free win, and off by
  default.

Prefix reuse is no longer in that category — it is measured and on by default:

| 3-turn agent conversation (~2.3k-token transcript, greedy) | turn 0 | turn 1 | turn 2 |
|---|---|---|---|
| no reuse — TTFT | 58.6 s | 60.7 s | 61.1 s |
| with reuse — TTFT | 59.1 s | **0.66 s** | **0.63 s** |

Byte-identical output across all three turns, so the 92× on continuation turns costs
nothing in fidelity. Getting there required fixing a bug worth recording: the slot's
token history was appended on token *emit*, but a token only enters the KV cache when it
is later *fed* to a forward pass, and the last token of a turn never is. History sat
permanently one token ahead of the cache, so the next turn's prefix match claimed a
position whose K/V was never computed — stale buffer bytes read as context. No crash,
coherent output, different wording. The invariant is now explicit: **`hist_len ==
kv.len`**, maintained in the prefill step and after `forward_batch`, never on emit.

## Where prefill time goes

Prefill is 70-85% of an agent turn, so it got its own profile (`LAG_PROF=1` now reports a
prefill breakdown, not just decode). 2139-token prompt, one stream, 99% of wall accounted:

| phase | share | ms per 128-token chunk |
|---|---|---|
| **experts CPU** (10 of 47 layers) | **27.7%** | 935 |
| **qkv+rope projections** | **22.0%** | 742 |
| gate + o_proj | 15.2% | 514 |
| attention | 13.5% | 456 |
| experts VRAM (37 layers) | 10.7% | 360 |
| routing + group sort | 6.3% | 214 |
| shared expert | 3.3% | 110 |

There is no single lever here, and the obvious suspects are not it:

- **Chunk size does nothing.** 128 vs 512 rows: 57.2s vs 57.6s. (An earlier sweep that
  showed 83.8/76.2/72.1s was measured under contention and is retired.)
- **Attention is only 13.5%**, so GQA-grouped attention — which cuts that phase ~20% —
  moves total prefill ~2%.
- **Caching is already maxed.** A continuation turn reuses the whole prior transcript;
  only the new delta prefills, which is why turn 2 of an agent conversation costs *less*
  wall time than turn 1 despite a larger context.

The cost is arithmetic spread across everything, and the *resident* matmuls (qkv/rope +
gate/o_proj = 37%) outweigh the experts. Those already run on the GPU, so they are limited
by our kernels rather than by placement — the same MMQ-style tensor-core ceiling described
in `llamacpp-notes.md`, and the same ceiling for batched decode. Clean prefill is currently
**37.4 tok/s** (2139 tokens in 57.3s).

## GQA-grouped attention

Laguna has 8 KV heads serving 48-72 query heads, so a per-head loop reads each KV head's K
and V 6-9 times per layer. `sdpa_group` attends a whole query group in ONE pass over the
window, using the online-softmax form (running max/sum, no score buffer) — the same form
that measured a wash for a single query row, because with 6-9 rows sharing the stream there
is finally something to save.

Measured on prefill (interleaved A/B/A/B, one engine): attention phase 8.9 → 6.8s and
8.2 → 7.9s; total prefill 58.2 → 55.9s and 57.7 → 57.1s. Validated against the per-head
double-accumulate reference at L2rel 1.3e-6 in `make kernel-check`. Parallelism drops from
B·heads to B·kv_heads work items, so it engages only when that still fills the cores
(≥24 items) — `LAG_ATTN_GROUPED` forces either way.
