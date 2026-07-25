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
| `LAG_IDOT=1` | CPU experts via int8-VNNI (~+19% on the CPU tier; ~0.4% quant noise) |
| `BATCH=N` | batched-decode throughput bench (N copies of the prompt) |
| `BATCH_VARY=1` | bench with *divergent* streams (sampled, per-stream seed) — the honest multi-tenant number |
| `BATCH_AB=1` | bench both MoE paths in one process (+ asserts they are token-identical) |
| `LAG_NOGROUP=1` | disable group-by-expert batching (A/B knob; output is identical either way) |
| `LAG_GROUP_CHUNK=N` | rows per grouping chunk, default 128 (prefill is chunked; bounds scratch) |
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
   speeds up **prefill**, where a chunk of 64 prompt tokens groups the same way.

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

**v1 limitations, both measured:**
- Admission is blocking — a newly arriving request prefills before the next decode
  step, so a long agentic prompt pauses the other slots for the duration of its
  prefill. With 8 requests × 48 tokens, prefill is ~37% of the wall. Chunked prefill
  interleaved with decode is the next step.
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
