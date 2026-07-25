# sabrewing

**Large open MoE models on hardware you own.** A pure-C, dependency-free
inference engine for mixture-of-experts LLMs — from a 118B coding model that fits
one GPU to a 975B model streamed from NVMe — with an OpenAI- and
Anthropic-compatible API and a browser chat UI in front. No Python in the
inference path, no frameworks, no cluster. Validated **token-exact** against the
HuggingFace `transformers` reference.

*A sabrewing (**Campylopterus**) is one of the largest hummingbirds. This project
is a fork of [JustVugg/colibrì](https://github.com/JustVugg/colibri) — colibrì
being Italian for hummingbird — grown to carry heavier models. It was built on a
machine named `sabre`, and the name was sitting right there.*

## Supported models

Each model is a small standalone engine over a shared MoE runtime (see
[Architecture](#the-moe-native-runtime)). All are validated token-exact against a
tiny-model `transformers` oracle.

| Engine | Model | Size (total / active) | Notes |
|---|---|---|---|
| `laguna` | [Poolside Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) | 118B / 8B | Agentic coding model; **fits one 48 GB GPU**. YaRN + sliding attention, softplus gate, 256 experts top-10 + shared |
| `inkling` | [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling) | 975B / 41B | Streamed from NVMe on a workstation. Learned rel-bias, short-convs, MTP |
| `colibri` | GLM-5.2 (and the upstream arch family) | — | The multi-arch engine sabrewing forks from; MLA + DSA indexer |
| `olmoe` | [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | 7B / 1B | Softmax router, whole-vector QK-norm |

Pre-converted int4 weights on the Hub:
[`nbeerbower/Laguna-S-2.1-colibri-int4`](https://huggingface.co/nbeerbower/Laguna-S-2.1-colibri-int4),
[`nbeerbower/Inkling-colibri-int4`](https://huggingface.co/nbeerbower/Inkling-colibri-int4).

## What it does

- **Runs models that don't fit** — int4 experts + bf16/int8 residents, with a
  hierarchy that keeps hot weights close: an expert VRAM cache on GPU, a RAM cache,
  an NVMe stream for the truly enormous. Laguna's 118B runs entirely on one A6000;
  Inkling's 975B streams on a 187 GB workstation.
- **Token-exact numerics** — every engine validates bit-for-bit against
  `transformers` via a tiny-model oracle; the quantized kernels validate against a
  double-precision reference (`make kernel-check`). No "fast but subtly wrong."
- **A6000 GPU tier** — bf16/int8 residents in VRAM, a per-layer int4 expert cache,
  batched GPU expert kernels.
- **Batched serving** — decode many requests together so the resident weights are
  read once for the whole batch *and* each distinct expert once per step
  (group-by-expert GEMM): **47 tok/s aggregate at batch 32 vs 16 single-stream**,
  every stream bit-identical to single-stream. `--kv-slots N` puts it behind the API
  as continuous batching (+76% aggregate, 14× better tail latency on 8 concurrent
  requests).
- **OpenAI + Anthropic APIs** (`/v1/chat/completions`, `/v1/messages`, streaming,
  temperature/top-p) with each model's chat template, plus a browser chat UI.

## Quickstart

```sh
git clone git@github.com:Schneewolf-Labs/sabrewing.git && cd sabrewing/c

# --- Laguna 118B: fits a single 48 GB GPU ---
hf download nbeerbower/Laguna-S-2.1-colibri-int4 --local-dir ~/models/laguna_i4
make laguna CUDA=1 ARCH=native            # GPU tier (or `make laguna` for CPU)

# one-shot generation
SNAP=~/models/laguna_i4 ./laguna -p "def is_prime(n):" -n 128

# OpenAI + Anthropic API + web chat UI on http://127.0.0.1:8000
# --kv-slots 8 = continuous batching: 8 requests in flight, decoded in lockstep
python3 openai_server.py --arch laguna --engine ./laguna --model ~/models/laguna_i4 --kv-slots 8
```

Speed levers (all opt-in; the oracle stays exact): `RES8=1` int8 residents in VRAM
(fits more experts), `LAG_IDOT=1` int8-VNNI CPU experts, `BATCH=N` batched-decode
throughput bench. Build `ARCH=native` for AVX-512/VNNI; `CUDA=1` for the GPU tier.
See [`docs/laguna.md`](docs/laguna.md) for the container format, GPU-tier knobs,
and batched serving; [`docs/inkling.md`](docs/inkling.md) for the 975B streaming
setup.

## The MoE-native runtime

The engines are not four copies of the same code. They share one substrate and
diverge only where the architectures genuinely differ:

- **Shared kernels & core** (`c/moe_*.h`): the f32/bf16/int4/int8 matmuls, RMSNorm,
  softmax, sampling, the serve protocol, scaled-dot-product attention
  (`sdpa_head`), and a descriptor-driven MoE block (`moe_block` — route → top-k →
  combine). Contract differences (activation precision, RoPE vs learned bias,
  KV-cache layout, shared-expert policy) are *parameters*, not forks.
- **Per-engine forward** wires those primitives together for each architecture's
  quirks (Laguna's softplus gate, Inkling's short-convs, GLM's MLA/DSA indexer).
- **Oracle-gated refactor**: the shared substrate was extracted step by step with
  every engine's token-exact oracle green at each commit — the dedup cost zero
  numeric drift.

Design docs: [`docs/moe-runtime-plan.md`](docs/moe-runtime-plan.md) (the five-pillar
plan) and [`docs/moe-arch-survey.md`](docs/moe-arch-survey.md) (how the four
architectures map to descriptors).

## Performance

Ryzen 9 7900 · 187 GB DDR5 · RTX A6000 · NVMe.

**Laguna 118B (fits the A6000).** Single-stream decode is bandwidth/latency-bound
at ~12–16 tok/s; the real win is throughput. Two levers, both bit-exact: batched
decode amortizes the *resident* read across streams, and group-by-expert (Megablocks
-style) MoE reads each *distinct* expert once per step no matter how many streams
routed to it — which is what carries past the batch-16 wall:

| Batch | resident-batched | + group-by-expert | divergent streams |
|---|---|---|---|
| 1 | 15.7 | 16.4 | — |
| 8 | 30.5 | **40.9** | 36.0 |
| 16 | 34.0 | **43.2** | 38.4 |
| 32 | 30.3 | **47.2** | 44.3 |

Resident-only batching saturates ~30–34 tok/s (B=32 is no better than B=16 — every
stream still paid for its own expert reads); grouping keeps the curve climbing.

Grouping also speeds prefill (a chunk of prompt tokens groups the same way).
Divergent-stream numbers are reported alongside identical-stream ones because
identical streams route identically and flatter the grouping — see
[`docs/laguna.md`](docs/laguna.md).

**Inkling 975B (streamed).** Decode speed is a function of cache warmth, and the
cache learns your workload — from ~0.06 tok/s cold to ~2.5 tok/s once your hot
experts are pinned. Honest phase-profiling: at high hit rates the bottleneck is
CPU expert matmul.

## `swing` CLI

```sh
swing run "prompt" [-n N]     # one-shot generation
swing api [args...]           # OpenAI/Anthropic API + web UI (port 8000)
swing build [cuda]            # build the engine
swing info                    # model / cache / hardware status
```

## Roadmap

- Online-softmax attention + int8 KV (attention is 41% of a 2k-context step; would
  also take KV from 1.11 GB/slot to ~0.28) — see [`docs/llamacpp-notes.md`](docs/llamacpp-notes.md)
- Per-*expert* VRAM cache instead of per-layer all-or-nothing (the capacity gap is
  ~⅓ of a short-context step, 10× per layer)
- Tensor-core int8 expert GEMM (our warp-per-row kernel is the last big GPU ceiling)
- Unified VRAM↔RAM↔NVMe pager and cross-layer routing lookahead (the five-pillar plan)
- Heat-tiered quantization (measured, not vibed) for capacity
- More of the hummingbird catalog as open MoE models land

## Relationship to colibrì

sabrewing is a friendly fork, not a rival. The GLM engine (`c/colibri.c`) that
colibrì is built around ships here and works; the substrate this fork stands on —
expert streaming, the int4 container, the oracle-validation culture — is colibrì's
design, and improvements that belong upstream go upstream. Licensed
[Apache 2.0](LICENSE), same as upstream.

---

A [Schneewolf Labs](https://huggingface.co/schneewolflabs) project.
