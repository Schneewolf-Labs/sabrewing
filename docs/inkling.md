# Inkling (Thinking Machines MoE) on colibri

`c/inkling.c` runs both sizes of [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling)
(Apache 2.0) with colibri's expert-streaming approach: dense weights resident
(RAM or VRAM), routed experts streamed from disk with an LRU + pinned cache.
Text-only; vision/audio encoders are not loaded. The MTP draft head loads on
demand for speculative decode (`MTP=1`; see below).

| Model | Total / active | int4 container | Notes |
|---|---|---|---|
| [Inkling](https://huggingface.co/thinkingmachines/Inkling) | 975B / 41B | ~469 GiB | NVMe-streamed; the capacity vehicle |
| [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) | ~266B / ~10B | **131 GiB** | fits in RAM on a 187 GB box |

**Inkling-Small needs no engine changes.** It is a straight scale-down — same MoE
structure (256 routed experts top-6, 2 shared, 201k vocab), only dimensions differ
(hidden 4096 vs 6144, 42 layers vs 66, 32 heads vs 64, expert intermediate 2048 vs
3072, `logits_mup_width_multiplier` 16 vs 24). Both the engine and the converter
read every dimension from `config.json`; the `6144` literals in `inkling.c` are in
the CUDA self-test, not the model path.

## Quickstart

Pre-converted weights (int4 experts + bf16 residents):

```sh
# 975B — ~469 GiB
hf download nbeerbower/Inkling-colibri-int4 --local-dir ~/Models/inkling_i4
# Small — ~131 GiB
hf download sabrewing-engine/Inkling-Small-colibri-int4 --local-dir ~/Models/inkling_small_i4
```

or convert the original bf16 checkpoint yourself (shard-resumable; `--watch`
converts while the download is still running):

```sh
python3 c/tools/convert_inkling_int4.py --indir <bf16-checkpoint> --outdir ~/Models/inkling_i4
```

> `--watch` is a win when the source is on NVMe. When the source download and the
> converter output share one **spinning disk** it is a ~3x pessimization — the head
> thrashes between the download's writes and the converter's reads. Measured on an
> 8 TB WD: 197 MB/s with the disk to itself vs 60 MB/s overlapped. On HDD, let the
> download finish first, and always put the *output* snapshot on NVMe since the
> engine reads experts from it on every cache miss.

Build and run:

```sh
make -C c inkling ARCH=native    # pure CPU (dependency-free, like glm)
make -C c inkling CUDA=1 ARCH=native   # + bf16 residents in VRAM (needs ~37 GB free for 975B)

SNAP=~/Models/inkling_i4       ./c/inkling -p "The capital of France is" -n 64
SNAP=~/Models/inkling_small_i4 ./c/inkling -p "The capital of France is" -n 64
```

Requirements: 975B wants ~120 GB RAM (CPU build keeps ~86 GB of bf16 residents in
RAM; the CUDA build moves them to VRAM and uses the freed RAM for a larger expert
cache). Small's whole 131 GiB container fits in page cache on a 187 GB machine, so
a warmed run does no disk I/O at all. NVMe storage for the snapshot either way.

## Modes

| Invocation | What it does |
|---|---|
| `-p "text" [-n N]` | streaming greedy generation (stops at eos or N tokens) |
| `-f prompts.txt [-n N]` | one prompt per line (`#` comments skipped), single model load, state reset between prompts — the cache-warming workflow below |
| `[cap] [bits] [ref.json]` | token-exact oracle harness against a `tools/make_tiny_inkling.py` fixture (CI-style validation) |

## Cache warming (same idea as glm's `.coli_usage`)

Expert selections are counted per `(layer, expert)` and written to
`SNAP/.coli_usage` after each generation run. On startup the top `PIN_N`
experts per layer are pinned (non-evictable, loaded in one parallel burst).
Counts **accumulate across runs**, so the ranking converges toward your real
workload — pins trained on a single prompt overfit badly (see the benchmark
table), which is why a diverse warmup matters:

```sh
SNAP=~/Models/inkling_i4 ./c/inkling -f warmup_prompts.txt -n 32
```

| Env | Effect |
|---|---|
| `PIN=off` | disable warming entirely: no seeding, no pins, no history rewrite |
| `PIN=<path>` | alternate history file |
| `PIN_N=<n>` | pins per layer (default `cap/2`; 0 = seed ranking only) |
| `USAGE_SAVE=0` | don't rewrite the history (benchmark runs) |
| `NOGPU=1` / `GPU_DEV=<n>` | disable CUDA / select device |
| `IDOT=0` | byte-exact scalar int kernels (debugging) |
| `INK_CACHE_CHECK=1` | assert every routed pair still holds the expert it was acquired for (see below) |
| First positional arg | expert-cache cap per layer (`0` = auto-size from free RAM) |

### The cap does not have to hold the batch

`moe()` acquires, fills and computes routed `(token, expert)` pairs in **rounds**
bounded by the number of unpinned slots. This is a correctness requirement:
`slot_acquire` evicts the LRU unpinned slot and cannot tell that a slot is still
owed to a pair routed earlier in the same call. The engine used to acquire every
pair up front, so once a call touched more distinct experts than the cache had
unpinned slots, a late acquire recycled a slot still in use — the pointer stayed
valid, the expert behind it changed, and the layer computed with the wrong
weights. No crash, no warning, just degraded output.

Measured on `inkling_small_i4` (256 experts, top-6, 42 layers), 13-token prompt,
before the fix: `cap=8` served the wrong expert at the *first* MoE layer, and
`cap=32` (16 pinned, 16 evictable) did too. The tiny oracle cannot see this —
its expert count is small enough that the cap clamps to full residency and
nothing ever evicts.

With rounds, the cap is a pure memory/IO trade-off and can go below `topk`:

| Cap | Output (12 greedy tokens, same prompt) | RSS | Expert loads |
|---|---|---|---|
| auto (256, full residency) | `technology and technology's impact on society. I.name is Ink` | 36.8 GB | 2159 |
| 8 | identical | 15.2 GB | 3725 (42% churn) |
| 1 | identical (3-token run) | 11.9 GB | — |

Pairs are walked in (token, rank) order and each token's contributions still land
in ascending rank, so a round boundary is **bit-exact**, not merely equivalent —
and a cap that holds the whole call is a single round, i.e. the old code path.
`INK_CACHE_CHECK=1` asserts the invariant on every layer call; it is cheap enough
to leave on for a validation run and exits on the first violation.

## Performance (975B, Ryzen 9 7900 / 24t, 187 GB DDR5, RTX A6000, NVMe)

24-token greedy decode, 5-token prompt, commit-tagged runs, single run each.
"Trained" = usage history built from the same prompt; "novel" = never-seen prompt.

| Configuration | Prefill | Decode | Cache hit |
|---|---|---|---|
| Stage A (plain LRU, serial I/O, CPU) | 150.2 s | 0.06 tok/s | ~0% |
| + packed-int4 cache, parallel fills, pins (CPU) | 21.1 s | 0.25 tok/s | 81.5% |
| + CUDA resident tier (A6000) | 18.4 s | 0.32 tok/s | 83.6% |
| + deep pins, trained prompt (`PIN_N=64`) | 1.9 s | 2.51 tok/s | 100.0% |
| deep pins, novel prompt (overfit pins) | 33.8 s | 0.17 tok/s | 79.8% |
| **steady state: 11-prompt diverse history, default pins, novel prompt** ¹ | **35.4 s** | **0.25 tok/s** | **82.2%** |

¹ 48-token generation, 13-token prompt. With only a small warmup corpus the
ranking is barely ahead of plain LRU; hit rate (and therefore decode speed)
grows toward the trained-prompt number as real-use history accumulates.

Phase profile at high hit rates: ~90% CPU expert matmul — the next lever is
expert compute on the GPU, not more I/O work. The GPU int4 expert GEMM kernel
(`ink_cuda_matmul_q4`) is in place and validated (`--cuda-q4-test`); the VRAM
expert cache tier that feeds it is the remaining piece — see
`docs/gpu-experts-design.md`.

### Inkling-Small (preliminary, CPU build)

Same box, `ARCH=native`, 24-token greedy decode, snapshot on NVMe. Single runs —
these are first-light numbers, not a tuned benchmark.

| Configuration | Prefill | Decode | Cache hit |
|---|---|---|---|
| warm, pins trained on the prompt | 1.1 s | **3.0 tok/s** | 100.0% |
| warm, novel prompt | 15.5 s | 0.86 tok/s | 82.0% |

At 100% hit the run reports `fill 0.0s` — zero disk reads — so the trained-prompt
number is pure compute. That is the interesting difference from the 975B: Small's
whole container fits in RAM, so I/O stops being the constraint and expert matmul
becomes the ceiling. The CUDA resident tier is therefore the obvious next lever
here too, and is **not yet measured** for this model.

## Speculative decode (MTP)

The checkpoint ships an 8-module MTP draft head (`out-mtp.safetensors`). The
depth-0 drafter is wired and **lossless** (output identical to plain greedy) — on
the real base head it accepts its next-next-token guess ~38.5% of the time, ~1.4×
decode. It is **opt-in** (`MTP=1`) so a normal serve pays nothing for the 10.5 GB
head until asked. The multi-depth chain (toward 2–3×) and sampling-serve wiring
are pending — see `docs/mtp-design.md`. Diagnostics:

```sh
# per-depth acceptance on a token sequence (measures the speedup ceiling)
MTP=1 SNAP=~/Models/inkling_i4 ./inkling 0 0 --mtp-accept tokens.json
# token-exact vs the transformers reference on the tiny fixture
SNAP=tiny_inkling ./inkling 8 0 --mtp-oracle tiny_inkling/ref_mtp.json
```

## Chat serving & reasoning effort (OpenAI gateway)

`openai_server.py` renders Inkling's chat template and exposes an
OpenAI-compatible API (`--engine ./inkling --model <snapshot>`). Reasoning is
controlled per request by `reasoning_effort`, mapped to Inkling's thinking hint:

| `reasoning_effort` | thinking hint | behavior |
|---|---|---|
| unset / `none` | `0` | **thinking off** — the prompt is prefilled to the content channel (`<\|message_model\|><\|content_text\|>`) so the model answers directly |
| `minimal` / `low` | 0.1 / 0.2 | brief reasoning |
| `medium` | 0.7 | moderate reasoning |
| `high` / `max` | 0.9 / 0.99 | full reasoning |

The effort hint is only a *soft* signal. With it set to 0 but no prefill, the
model can still sample `<\|content_thinking\|>` as its first token, open a
reasoning block, and spend the entire `max_tokens` planning — never reaching
`<\|content_text\|>`, which the marker splitter then strips to an empty answer
(the failure is seed-dependent, so it looks intermittent). The content-channel
prefill forecloses that when thinking is off, and is exactly the token sequence
every non-thinking turn is trained on.

When thinking is on, the reasoning is surfaced as `reasoning_content` (streamed
as `reasoning_content` deltas, or in the final message's `reasoning_content`
field) instead of being mixed into — or silently dropped from — the answer.
Reasoning consumes the token budget, so pair `reasoning_effort: high` with a
generous `max_tokens` or the whole budget can go to thinking before any answer
is emitted.

## LoRA adapter serving (Tinker raw format)

Fine-tune on [Tinker](https://thinkingmachines.ai/tinker), serve the adapter
here — point the engine at the *raw* Tinker checkpoint directory
(`adapter_model.safetensors` + `adapter_config.json`; not the PEFT
conversion, which materializes Tinker's shared factors per-expert and is 3x
the size for no gain):

```sh
SNAP=~/models/inkling_i4 ./inkling -l /path/to/adapter -p "..." -n 128
python3 openai_server.py --engine ./inkling --model ~/models/inkling_i4 --lora /path/to/adapter
```

`LORA=<dir>` is the env equivalent of `-l`; `LORA_SCALE=<f>` multiplies the
adapter's `alpha/r`. One adapter per process.

Two application paths, split by weight class:

- **Residents** (attention, dense MLP, shared experts, `lm_head`) are merged
  into the resident weights at load — `W += (alpha/r)·B·A`, round-to-nearest-even
  into bf16, before any CUDA upload. Zero decode-time cost.
- **Routed experts** are *never* merged: a rank-32 delta is smaller than the
  int4 quantization step, so baking it in would destroy most of the adapter
  signal (and would mean requantizing the container per adapter). Instead the
  A/B tensors stay resident (~19 GB f32 for the 975B rank-32 adapter, loaded
  before the cache auto-cap sizes itself) and a low-rank correction wraps each
  expert's int4 matmul. Tinker's factor sharing (one `lora_A` per layer for
  gate/up, one `lora_B` per layer for down) keeps the overhead under 1% of
  expert compute, and the expert LRU cache and pinned set are untouched —
  hit rates and warmed histories carry over unchanged.

## Validation

Every mode is token-exact against HF transformers on a tiny random-init oracle
(`c/tools/make_tiny_inkling.py`): f32, int4-container (VNNI and `IDOT=0`
scalar), bf16 residents on CPU, and bf16 residents through the CUDA kernel.
The tokenizer (o200k family, auto-detected by `tok.h`) encodes 357/357 test
strings identically to HF `tokenizers`. The converter round-trips a fabricated
TML-layout checkpoint back through the engine exactly (`--selftest-e2e`).

The LoRA paths have their own fixture: `c/tools/make_tiny_lora.py` builds a
random adapter in the exact Tinker tensor structure (shared factors, fused
dense gate_up, fused shared experts) plus a merged-model reference, and

```sh
SNAP=tiny_inkling LORA=tiny_lora ./inkling 8 0 tiny_lora/ref_inkling_lora.json
```

reproduces it token-for-token (0.00% perplexity vs the merged transformers
reference, 24/24 continuation tokens), which pins down the name mapping, the
fused row order, the alpha/r scale, and both application paths at once. The
oracle needs an Inkling-capable transformers (`>= 5.4`) to build the reference.
