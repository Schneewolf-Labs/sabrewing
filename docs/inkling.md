# Inkling (Thinking Machines 975B MoE) on colibri

`c/inkling.c` runs [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling)
(975B total / 41B active, Apache 2.0) with colibri's expert-streaming approach:
dense weights resident (RAM or VRAM), routed experts streamed from disk with an
LRU + pinned cache. Text-only; vision/audio encoders and the MTP head are not loaded.

## Quickstart

Pre-converted weights (int4 experts + bf16 residents, ~469 GiB):

```sh
hf download nbeerbower/Inkling-colibri-int4 --local-dir ~/Models/inkling_i4
```

or convert the original bf16 checkpoint yourself (shard-resumable; `--watch`
converts while the download is still running):

```sh
python3 c/tools/convert_inkling_int4.py --indir <bf16-checkpoint> --outdir ~/Models/inkling_i4
```

Build and run:

```sh
make -C c inkling                # pure CPU (dependency-free, like glm)
make -C c inkling CUDA=1         # + bf16 residents in VRAM (needs ~37 GB free)

SNAP=~/Models/inkling_i4 ./c/inkling -p "The capital of France is" -n 64
```

Requirements: ~120 GB RAM (CPU build keeps ~86 GB of bf16 residents in RAM;
the CUDA build moves them to VRAM and uses the freed RAM for a larger expert
cache), NVMe storage for the snapshot.

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
| First positional arg | expert-cache cap per layer (`0` = auto-size from free RAM) |

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
expert compute on the GPU, not more I/O work.

## Validation

Every mode is token-exact against HF transformers on a tiny random-init oracle
(`c/tools/make_tiny_inkling.py`): f32, int4-container (VNNI and `IDOT=0`
scalar), bf16 residents on CPU, and bf16 residents through the CUDA kernel.
The tokenizer (o200k family, auto-detected by `tok.h`) encodes 357/357 test
strings identically to HF `tokenizers`. The converter round-trips a fabricated
TML-layout checkpoint back through the engine exactly (`--selftest-e2e`).
