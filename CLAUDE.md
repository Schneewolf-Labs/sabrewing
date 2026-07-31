# sabrewing — working notes for Claude

A dependency-free C runtime for **MoE inference**: small readable engines over a
shared MoE core, validated bit-for-bit against `transformers`. A friendly fork of
[colibrì](https://github.com/Schneewolf-Labs/colibri) (see the README section on
that relationship — improvements to the shared substrate go upstream).

## Primary focus: Laguna

**Laguna-S-2.1 (118B total / ~8B active) is the flagship target.** It is the
*smallest* model this runtime supports while still having strong coding and agentic
ability — and its int4 container (~57 GB routed experts + ~8 GB residents) is the
only supported model that comes close to fitting a single 48 GB GPU. That
combination is what sabrewing exists to exploit: **long-running, high-intelligence
agents** need many tokens over many hours at high aggregate throughput, which is a
*throughput* problem, not a peak-single-stream problem. Laguna is where the MoE
serving levers (batched decode, group-by-expert GEMM, the VRAM/RAM pager) pay off
on hardware someone actually owns.

Practical consequences for work in this repo:
- Prefer Laguna (`c/laguna.c`, `docs/laguna.md`) when choosing where to land a
  runtime optimization; generalize into the shared core (`c/moe_*.h`) when a second
  engine can use it.
- Optimize **aggregate throughput and prefill**, not just single-stream tok/s.
- Inkling (975B, NVMe-streamed) remains the capacity/streaming research vehicle;
  GLM (`c/colibri.c`) and OLMoE are the other descriptor-driven engines.

## Layout

| Path | What |
|---|---|
| `c/laguna.c` | Laguna engine (primary) — see `docs/laguna.md` |
| `c/inkling.c` | Inkling 975B engine (NVMe expert streaming) — `docs/inkling.md` |
| `c/deepseek.c` | DeepSeek-V4-Flash engine (mHC + CSA/HCA compressed attention) — `docs/deepseek.md`. Stage A: f32 CPU, transformers oracle not yet run |
| `c/colibri.c` | GLM engine (vendored upstream substrate) |
| `c/olmoe.c` | OLMoE (small, fast arch-generalization check) |
| `c/moe_*.h` | shared MoE runtime: `moe_arch.h` descriptors + hooks, `moe_block.h` routing/combine, `moe_attn.h` SDPA, `moe_matmul.h` f32 GEMM, `moe_quant.h` int4/int8 kernels, `moe_sample.h`, `moe_serve.h` |
| `c/backend_cuda_laguna.{cu,h}` | Laguna CUDA tier (VRAM residents + expert cache + grouped expert GEMM) |
| `c/openai_server.py` | OpenAI + Anthropic gateway, web UI (`web/`) |
| `c/tools/` | converters (`convert_laguna_int4.py`), tiny-oracle builders (`make_tiny_*.py`), `kernel_check.c` |
| `docs/moe-runtime-plan.md` | the five-pillar architecture plan + phase sequencing |
| `docs/backend-todo.md` | live loose-ends/roadmap ledger — cross items off here |

## Build & run

```sh
cd c
make laguna CUDA=1 ARCH=native          # GPU tier; plain `make laguna` = CPU only
SNAP=~/Models/laguna_i4 ./laguna -f chat.txt -n 128
BATCH=16 SNAP=... ./laguna -f chat.txt -n 24            # throughput bench (identical streams)
BATCH=16 BATCH_VARY=1 SNAP=... ./laguna -f chat.txt -n 24   # divergent streams (honest number)
```

Laguna is an instruct/RL model — always use the chat template (`-f`, or the
gateway), never a raw base-completion prompt.

Serve mode: `KV_SLOTS=N` (the gateway's `--kv-slots N`) enables continuous batching
— N requests in flight, decoded in lockstep. `KV_SLOTS=1` is the old serial loop.

## Validation is the non-negotiable

Every arch validates **token-exact against `transformers`**; no optimization lands
that can't reproduce its oracle (quant paths validate within measured noise).

```sh
SNAP=/tmp/tlag  ./laguna /tmp/tlag/ref_laguna.json      # f32 tiny: 36/36 tf, 24/24 greedy, ppl 0.00%
SNAP=/tmp/tlag_i4 ./laguna /tmp/tlag_i4/ref_laguna.json # int4 tiny container
make kernel-check                                        # quant kernels vs double dequant reference
```

Those `/tmp/tlag*` snapshots were built by `tools/make_tiny_laguna.py`, which needs
`transformers >= 5.7` (the version installed here is older — rebuild the snapshots
on a box with a newer transformers, don't delete them casually).

Extra checks that matter for batching/grouping work:
- `BATCH=N` asserts every stream reproduces the single-stream greedy output.
- `BATCH_AB=1` times per-token vs grouped MoE **in one process** (KV rewound
  between passes) and asserts the two passes are token-identical.

**Benchmarking rules — violating these has produced wrong numbers here more than
once, and a wrong number is worse than no number:**
1. **Nothing else runs during a bench.** No `make`, no oracle run, no second engine.
   A compile alongside a 24-core bench inflated one config by ~4× and made a change
   look good that wasn't; a still-exiting engine holding 47 GB of VRAM crippled the
   next run's expert cache.
2. **Never compare across builds.** Timings on this model swing ~2× with page-cache
   and allocation state. Put both arms of every A/B in ONE pass over ONE binary
   (`BATCH_AB=1`, `LAG_ATTN_EXACT=1`, `LAG_NOGROUP=1`, `LAG_KVFULL=1` all exist for
   this).
3. **Check what a number is per.** A B=8 run prefills eight prompts; a B=1 run
   prefills one. Comparing their prefill totals produced a fictitious "7.8× slower"
   claim about `LAG_IDOT`.
4. **Don't filter a test's stderr into oblivion.** `grep ERROR` swallowed a Python
   traceback and left an empty log that read as a successful run.
5. **`pkill -f <pattern>` matches your own shell** if the pattern appears in its
   command line. Twice this killed the harness instead of the target.
- `LAG_NOGROUP=1` runs the per-token MoE path — an A/B against the grouped path
  must be **token-identical** (grouping is bit-exact by construction: per-row
  reductions keep their order and the combine sums in top-k rank order).
- `LAG_CUDA_TEST=1 ./laguna` — GPU kernels vs CPU, plus grouped-vs-per-token
  experts (expects max|abs| = 0). No model needed.

## Conventions

- **Readable and dependency-free.** Borrow ideas from llama.cpp/ggml (group scales,
  SIMD layouts, mmap, online softmax); never vendor the code. The core stays
  something you can read end-to-end.
- Measure, don't vibe. Perf claims in docs/commits carry the numbers and the
  hardware; ruled-out ideas get recorded as ruled out (see the prefetch entry in
  `docs/backend-todo.md`).
- Beware benches that flatter the change: `BATCH=N` alone decodes *identical*
  streams, so expert grouping collapses to top-k and overstates the win — report
  `BATCH_VARY=1` too.
- Keep engine-specific weight layout behind the `moe_arch.h` hooks; shared code
  stays layout-agnostic.
- Hardware here: RTX A6000 48 GB, 24-core AVX-512 CPU, 187 GB RAM; models on
  `/home/nbeerbower/Models`, `/mnt/AZURA`, `/mnt/COVENANT`.
