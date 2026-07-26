# Qwen3.5-35B-A3B on sabrewing vs Qwen3.6-27B dense on llama.cpp

Measured to answer one question: which backend should drive an agent loop today?

Method — `c/tools/bench_endpoints.py`, which reports it honestly or not at all:

- **One engine at a time.** Both want most of the 48 GB card; VRAM was verified back to 30 MiB
  between runs, and load average was under 0.8 before each. A resident second engine has
  produced wrong numbers in this repo before.
- **Identical harness, identical prompts.** Matching `prompt_tokens` (565 / 2275 / 4555) on both
  sides confirms the prompts were the same, not just nominally similar.
- **Thinking disabled on both** (`enable_thinking` for sabrewing's gateway,
  `chat_template_kwargs` for llama.cpp). Reasoning-token volume differs wildly between models
  and would otherwise dominate the decode comparison.
- **Matched serving config**: 4 slots × 16384 context each.
- Prompts carry a unique nonce so a warm server cannot serve them from its prefix cache —
  except the explicit warm case, which repeats a prompt on purpose to price that cache.

Hardware: RTX A6000 48 GB, Ryzen 9 7900, 187 GB RAM.

| | sabrewing · Qwen3.5-35B-A3B int4 | llama.cpp · Qwen3.6-27B-TIES Q6_K | gap |
|---|---|---|---|
| prefill, 565 tok | 5.24 s | 0.69 s | 7.6× |
| prefill, 2275 tok | 22.63 s | 2.09 s | 10.8× |
| prefill, 4555 tok | 47.64 s | 4.39 s | 10.9× |
| **prefill throughput** | **96 tok/s** | **1088 tok/s** | **11.3×** |
| **decode** | **24.3 tok/s** | **26.5 tok/s** | **1.09× — tied** |
| warm repeat (prefix cache) | 1.0× (22.6 s) | **32.5×** (0.07 s) | — |
| 4 concurrent, aggregate output | 2.70 tok/s | 14.75 tok/s | 5.5× |
| VRAM | **19.4 GB** | 25.6 GB | sabrewing lighter |

The sabrewing column is after the conv/routing fixes below; it read 81 tok/s prefill
(13.4× gap) before them.

llama.cpp's own `timings.prompt_per_second` (1087–1093) agrees with the TTFT-derived 1038, so
the prefill figures come from two independent sources.

## What this says

**Decode is a dead heat.** A 35B MoE at int4 on a dependency-free C engine matches a 27B dense
at Q6_K on llama.cpp, per token per second, in 6 GB less VRAM. That is the MoE serving thesis
working: ~3 B active parameters is genuinely cheaper per token than 27 B dense, and it survives
being implemented from scratch.

**Every other column is prefill**, and the two deficits are different in kind:

1. **Raw prefill, 11.3×** — engineering, no open design questions. The remaining cost is
   concentrated in one place: see the profile below.
2. **Prefix caching, 32.5× vs 1.0×** — architectural. A recurrent state cannot be rewound, so
   reuse is limited to pure extension, and an OpenAI chat turn re-renders the assistant message
   differently from how it was generated, which breaks the prefix. Needs periodic
   recurrent-state checkpoints (see `linear-attention.md`).

Concurrency (6.5×) is prefill again, not a batching defect: 4 × 2275 tokens at 81 tok/s is
~112 s of prefill before much output exists.

## Where sabrewing's prefill actually goes

Profiled at 3321 tokens (`QW_PROF=1`), and the answer was not the MoE. Before any fixes, the
**gated-delta-net plumbing was 56% of prefill while the experts — 32 B of the 35 B parameters —
were 7.7%**. Two of the three biggest items were plain defects:

| | before | after | |
|---|---|---|---|
| linattn conv (4-tap depthwise) | 7.29 s | **2.34 s** | 3.1× |
| router GEMM | 3.65 s | **0.73 s** | 5.0× |
| routing top-k + sort | ~0.93 s | 0.73 s | |
| **prefill total** | **41.79 s (79.5 tok/s)** | **34.00 s (97.7 tok/s)** | **+23%** |

- **The conv was cache-hostile.** A 4-tap depthwise conv over 8192 channels is ~3.3 GMAC for
  the whole prefill, yet took 7.29 s — 0.45 GMAC/s. The loop was channel-outer over
  token-major data, so walking `t` for one channel strode 32 KB and every access pulled a
  64-byte line for 4 useful bytes. Token-outer with a small ring of input rows (needed because
  the output may alias the input) fixed it, bit-identically — the span-invariance assertion in
  `linattn_check.c` proved that for free.
- **The router GEMM was single-threaded**, and this was costing *every* MoE engine here. The
  GEMM kernels are shaped `for o in O { for s in S }`, so the work is `S*O*I`, but the OpenMP
  guard tested `O >= 512` alone. A 256-expert router is 256 wide, so it never parallelized no
  matter how many rows it was given. Now gated on `S*O` (`MOE_MM_PAR` in `moe_math.h`).
  Numerically free: each output is an independent reduction, so results are bit-identical.
- Routing top-k was O(K²E) with a `malloc` per row; a used-mask makes it O(KE).

After those, prefill is dominated by one item:

| phase | s | % |
|---|---|---|
| **linattn proj (GPU GEMM)** | **11.00** | **32.3** |
| linattn norm+out | 4.11 | 12.1 |
| attention (SDPA) | 3.30 | 9.7 |
| experts (GPU) | 3.20 | 9.4 |
| qkv+rope proj | 2.52 | 7.4 |
| linattn conv | 2.34 | 6.9 |
| shared expert | 1.94 | 5.7 |
| linattn recurrence | 1.17 | 3.4 |
| gate+o_proj, router, top-k | 2.40 | 7.1 |

`linattn proj` is ~6.7 TFLOP in 11.0 s = **609 GFLOP/s**, about **1.6% of the A6000's fp32
peak** and ~1% of its bf16 tensor-core peak. It is a hand-written fp32 FMA kernel with no
tensor cores and no real tiling, called once per GEMM with a host round-trip and a full stream
sync. llama.cpp's MMQ path achieves ~59 TFLOP/s effective. **That single kernel is now most of
the remaining 11× gap**, and unlike the two fixes above it is real work, not a defect.

## Recommendation

**Use llama.cpp for agent work today.** Agent turns are prefill-dominated — a measured egirl
turn spent 125 of 129 seconds in prefill — which is exactly sabrewing's weak axis and exactly
llama.cpp's strong one. On the same egirl task: 50.3 s (llama.cpp) vs 129 s (sabrewing).

Use sabrewing/qwen35 as the runtime development target. Closing the prefill gap flips the
recommendation, and the two items above are the whole list.

**Quality is not measured here.** These are different models and this file contains no
capability benchmark. If Qwen3.5-35B-A3B is materially better at coding or agentic work, a
prefill deficit may be worth paying on long-lived warm sessions — but nothing here supports or
refutes that.
