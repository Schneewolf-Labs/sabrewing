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
| prefill, 565 tok | 6.27 s | 0.69 s | 9.1× |
| prefill, 2275 tok | 27.03 s | 2.09 s | 12.9× |
| prefill, 4555 tok | 56.18 s | 4.39 s | 12.8× |
| **prefill throughput** | **81 tok/s** | **1088 tok/s** | **13.4×** |
| **decode** | **25.5 tok/s** | **26.5 tok/s** | **1.04× — tied** |
| warm repeat (prefix cache) | 1.0× (27.0 s) | **32.5×** (0.07 s) | — |
| 4 concurrent, aggregate output | 2.26 tok/s | 14.75 tok/s | 6.5× |
| VRAM | **19.4 GB** | 25.6 GB | sabrewing lighter |

llama.cpp's own `timings.prompt_per_second` (1087–1093) agrees with the TTFT-derived 1038, so
the prefill figures come from two independent sources.

## What this says

**Decode is a dead heat.** A 35B MoE at int4 on a dependency-free C engine matches a 27B dense
at Q6_K on llama.cpp, per token per second, in 6 GB less VRAM. That is the MoE serving thesis
working: ~3 B active parameters is genuinely cheaper per token than 27 B dense, and it survives
being implemented from scratch.

**Every other column is prefill**, and the two deficits are different in kind:

1. **Raw prefill, 13.4×** — engineering, no open design questions. Activations never stay on
   device, so every GEMM is a host→device→kernel→device→host round-trip with a full stream
   sync; routing (10.8% of prefill) and the recurrence still run on the CPU.
2. **Prefix caching, 32.5× vs 1.0×** — architectural. A recurrent state cannot be rewound, so
   reuse is limited to pure extension, and an OpenAI chat turn re-renders the assistant message
   differently from how it was generated, which breaks the prefix. Needs periodic
   recurrent-state checkpoints (see `linear-attention.md`).

Concurrency (6.5×) is prefill again, not a batching defect: 4 × 2275 tokens at 81 tok/s is
~112 s of prefill before much output exists.

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
