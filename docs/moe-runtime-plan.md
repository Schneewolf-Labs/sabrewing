# Plan: an MoE-native runtime for sabrewing

> Status: **Phase 0 complete** (GLM, Inkling, and Laguna all run token-exact;
> Laguna reached Inkling parity — SIMD int4, KV cache, serve mode, and an A6000
> CUDA tier with a VRAM expert cache at ~12 tok/s / 5× CPU). This document is the
> organizing plan for Phases 1–5: collapsing the three per-model engines into one
> MoE-native runtime. No code has landed for it yet.

## The thesis (what's wrong with both baselines)

**llama.cpp** is a *graph-general* runtime. Its MoE support is a special-cased op
on top of a dense-first design: coarse expert offload (`-ngl`), OS page-cache
residency, no awareness that routing is *predictable* or that a token's experts
could run on two compute units at once. Excellent at everything; optimal at MoE
serving specifically — no.

**Current sabrewing** is the opposite: per-model *monoliths* (`glm/inkling/
laguna.c`, 1,200–5,200 lines each, copy-pasted substrate) with a bolt-on LRU+pin
expert cache. That work measured the real wall — on a model that doesn't fit,
decode is **94% cold-first-touch** (an expert is needed for the first time this
generation, so history/LRU can't help) and stalls at ~0.1 tok/s. The Inkling work
proved capacity is the only *static* lever and explicitly parked the one dynamic
lever that could work: **cross-layer routing lookahead.**

The bet for the new architecture: **stop treating experts as a cache to fill
on-demand, and treat MoE decode as a dataflow you can run ahead of.** Compute
layer L's experts *while* the loads for layer L+1 are already in flight, and split
each token's active experts across every compute+memory bus you own at once. That
is the thing neither baseline does, and it is exactly the MoE structure to exploit.

## Five pillars

**1. A model-agnostic MoE core + thin arch descriptors.**
Kill the copy-paste. One runtime; each model becomes a small *descriptor* —
attention flavor (RoPE/YaRN/sliding/global schedule), router semantics (sigmoid
loss-free, softmax, top-k, scaling), layer plan (dense/MoE, shared-expert count),
norm placement. GLM, Inkling, Laguna all reduce to descriptors + shared expert/
attention/quant machinery. This is the *migration*: it de-dups what we have and is
the precondition for everything below (you can't optimize three monoliths in
parallel). Improves on llama.cpp by staying a file you can read; improves on us by
not being three of them.

**2. A unified hierarchical pager (the memory subsystem).**
One pager owns **VRAM ↔ RAM ↔ NVMe** for *both* experts and KV, in fixed-size
pages (an expert-block, a KV-block). Explicit — not OS page cache — because *we
know the access pattern* (the router tells us). RAM tier is mmap-backed for
zero-copy load and cross-restart warmth; NVMe tier uses **io_uring** (we already
have `uring.h`) with deep queue depth so cold reads overlap rather than stall (the
research showed QD~1 latency-bound was the killer). Eviction is heat-driven,
unified across experts and KV. Subsumes the current ad-hoc LCache/Slot/tier code
into one coherent thing, and makes "experts in and out of RAM cache" a first-class,
measurable mechanism rather than emergent LRU behavior.

**3. The overlap engine (the scheduler — this is the differentiator).**
Two moves that hide memory latency behind compute:
- **Pipeline the known future.** The instant layer L's router fires, submit layer
  L+1's loads… except you don't yet know L+1's experts. So:
- **Cross-layer routing lookahead.** A cheap predictor (start heuristic: "next
  layer reuses a biased subset given this layer's routing + position"; upgrade
  path: a tiny learned head, or the model's own MTP/DFlash draft states, which
  *already* preview future hidden states). Predict L+1's expert set from L's
  activations, prefetch speculatively, confirm when the real router fires, pay only
  for mispredicts. This attacks the sequential dependency that makes cold-first-
  touch unavoidable today. For a model that fits RAM (Laguna) it hides PCIe; for
  one that doesn't (Inkling) it hides NVMe — same mechanism, the wall either way.

This is the piece the Inkling notes called "research-grade, deprioritized." It is
the whole point of the new architecture.

**4. Split-bandwidth dispatch (use all compute at once).**
Per token, an expert set is partitioned by *where its weights already live*:
VRAM-resident experts → GPU int4 kernel; RAM-resident → CPU AVX-512 VNNI; both
**run concurrently** and combine. Today we do GPU experts *or* CPU experts; here a
single token's active experts saturate the A6000's ~768 GB/s *and* DDR5's ~60 GB/s
simultaneously. Plus **grouped/batched expert GEMM** (Megablocks-style: sort tokens
by expert, load each expert's weights once, apply to all its tokens) — how you
serve *many* agentic requests without re-streaming, and how prefill goes fast.
Neither baseline splits a token across buses.

**5. An MoE-native quant format.**
Replace per-row symmetric int4 with a container designed for the sparsity: **group
scales** (K-quant-style 32-elem sub-blocks — the heat-quant research already
fingered this as the #1 quality lever) + **imatrix** importance weighting +
**heat-tiered mixed precision** (hot experts 4-bit, cold tail 2–3-bit, driven by
measured routing mass). Buys both quality-per-byte *and* capacity — a bigger
fraction of the model in fast memory, which multiplies with pillars 2–4. The prior
heat-quant analysis already has the Pareto numbers; this productionizes them.

## Sequencing (each phase gated on the non-negotiables)

0. **Finish Laguna to Inkling parity** — SIMD int4 kernel, KV cache, serve mode,
   CUDA tier. **DONE.** Also stress-tested the descriptor boundary and gave a
   second/third MoE arch to generalize from.
1. **Core refactor** → shared runtime + GLM/Inkling/Laguna descriptors. Highest
   leverage, lowest risk. *(The migration this branch begins.)*
2. **Unified pager** → RAM/VRAM + mmap first; io_uring NVMe tier second.
3. **Overlap engine** → pipeline prefetch (easy win) → cross-layer lookahead (the bet).
4. **Split-bandwidth dispatch** + batched expert GEMM.
5. **MoE-native quant** → group scales → imatrix → heat-tiered.

## Two non-negotiables that keep it *sabrewing* and not a ggml clone

- **Token-exact oracle survives every phase.** What makes this engine trustworthy
  is that each arch validates bit-for-bit against `transformers`. No optimization
  lands that can't reproduce the oracle (quant paths validate within measured
  noise, as today). This is the design constraint that stops "fast but subtly wrong."
- **Readable and dependency-free.** The core stays something you can read
  end-to-end. We borrow llama.cpp's *ideas* (group scales, SIMD block layout, mmap,
  flash-attn online softmax, quantized KV) reimplemented in the small engine —
  never vendor the codebase, because the whole value proposition is comprehension +
  a research surface. Distill the idea, but respect *why* the complexity exists
  before throwing it out: ggml's quant layouts are ugly because they're SIMD-optimal;
  llama.cpp has a dozen K-quants because quality-per-bit is real. Keep those, drop
  the breadth. Simple-because-focused, not simple-because-naive.

## The one-line version

**A dataflow MoE runtime that predicts its own expert working set a layer ahead,
pages it through a unified VRAM/RAM/NVMe hierarchy, and executes each token across
CPU+GPU simultaneously — with a heat-tiered group-scale quant for capacity.**
llama.cpp doesn't specialize on MoE serving; today's sabrewing specializes but
reacts instead of predicts. This predicts.
