# GPU expert compute — design & scoping

The single biggest decode lever: at high cache-hit rates ~90% of warm decode is
**CPU routed-expert int4 matmul** (`moe()` pass 3, `matmul_q4`). Everything else
(bf16 residents: attention, dense MLP, shared experts, lm_head) already runs on
the A6000 via the deliberately-tiny CUDA backend (`ink_cuda_matmul_bf16`).

## The constraint is bandwidth, not FLOPs

Decode reads ~35 GB of weights per token; CPU decode is capped by DDR5 bandwidth
(~1.2 s/token). The GPU feeds the same reads at ~12×. But the routed experts are
**469 GB** — they cannot live in 48 GB of VRAM. And the naive fix (upload each
token's active experts to VRAM, matmul, discard) does not help: ~6 experts/layer
× 66 layers × ~28 MB ≈ 11 GB per token over PCIe (~25 GB/s) ≈ 0.4 s/token of
upload — the same wall, moved to a different bus.

**So the win requires a VRAM expert cache TIER**: keep the *hottest* experts
resident in VRAM and matmul them on-device; only cache misses pay the upload.
With the current RAM-cache hit rates (80–94% from pins + LRU), most expert
matmuls would run on the GPU with no per-token transfer.

## What already exists to build on

- `ink_cuda_matmul_bf16` / `ink_cuda_upload` — device matmul + allocation.
- The RAM LRU + pinned expert cache (`LCache`/`Slot`), heat counters (`eusage`).
- `tier.h` `tier_pick_swap()` — heat-based promotion of a hot expert into a
  hot-store slot (RAM/VRAM), with anti-ping-pong margin. Written for exactly this.
- The `EMAP` dashboard protocol already encodes a 2-bit tier (0 disk / 1 RAM /
  2 VRAM) per expert — the VRAM tier is plumbed for display, just not populated.
- `model_init` already reserves ~3 GB VRAM headroom "for activation buffers +
  future tiers" (inkling.c:516). ~11 GB of headroom is available on the A6000
  after the ~37 GB of bf16 residents.

## Plan

1. **GPU int4 expert GEMM kernel** — dequant the packed-nibble container
   (U8 low/high nibble, +8 offset, per-row f32 `.qs` scales) × f32 activation →
   f32 out, for the fused gate+up (`D→2I`) and down (`I→D`) matmuls. Mirrors
   `matmul_q4`. **Milestone gate: token-exact vs the CPU kernel** on the tiny
   oracle (`IDOT=0` scalar path is the reference), the same bar every other
   kernel in this repo clears.
2. **VRAM expert cache tier** — a GPU-resident tier above the RAM cache: upload
   hot experts (packed int4, ~28 MB each) to VRAM, sized to the ~11 GB headroom
   (~400 experts). Evict by heat (`tier_pick_swap`). Slot gains a `dev` pointer
   like `Wt`.
3. **Dispatch** — in `moe()` pass 3, if the expert is VRAM-resident, run the int4
   GEMM on-device; else fall back to the CPU path (unchanged). Activations for a
   token's experts batch to the GPU once, not per-expert.
4. **Tier management + tune** — promote RAM→VRAM by heat each turn (reuse the
   pin/usage machinery); measure the real decode speedup (bounded by the VRAM
   tier hit rate, just as CPU decode is bounded by the RAM hit rate).

## Effort & risk

Multi-session. The int4 GEMM kernel + token-exact validation is the first
self-contained chunk (and de-risks the numerics before the caching plumbing).
The VRAM tier is the larger systems piece (allocation, eviction, upload
scheduling). Payoff: the 90% moves to ~12× bandwidth for VRAM-resident experts —
the largest single decode win available, and it **multiplies** with MTP
speculative decode (fewer, fuller forwards, each faster).

## First step

`backend_cuda_ink.cu`: add `ink_cuda_matmul_q4(y, x, packed, scales, S, In, Out)`,
validate token-exact against `matmul_q4` via the tiny oracle (bits=0 container),
before any caching work.
