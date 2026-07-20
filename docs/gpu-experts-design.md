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

## First step — DONE

`ink_cuda_matmul_q4` added + validated token-exact vs the CPU path
(`--cuda-q4-test`, scaled-rel 1.6e-5).

## v1 — LANDED (static VRAM expert tier)

`Slot` gained device copies (`d13`/`ds13`/`d2`/`ds2`) + a `gpu` flag; `pins_load`
uploads the globally-hottest pinned experts (by usage) into VRAM up to a budget
(free VRAM − 3 GB headroom, `CUDA_EXPERT_GB` cap); `moe()` pass 3 dispatches the
GPU kernel for VRAM-resident experts. All `#ifdef COLI_CUDA`; CPU builds
byte-identical (oracle 0.00%).

**Measured on the real base head (A6000, 48 tokens, novel prompt):**
- 402 experts → 11.4 GB VRAM in 1.0s; output coherent.
- `expert-mm`: GPU 18.8s vs CPU 21.4s (~12%).

**Why the win is small (and what it means):**
1. **VRAM-capacity bound** — only 402 of ~2496 hot experts fit in ~11 GB free
   (residents eat ~37 GB), so most cache hits still land on RAM (CPU). Only the
   top ~2.4% of experts are on-device.
2. **Novel-prompt decode is disk-bound, not compute-bound** — `fill` (NVMe misses)
   is 167–222s vs `expert-mm` ~20s. GPU expert compute is a **warm-regime lever**
   (trained cache, ~100% hit, the docs' 2.5 tok/s scenario), not a novel-prompt one.

## Next (to actually unlock it)

1. [x] **int8 residents — DEFAULT under CUDA** (`Q8=0` opts out). Frees VRAM so the
   expert tier went **402 → 780 resident (22.1 GB)**. Lossless (ppl 7.35 vs 8.01).
2. [x] **Fused GPU expert kernel** (`ink_cuda_expert_q4`) — gate+up → silu-glu →
   down in one launch/sync on device, no host silu bounce, 2→1 syncs per expert.
   Decode-only (S=1), no-LoRA; token-exact (`--cuda-q4-test`, scaled-rel 2.2e-5).
   The VRAM experts are the hottest (~1/3 of routing mass in-distribution), so this
   is where the warm-decode per-expert overhead actually is. (colibri.c parallel:
   `coli_cuda_expert_group` — this is the Inkling equivalent.)
3. **Dynamic RAM↔VRAM repin** (colibri.c `repin_pass`) — keep the *actually-routed*
   experts on-device, not just the statically-hottest.

Note (2026-07-20): these are **warm-regime** levers (compute efficiency on the
resident tier). They do **not** help the novel-prompt disk wall — that's 94%
cold-first-touch (capacity-bound, see the coverage measurement above and
`backend-todo.md`). Warm and cold are separate problems; these ship the warm side.

## Coverage measurement (2026-07-20) — where the disk wall really is

Grounded the "does it fit" question. Model: 256 experts/layer × 64 sparse layers,
~28 MB int4 each = **~465 GB experts** vs 176 GB free RAM + 48 GB VRAM.

**Cold coverage** (analytic, from the `.coli_usage` histogram — fraction of routing
mass in the top-X hottest experts per layer): top-40 → 63%, top-79 → 82%,
top-192 → 99%. But top-192 = 355 GB, does not fit RAM.

**Two regimes** (the actual finding):
- *Warm / in-distribution*: LRU reuse within a generation lifts the live hit ~20 pts
  above cold coverage. ~80/layer (146 GB, fits RAM today) → ~90–96% hit. Basically
  fits now; REAP not required.
- *Novel prompt*: measured 82.3% hit / 0.12 tok/s at 40 pins, disk-bound (`fill`
  289 s vs `expert-mm` 18 s). **Pin quality is irrelevant here** — a clean ranking
  (82.3%) ≈ a FORCE-poisoned one (83.5%), because a novel domain routes to experts
  that were never in the history, so they miss regardless of what's pinned.

**Compute ceiling** (`FORCE_EXPERTS=1`, fixed expert set, 100% hit): 2.20 tok/s —
so the disk tail costs the novel-prompt run ~18×, and `expert-mm` (~18 s) is
near-identical warm or cold, i.e. compute is not the bottleneck.

**Fit arithmetic**: addressable ≈ 176 GB RAM + ~30 GB VRAM (int8 residents) ≈
195 GB. Zero-disk fit needs experts ≤195 GB → prune to K≈107/256 (42%), deeper
than GLM's repetition-loop threshold (144) — quality-fatal. REAP-192 (safe) + RAM
+ VRAM ≈ 55% resident → ~92% cold / ~97% warm; halves the novel gap but is not a
full fit. **Conclusion: the residual novel-prompt cold-touch is a _prefetch_
problem, not a capacity one** — prefetch (overlap the ~35 ms load with compute)
is the only lever that helps first-touch regardless of distribution, at zero
quality cost. REAP + int8 + RAM raise the ceiling but cannot make 465 GB fit 195 GB.

Diagnostic knob: `FORCE_EXPERTS=1` (routes every token to experts 0..K-1; implies
`USAGE_SAVE=0` so it can't poison the ranking).
