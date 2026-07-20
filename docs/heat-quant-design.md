# Heat-tiered expert quantization — analysis & design

Sub-int4 precision as the *bytes-per-expert* capacity lever (complements REAP's
*expert-count* lever). Both shrink the 465 GB expert working set that drives the
novel-prompt disk wall (see `gpu-experts-design.md`).

## The core idea

Quantize experts **by heat**: keep the hot experts at int4, push the cold tail to
int2/int3. Damage is concentrated exactly where the routing mass isn't.

## Why it works — an asymmetry (measured 2026-07-20)

Two measurements on the real int4 model (`.coli_usage` mass + per-expert
reconstruction error, layer 30 probe; error is Frobenius `‖W−W_q‖/‖W‖` of a
dequant→requant→dequant round-trip):

1. **Per-expert quant error is uniform across heat.** The hottest expert (rank 0,
   6.1% mass) and the coldest (rank 255, ~0% mass) both degrade identically
   (int2-per-row ≈ 0.55, int3-per-row ≈ 0.29). Cold experts do **not** tolerate
   low precision intrinsically better — so heat-tiering is a *mass-weighting*
   argument, not an intrinsic-robustness one.

2. **Mass is concentrated but bytes are distributed.** Top-40/256 experts/layer
   carry ~64% of routing mass but are only 16% of the expert *count*. So you can
   protect 64% of the quality by keeping just 16% of the bytes at int4, and save
   bytes broadly by pushing the other 84% of experts to int2.

Model damage ≈ Σ_e mass_e · err_e. Zero the error on the high-mass hot set →
total damage collapses even though most experts are now int2.

## The Pareto table (damage = mass-weighted rel-err; 0 = lossless int4)

| scheme | eff. bpw | damage | experts |
|---|---|---|---|
| uniform int3 (per-row) | 3.00 | 0.286 | 349 GB |
| uniform int2 (per-row) | 2.00 | 0.548 | 233 GB |
| uniform int2 (group-64) | 2.25 | 0.427 | 262 GB |
| **heat: top-40 int4 + rest int2-row** | **2.31** | **0.196** | **268 GB** |
| heat: top-40 int4 + rest int2-g64 | 2.31 | 0.153 | 268 GB |
| heat: top-79 int4 + rest int2-g64 | 2.62 | 0.075 | 305 GB |

**Heat top-40 is smaller than uniform int3 (2.31 vs 3.0 bpw) AND less damaging
(0.196 vs 0.286).** A strict Pareto win. top-79 + group scales lands at 2.62 bpw
(305 GB, −34% vs int4) with near-int4 damage.

## Honest caveats

- **Reconstruction error ≠ perplexity.** 0.196 mass-weighted rel-err is a
  pessimistic proxy (naive round, no error correction; MoE has redundancy). The
  real ppl test is still owed — but a full-model teacher-forced forward on sabre
  is disk-bound and times out (>7 min even for 20 tokens), so ppl iteration needs
  a faster path (smaller calibration model, or an offline expert-only harness).
- **Absolute cold-expert error is large** (int2 ≈ 0.5). Even mass-weighted-small,
  a prompt that routes heavily to a degraded cold expert could dip. Group scales
  (int2-g64: 0.43 vs 0.55) and GPTQ/AWQ-style error correction would cut this
  materially — that's the difference between "viable 2-bit" and "naive 2-bit".
- **The `.coli_usage` ranking is weak** (~1587 tokens). A larger multi-domain
  calibration set sharpens the hot/cold split and the mass estimate.
- **Ternary/binary need QAT** (BitNet) — not reachable by PTQ from a bf16 model.
  Off the table without a distill/retrain.

## Payoff

268 GB (heat top-40) is 42% smaller than int4's 465 GB → the disk working set
shrinks proportionally (fewer/faster cold misses), and combined with REAP or a
modest RAM bump it plausibly becomes RAM-resident → collapses the disk wall
toward the 2.2 tok/s compute ceiling. This is the same prize REAP chases, reached
on the orthogonal (bytes-per-expert) axis, and the two **compose**.

## Probe scaffold (shipped, env-gated, default off)

`slot_fill` degrades a cold expert's int4 nibbles to N-bit in place (per-row,
sharing the int4 row scale — the naive floor). Controls:
- `QSIM_BITS=2|3` — target bits for degraded (cold) experts.
- `QSIM_HOT_KEEP=N` — experts/layer kept at int4 (default = `PIN_N`, the pinned
  hot set); heat computed from the `.coli_usage`-seeded per-layer threshold.
- `QSIM_ALL=1` — degrade every expert (uniform control).

This is a *quality* probe (perf irrelevant). A real implementation would store
mixed-precision experts on disk + add an int2 GEMM path (and ideally group scales
+ error correction). The weight-space Pareto analysis above (instant, no forward)
is the tool for iterating the scheme before that build.

## Recommended next steps

1. **Group scales for the int2 path** (int2-g64) — the single biggest quality
   lever, already measured (~0.43 vs 0.55). Prerequisite for viable sub-int4.
2. **A fast ppl harness** — offline expert-only error → output MSE on a fixed
   activation set, or a tiny calibration model, to turn the reconstruction proxy
   into a real quality number without the disk-bound full forward.
3. **Then** build the mixed-precision storage + int2 GEMM, gated on that ppl.
