# Linear attention (Qwen3.5's gated delta net)

Qwen3.5-35B-A3B replaces softmax attention in **30 of its 40 layers** with a *gated delta
net* — a recurrent linear-attention mixer. Only every 4th layer is full attention
(`full_attention_interval: 4`). That is the part of the architecture worth caring about here,
because it attacks the constraint that actually limits this box: KV capacity.

Status: **working and token-exact.** The kernel is `c/moe_linattn.h` (validated standalone by
`make linattn-test`), the engine is `c/qwen35.c` (validated end-to-end by `make qwen35-test`),
and `tools/convert_qwen35_int4.py` produces the 21.7 GB int4 container it runs on. Single
stream only — no batching, no CUDA tier, MTP draft head converted but unused.

## Geometry (from the shipped `config.json`)

| | |
|---|---|
| hidden | 2048 |
| layers | 40 — 30 `linear_attention`, 10 `full_attention` |
| linear heads | 16 key / **32 value**, `k_dim = v_dim = 128`, conv kernel 4 |
| full-attn heads | 16 query / 2 KV, `head_dim` **256**, `attn_output_gate: true` |
| MoE | 256 experts, top-8, `moe_intermediate_size` 512, shared expert 512 |
| vocab | 248320 · max positions 262144 |

Parameter split: the MoE is ~32 B of the 35 B (256 experts × 3 × 512 × 2048 × 40 layers), the
linear-attention projections are ~1.0 B, full attention ~0.27 B. So the mixer is cheap in
weights — its cost is state traffic, not GEMM.

## The recurrence

Per head, per token. `q` and `k` are L2-normalized (eps added to the *sum*, not the mean), `q`
pre-scaled by `1/sqrt(k_dim)`, `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`:

```
S   <- S * exp(g)          decay (g < 0), a scalar per (head, token)
mem <- k^T S               [v_dim]
d   <- (v - mem) * beta
S   <- S + k (x) d         rank-1 update
out <- q^T S               read the UPDATED state
```

Two orderings here are easy to get wrong and both produce plausible activations rather than
obvious garbage: the decay applies **before** the memory read, and the output reads the state
**after** the update. `tools/linattn_check.c` pins both — inverting the first moves relative
L2 error from `3e-7` to `1.2e-1`, so the test discriminates rather than merely passing.

Around it: `in_proj_qkv` → depthwise **causal** conv1d (kernel 4, per-channel) → SiLU → split
q/k/v → recurrence → `RMSNormGated(core, z)` = `x * rsqrt(mean(x²)+eps) * weight * silu(z)` →
`out_proj`. Value heads share key heads 2:1 (`repeat_interleave`).

## Why this matters for serving: O(1) state

The per-sequence state is one `[k_dim × v_dim]` matrix per head — **2.1 MB per layer,
independent of context length** — plus a 128 KB conv window. Against the full-attention layers
at f32:

| context | 30 linear layers (state) | 10 full-attn layers (KV) | total |
|---|---|---|---|
| 2 k | 67 MB | 84 MB | 151 MB |
| 8 k | 67 MB | 336 MB | 403 MB |
| 32 k | 67 MB | 1.34 GB | **1.41 GB** |
| 256 k | 67 MB | 10.7 GB | 10.8 GB |

For comparison, one Laguna slot at 32 k reserves **3.52 GB** (see `kv-cache-design.md`), and a
hypothetical all-softmax version of this model would need 5.37 GB.

Read the table honestly in both directions. The linear state is a **fixed 67 MB tax**: full
attention across 10 layers costs 40 KB/token, so below roughly **1700 tokens the state costs
more memory than the attention layers do**. It only wins past that — and then it wins
enormously, because it never grows. At 32 k it is 20× smaller than the attention KV; at 256 k,
160×. For the agent-fleet workload (long-lived conversations, 187 GB of RAM) that is the
difference between ~34 concurrent 32k sessions and ~130.

## Measured cost

`make linattn-test` benches the recurrence at the real shapes (30 layers, 32 heads, 128×128),
min-of-5, rotating through 30 separate states so each step touches a different layer's memory
the way the real model does. Ryzen 9 7900, 12 cores, head-parallel via OpenMP:

| kernel | decode (S=1) | prefill (S=128) |
|---|---|---|
| naive, 3 sweeps | 1.34–1.53 ms/tok | 0.428–0.444 ms/tok (~2270 tok/s) |
| **fused, 2 sweeps** | **0.98–1.00 ms/tok** | **0.333–0.355 ms/tok (~2900 tok/s)** |

### The fused kernel

The naive form walks the state three times (decay, update, output) and writes it twice. Two
identities collapse that to two sweeps, one of them read-only:

1. **decay is a scalar**, so it factors out of the read: `k^T (decay·S) == decay · (k^T S)`.
   No decay pass is needed — fold it into the update sweep as `S = decay·S + k⊗d`.
2. **the output expands**: `q^T (decay·S + k⊗d) == decay·(q^T S) + (q·k)·d`, so `out` can be
   accumulated in the same read-only sweep as `mem`, from the *old* state, then corrected with
   one dot product.

`2R + 1W` instead of `3R + 2W` — 40% less state traffic, and measured 1.4× on decode, 1.25× on
prefill. It is not bit-identical to the naive form (different summation order for `out`), so
both are checked against `transformers` separately rather than against each other; the naive
form is kept as the literal reference, same as the `MOE_QK_*` modes in `moe_attn.h`.

### Where the remaining cost sits, and the caveat

Arithmetic intensity is ~3 flops per state element touched, so this is a bandwidth problem,
not a FLOP problem. The prefill number works out to ~275 GFLOP/s, about 15% of this CPU's
AVX-512 peak — and a per-core estimate puts it almost exactly at the L2 bandwidth/compute
crossover (~3072 cycles of L2 traffic against ~3072 cycles of FMA per head-token). So the
naive-scan shape is close to its own ceiling, and the way past it is the **chunked/WY form**,
which turns per-token rank-1 updates into blocked GEMMs and raises intensity. That is real
work and is not needed yet.

**The decode number is optimistic and should be read as a floor.** 63 MB of state against this
CPU's 64 MB of L3 means the bench is largely cache-resident: 189 MB of traffic in 1.00 ms
implies 189 GB/s, well above the ~83 GB/s this DDR5 config can actually deliver. In the real
model, streaming ~500 MB of expert weights per decode step evicts all of it, which puts true
decode cost somewhere in **1.0–3.2 ms/token**. Even at the pessimistic end that is a few
percent of a decode step, so decode is not where this hurts.

**Prefill is where it hurts, relatively.** 0.34 ms/token does not amortize — it is per token,
whereas prefill amortizes the MoE weight reads across the whole batch. Against our best
measured prefill on Qwen-27B (1083 tok/s ≈ 0.92 ms/token) the recurrence would add ~27% on
top. That is the argument for the chunked kernel, and it is a prefill argument only.

## What this changes about prefix caching

Our reuse machinery (`psnap_*`, the longest-common-prefix walk in `slot_admit`) assumes
positions are **independent**: a KV cache can be truncated to any position, because position
*p*'s K/V does not depend on what came after it. A recurrent state has no such property.
`S` after *n* tokens does not contain `S` after *n−k*, and there is no inverse — the decay
multiplied information away.

Consequences, and they are not symmetric:

- **Extending a cached prefix is cheaper and simpler than it is today.** A snapshot is a
  fixed 67 MB blob with no per-position bookkeeping, no ring, no `next_pow2(window + span)`
  rule. `linattn_check.c` already pins the property that makes this safe: feeding a sequence
  as 1-token steps, as one span, and as uneven spans `{5, 1, rest}` gives **bit-identical**
  output, so state and conv window carry correctly across a chunk boundary.
- **Rewinding is impossible.** When a conversation diverges from its cached prefix — which
  `kv_reuse_floor` handles today by keeping the common prefix and dropping the tail — the
  recurrent layers must replay from the newest snapshot at or *before* the divergence point.
  So snapshot granularity stops being a memory/latency tradeoff and becomes a **correctness-
  adjacent cost floor**: with `PSNAP_GRAN 256`, a divergence at token 5000 costs up to 255
  tokens of replay through the linear layers that attention would have paid nothing for.
  The full-attention layers can still be truncated normally, so a Qwen3.5 cache is genuinely
  two different data structures with two different reuse rules.
- **Do not quantize the state.** int8 KV already flips 6.4% of predictions on confident text
  and stays off. The state is worse in kind: KV error is read-only per position, while state
  error **compounds through every subsequent step** of the recurrence. This should be treated
  as f32-only until there is evidence otherwise, not as the next obvious win.

## The container, and the thing it changes

`tools/convert_qwen35_int4.py` converts the checkpoint: **71.9 GB bf16 → 21.7 GB int4**, as
`out-top` / `out-layer-000..039` / `out-mtp`, published at
[nbeerbower/Qwen3.5-35B-A3B-sabrewing-int4](https://huggingface.co/nbeerbower/Qwen3.5-35B-A3B-sabrewing-int4).

Three things about the source checkpoint were verified against the shipped tensors rather than
assumed, and two of them were surprises:

- Names nest under `model.language_model.` (it is a multimodal checkpoint,
  `Qwen3_5MoeForConditionalGeneration`); the converter strips that to `model.` so the container
  matches every other engine's.
- Routed experts arrive **already fused** as `gate_up_proj [E, 2I, D]` and `down_proj [E, D, I]`,
  used as `F.linear(x, W[e]).chunk(2, dim=-1)` — which is *exactly* the block-concat layout
  laguna.c already wants. No transpose, no de-interleave; the conversion is a row-quantize.
- The **MTP head uses the old per-expert layout** (`mlp.experts.{e}.gate_proj.weight`) while the
  main layers use the fused 3D form. One checkpoint, two conventions, so the converter has both
  read paths. `--selftest` asserts they produce byte-identical containers, because a silent
  transpose here would only ever show up as a broken model.

The vision tower (333 tensors) is skipped — no engine support.

**The size is the point.** At 21.7 GB the *entire* model fits in the 48 GB A6000 with room to
spare. Contrast Laguna: 57 GB of int4 routed experts against ~40 GB of spare VRAM, so ~21% of
MoE layer-calls fall back to the CPU expert tier — and those 21% of calls consume **70% of
expert time** (measured: 170 CPU calls at 14.66 s against 629 VRAM calls at 6.14 s, 8.8× slower
per call). Qwen3.5 makes that whole tier disappear. Between that and O(1) recurrent state, this
is the first supported model where the CPU expert path and the KV-capacity wall both stop being
constraints on the same box.

## The engine

`c/qwen35.c` runs it. **Token-exact against transformers** on the tiny oracle
(`make qwen35-test`): 36/36 teacher forcing, 24/24 greedy, perplexity to four decimals.
The int4 path is validated separately by dequantizing the container in Python and re-scoring
with transformers — C reproduces that digit-for-digit (ppl 99.7712 both sides, 10/36 both
sides on the tiny model, where int4 damage is large because 64-wide rows of random weights
have no redundancy to absorb it).

Real model, CPU only, 21.7 GB container: loads in 7.1 s at 19.7 GB RSS, prefill 30.3 tok/s,
**decode 6.66 tok/s** — already above Laguna's ~4 tok/s, with no CUDA tier written yet.

Two bugs found on the way to exactness, both from misreading the reference rather than from
faulty logic, and both worth recording because neither produces obviously-broken output:

1. **`Qwen3_5MoeRMSNorm` scales by `(1 + weight)`, not `weight`** — while
   `Qwen3_5MoeRMSNormGated` (the linear-attn norm) uses `weight` directly. Two conventions in
   one model, which is exactly why the standalone gated-delta-net check passed while the
   engine scored 0/36 at ppl 254. The `+1` is folded in at load so the shared `rmsnorm_row`
   kernel stays untouched.
2. **`sdpa_head`'s `tau` is a score multiplier, not a softcap** (`sc = tau*(dot*scale+bias)`).
   Passing `0` for "no cap" zeroed every score, turning attention into a uniform mean over V.
   Qwen3.5 has no logit cap, so `tau = 1`.

Also needed: `tok.h` learned the older space-separated merge format (`"Ġ Ġ"`) that Qwen3.5
still ships, alongside the pair-array form it already handled.

## Serving, and what it costs

`SERVE=1` puts `qwen35.c` on the same stdin/stdout protocol as the other engines, so the
gateway drives it:

```sh
CTX_MAX=16384 python3 openai_server.py --model /path/to/qwen35_i4 --engine ./qwen35 --port 8099
```

`openai_server.py` gained `ARCH=qwen35`: a renderer that **byte-matches the shipped
chat_template.jinja** (verified against `apply_chat_template` across 10 cases — plain,
system, multi-turn, tools, tool-call + multiple results, thinking on and off), a reasoning
splitter, and a tool-call parser for Qwen3.5's `<function=>`/`<parameter=>` syntax, which is
neither GLM's `arg_key`/`arg_value` nor Qwen3's JSON-in-`<tool_call>`. Serving is **serial** —
one request at a time, no slot pool, no prefix reuse.

Driven from egirl (agent loop, tools, memory) it works end to end: native tool call parsed,
executed, fed back, answered in 2 turns. The cost is where the design predicted:

| turn | input tok | output tok | wall |
|---|---|---|---|
| 1 (tool call) | 4928 | 59 | 256 s |
| 2 (final answer) | 4998 | 1 | 249 s |

**Turn 2 re-prefilled 4998 tokens to emit one token.** Prefill is ~19 tok/s at these lengths
(not the 30 tok/s a 22-token prompt shows), and nothing is reused between turns, so an agent
loop pays for its whole context on every step. Prefix reuse is the single biggest lever here
and it is worth more on this architecture than on Laguna: the linear layers' share of the
snapshot is a fixed 67 MB blob with no ring and no per-position bookkeeping. What makes it
non-trivial is the asymmetry in the section above — the 10 attention layers can be truncated
on divergence, the 30 recurrent ones cannot, so one cache holds two data structures with two
reuse rules.

## Still to do

Batching and the CUDA tier are not written: this is a single-stream engine, so none of the
serving levers (continuous batching, group-by-expert GEMM, prefix caching) apply to it yet,
and neither does the MTP draft head. Also missing: a `MoeSharedMode` variant for the sigmoid-gated shared expert
(`sigmoid(shared_expert_gate(x)) * shared(x)`, added to the routed sum — the router is plain
softmax → top-8 → renormalize, with no correction bias and no route scale, so `route_scale` is
1.0); the full-attention layers, which need a **per-element** output gate (`q_proj` emits
`[8192, 2048]` = 16 heads × 256 × 2, chunked into query and gate — the same M.1 gate-width trap
that already bit us on Laguna, so read the width from the checkpoint and fail loudly); mRoPE
(`mrope_section [11, 11, 10]`, `partial_rotary_factor` 0.25, theta 1e7) which degenerates to
standard RoPE for text-only input; a tiny random-init oracle (`make_tiny_qwen35.py`) checked
token-exact against `transformers`; then the engine.

The MTP head is converted but unused. It is the reason to expect single-stream decode to beat
the per-token weight-read floor here in a way it cannot on Laguna, which ships no draft head.
