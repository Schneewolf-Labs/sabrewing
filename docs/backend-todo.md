# inkling backend — loose ends

Tracking the gaps left by the fast initial build. Cross items off as done
(`[x]`) with a one-line note on the fix. Ordered within each tier by priority.

## 🔴 Correctness — quietly wrong or unverified

- [~] **MTP speculative decode** — draft+verify with the shipped 8-module head.
  Phase 1 (n-gram scaffolding + conv rollback) and depth-0 drafting are DONE and
  lossless-validated; the depth-0 recipe is correct (real acceptance 38.5% → the
  greedy `generate_spec_mtp` path is ~1.4×). Remaining: (a) the multi-depth chain
  is still ~random for depth ≥1 — needs the checkpoint's `mtp_model.py` (not in
  the public HF repo) to pin, would lift toward 2–3×; (b) wire the drafter into
  the *sampling* serve loop (rejection sampling) and gate the head-load behind a
  spec flag (currently opt-in via `MTP=1`). Full writeup: `docs/mtp-design.md`.
- [x] **Real-checkpoint perplexity oracle** — engine now reports teacher-forced
  perplexity; `make_tiny_inkling.py` emits the transformers `ppl_ref`. f32 matches
  to 0.00% (bit-faithful forward), int4 within quant noise (converter faithful).
  Works on any token list, so `./inkling N 0 tokens.json` is also a real-model
  health check.
- [x] **Multimodal token rejection** — text-only load, but an image/audio
  placeholder token (200054 / 200053) in a prompt reads a meaningless embedding
  row instead of erroring. Detect and reject with a clear message.
- [ ] **KV cache not trimmed to the sliding window** — global layers keep full KV;
  55/66 layers only need 512 tokens but we store all. Long context over-allocates
  and never recycles across requests. Trim sliding layers to their window.

## 🟡 Robustness — will bite in production

- [x] **Per-request context-length guard** — a prompt longer than `max_t` overruns
  the KV buffer (memory corruption, not a graceful 400). ~20 min.
- [ ] **Serial serve only** — one request at a time, no batching. Fine for personal
  use; bad the moment two clients connect.
- [x] **Sampling lacks repetition penalty / min-p** — base-model completions loop.
  The chat template mostly hides it; still thin.
- [ ] **CANCEL only honored between tokens** — a request stuck in a ~35 s cold
  expert fill can't abort until the token lands.

## 🟢 Performance — known, roadmapped (not loose ends)

- [~] GPU expert compute — ~90% of warm decode is CPU expert matmul. The int4
  expert GEMM kernel (`ink_cuda_matmul_q4`) is DONE and validated token-exact vs
  the CPU path (`--cuda-q4-test`). Next: the VRAM expert cache tier (hold hot
  experts on-device so the kernel runs without per-token PCIe upload) + dispatch
  in `moe()`. Design + steps: `docs/gpu-experts-design.md`.
- [ ] mmap expert path — RAM ≥ model machines (the R740 story): page cache = expert cache
- [~] int8 residents (`Q8=1`) — per-row int8 quant of the plain [O,I] residents
  (attention, dense, lm_head) kept in VRAM at half the bf16 footprint, freeing
  room for the GPU expert tier (and helping smaller-VRAM GPUs). Lossless on the
  real model (ppl 8.01 bf16 → 7.35 int8; the int8 path uses full-f32 activations).
  `ink_cuda_matmul_q8` validated vs CPU. **Measured: the GPU expert tier went
  402 → 780 experts (11.4 → 22.1 GB VRAM), ~1.94x, from the ~11 GB freed.**
  Remaining: int8 the fused shared experts too (currently bf16); make it the CUDA
  default once proven.
- [x] **router-driven prefetch — RULED OUT by measurement.** Instrumented the miss
  composition (`[misses]` line): on a novel prompt, **94% of misses are
  cold-first-touch** (expert never seen this generation) vs only **6% churn**
  (evicted then re-needed). History/temporal prefetch (prev-token, recent-window)
  can only catch the 6% churn tail — the 94% are experts reached for the first
  time, absent from all prior routing, so no history-based predictor can prefetch
  them. Router near-miss overfetch also ruled out (`[overfetch probe]`): at M=16
  only 15% of cold-touches were in the previous token's top-16, 85% jump into the
  top-6 from below rank 16 — and overfetching top-16 is 2.7× the disk traffic to
  catch 15%, net-negative. The only prefetch that *could* help is a speculative
  cross-layer router-preview (predict layer L+1's routing from L's hidden state,
  to load ahead of the sequential dependency) — research-grade, deprioritized.
- [ ] **capacity is the lever** (94% cold-first-touch ⇒ the expert must be resident
  before first use). Sub-levers, in order: (1) int8 residents as the CUDA default
  (frees VRAM → 780 vs 402 experts on-device; proven lossless); (2) **REAP** —
  shrink the 256-expert universe so a fixed RAM+VRAM budget covers a larger
  fraction AND routing redistributes onto hotter (more-likely-resident) experts;
  needs an Inkling converter + ppl-gated depth sweep (GLM's safe point was
  K=192/256); (3) push the RAM cache cap toward the ~87/layer RAM ceiling. Hard
  ceiling: 465 GB experts vs ~195 GB addressable — capacity narrows the gap
  (coverage → fewer cold touches) but cannot fully close it without a RAM upgrade.

## Done

- **Multimodal token rejection** — `prompt_reject()` refuses prompts containing
  the image/audio placeholder tokens (200054/200053).
- **Context-length guard** — `prompt_reject()` rejects `np + max_tokens > CTX_MAX`
  (default 8192) before it can overrun the KV buffer.
- **Repetition penalty** — `apply_rep_penalty()` in the serve/CLI decode loops,
  ring of 128 recent tokens, `REP_PEN` env (default 1.1). Oracle harness path
  unaffected (validation stays greedy-exact).
- **MTP converter** (stage 1) — `--mtp` mode converts the head into the snapshot.
- **upstream-sync tooling** — vendor colibrì's shared substrate.
- **Perplexity oracle** — teacher-forced PPL vs transformers `ppl_ref`; f32 0.00%
  diff (forward bit-faithful), int4 within noise (converter faithful). Doubles as
  a real-model health check on any token list.
- **LoRA adapter serving** (`c/lora.h`) — Tinker raw adapters: residents merge at
  load (`W += (α/r)·B·A`, RNE into bf16), routed experts stay int4 with a resident
  f32 low-rank correction hoisted out of the expert loop by linearity (<1% cost).
  Token-exact oracle (`make_tiny_lora.py`) passes both paths, 0.00% ppl vs merged
  transformers. `-l <dir>` / `LORA=<dir>` / gateway `--lora`.
- **Gateway thinking-effort handling** — `reasoning_effort` off prefills the
  content channel so the model can't open a thinking block and strand the answer
  as empty (was a seed-dependent failure); thinking-on is surfaced as
  `reasoning_content` in both stream and non-stream instead of being dropped.
