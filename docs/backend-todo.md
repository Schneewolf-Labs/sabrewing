# inkling backend — loose ends

Tracking the gaps left by the fast initial build. Cross items off as done
(`[x]`) with a one-line note on the fix. Ordered within each tier by priority.

## 🔴 Correctness — quietly wrong or unverified

- [ ] **MTP speculative decode** — the checkpoint ships an 8-layer MTP head; we
  skip it. Free ~2–3× decode via draft+verify, and wiring it also exercises the
  draft path against real weights. *(in progress)*
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

- [ ] GPU expert compute — ~90% of warm decode is CPU expert matmul
- [ ] mmap expert path — RAM ≥ model machines (the R740 story): page cache = expert cache
- [ ] int8 residents — smaller-VRAM GPUs and 128 GB boxes
- [ ] speculative cache warming — measure (token,layer,expert) routing predictability first

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
