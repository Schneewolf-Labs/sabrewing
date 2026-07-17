# inkling backend — loose ends

Tracking the gaps left by the fast initial build. Cross items off as done
(`[x]`) with a one-line note on the fix. Ordered within each tier by priority.

## 🔴 Correctness — quietly wrong or unverified

- [ ] **MTP speculative decode** — the checkpoint ships an 8-layer MTP head; we
  skip it. Free ~2–3× decode via draft+verify, and wiring it also exercises the
  draft path against real weights. *(in progress)*
- [ ] **Real-checkpoint perplexity oracle** — the tiny-model test proves the math,
  but nothing proves the 469 GB conversion is faithful. Drive the running API over
  a fixed text slice, compare NLL/perplexity to a transformers bf16 reference.
  This is the "can't rot silently" gate and what upstream asked for.
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
