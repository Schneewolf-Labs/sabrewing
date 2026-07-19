# Inkling MTP speculative decode — design & status

Goal: 2–3× decode via draft+verify using Inkling's shipped 8-module MTP head.

## The key property: losslessness makes MTP a *performance* concern, not a safety one

The verify step only ever emits a draft token that equals the main model's own
greedy token (or, under sampling, passes a rejection test that preserves the
target distribution). So a wrong, mis-wired, or approximate MTP head **cannot
corrupt output** — it can only lower the acceptance rate (slower) or raise it
(faster). This is why the drafter can be developed and tuned against a live
model without a token-exact oracle: correctness is guaranteed by the verifier.

## Phase 1 — DONE (`mtp-spec-decode`, commit ef21b06)

Speculative scaffolding + the one genuinely novel Inkling problem: rolling back
the four stateful causal short-convs on partial acceptance.

- `sconv_apply` optionally checkpoints per-position conv state (`vck` buffers,
  gated by `vck_on`); `conv_rollback(k)` restores all four convs to batch
  position `k`. KV is position-indexed, so rejected rows self-overwrite.
- Verify forward = `step()` over `[committed, draft…]`; its per-position argmax
  (`tf_out`) is the acceptance signal and the next committed token.
- `ngram_draft` (zero model cost) is the Phase-1 drafter. `generate_spec()` is
  greedy and token-identical to `generate()`.
- **Validated**: `SPEC_TEST=1` — speculative output == plain greedy, 160/160
  token-exact across draft widths G=1..8 on the tiny oracle (heavy partial-accept
  rollback stress). Base + LoRA oracles unchanged (0.00%). CUDA build compiles.

## The MTP head recipe (pinned)

Weights: `out-mtp.safetensors`, `model.mtp.{0..7}.*` (8 modules,
`num_nextn_predict_layers=8`). Each module is a full Inkling **dense** decoder
block (attention + rel bias + q/k norms + 4 short-convs + dense MLP at 24576)
plus three MTP-specific tensors:

- `input_proj` `[6144, 12288]` — fuses `[hidden ; embedding]` → hidden.
- `embed_norm` `[6144]` — RMSNorm on the token embedding.
- `hidden_norm` `[6144]` — RMSNorm on the incoming hidden.

Modules are **hybrid attention** (`mtp_local_layer_ids [0,2,4,5,6,7]` → 6 sliding
+ 2 global) and share the main model's `final_norm` + `lm_head` (no per-module
head tensors). Config flags: `mtp_hidden_states_first=True`,
`chain_hidden_post_norm=False`.

Per module `k` (predicting token *t+k+1*), input token `tok`, incoming hidden `h`:

```
emb = embed_norm_k( embed(tok) )
hn  = hidden_norm_k( h )
x   = input_proj_k( concat[ hn ; emb ] )        # hidden first (mtp_hidden_states_first)
h'  = InklingBlock_k( x )                        # same op sequence as step()'s layer loop
logit = lm_head( final_norm(h') / mup )          # shared head
tok_next = argmax(logit)  →  draft[k]
# chain: h ← h' (raw, chain_hidden_post_norm=False), tok ← tok_next
```

Module 0's incoming `h` is the main model's last-layer hidden (pre-final-norm)
at the current position.

Note: transformers deliberately drops the head
(`_keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]`), so there is **no
token-exact reference from `InklingForCausalLM`**. Validation is two-layer: a
hand-rolled torch reference (reusing `InklingDecoderLayer`) for C-impl fidelity,
and acceptance rate on the real model for recipe correctness.

## Phase 2 — torch reference DONE

`tools/make_tiny_mtp.py` builds a tiny MTP head (random weights in the real
`model.mtp.{k}.*` layout) and a **module-0 teacher-forced reference**: module 0
uses `t_{i+1}` (always committed), so its per-position predictions have no
circular dependency and pin the recipe cleanly. The reference drives
transformers' own `InklingDecoderLayer` for the block math.

Decisions pinned while building it:
- **Module 0's input hidden is the main model's PRE-final-norm hidden** (each
  module re-norms via its own `hidden_norm`; post-norm would double-norm). In C
  this is the residual stream `x` after the last layer, before `final_norm`.
  Verified in the reference by swapping `model.model.norm` to identity.
- Concat order is `[hidden ; embedding]` (`mtp_hidden_states_first=True`).
- MTP modules share the main `final_norm` + `lm_head`; `mup` logit scaling applies.

C integration points (mapped, ready to implement):
- Load: replicate the `LD`/`LDW` pattern (model_init:604-631) with a
  `model.mtp.%d.` prefix, reading from a **separate shards** for
  `out-mtp.safetensors` (`MTP=<dir>`; defaults to the snapshot dir). Store each
  module as `Layer` + `input_proj` (Wt) + `embed_norm` + `hidden_norm`.
- KV/conv: give module `k` a slot at index `n_layers+k` — set
  `c->local[n_layers+k]` (attention type), allocate `m->cs[j][n_layers+k]` and
  extend `kv_alloc` to `n_layers+n_mtp`, so `attention()` is reused verbatim.
- Expose the pre-final-norm hidden from `step()` (add an optional `float
  *hid_out` out-param).
- Oracle harness: `--mtp-oracle <ref_mtp.json>` runs main forward → pre-norm
  hidden → module-0 forward over positions → compare argmax to `mtp0_pred`.

## Phase 2 — recipe corrected & real-validated (depth 0)

The inferred recipe was wrong in three norm-boundary places — all invisible to
the self-consistent tiny oracle (C and the torch reference shared the wrong
assumptions; a self-consistent test only catches *disagreements*). The real
trained head, via `--mtp-accept`, exposed them. Corrected against vLLM's Day-0
Inkling source (`vllm/models/inkling/nvidia/{mtp,model}.py`):

1. **Embedding double-normed**: backbone `embed_norm` (whitening) THEN the depth's
   own `embed_norm` (near-identity trim).
2. **Input hidden is POST-final-norm**: the base `forward` returns
   `self.norm(hidden)`, which is also the drafter's `previous_hidden_states`.
3. **Output → `lm_head` directly**, no `final_norm` (mup is argmax-invariant).

Result on the real base head (175 tokens of prose), depth-0 acceptance:
**~3% → 38.5%**. Depth-0 alone is ~1.39x; the (imperfect) chain reaches ~1.48x.

**Multi-depth chain is unresolved.** vLLM ports only depth 0 (raises on
`spec_step_idx != 0`), and the checkpoint's reference `mtp_model.py` is not in the
public HF repo, so the exact 8-module chaining can't be read off. Measured
empirically (`--mtp-accept`, `MTP_MAIN_HIDDEN` toggle):
- sequential (module k from module k-1's output): depths 1-7 ≈ 20% (partial signal)
- parallel (all modules off the main hidden): depths 1-7 ≈ 1% (worse)
So the chain is sequential but the exact hidden/position/norm handling for depth
≥1 is still off. Needs the reference file, or more empirical search. Depth-0 is
the solid, validated win.

## Phase 2 — remaining work

1. **C loader** `mtp_load()` — parse `out-mtp.safetensors` into 8 `Layer`
   structs (the existing struct already has every field) + `input_proj`,
   `embed_norm`, `hidden_norm` per module. Resident bf16/f32; ~10.5 GB.
2. **C draft forward** `mtp_draft()` — the chain above, reusing `attention()`,
   `dense_mlp()`, `sconv_apply()`, `rmsnorm_row()`.
3. **MTP KV absorb/rollback** (the intricate core) — each module keeps its own
   attention KV + conv state across draft rounds. On acceptance, absorb the
   verified tokens into the MTP caches; on partial rejection, roll back (the
   Phase-1 `vck`/conv-rollback pattern generalizes; KV is position-indexed).
4. **Wire into `generate_spec`/`serve_one`** behind `DRAFT`/`MTP` env (mirror
   glm.c's auto-resolve: DRAFT=3 when a head is present).
5. **Torch reference** `tools/make_tiny_mtp.py` — fabricate tiny MTP weights,
   compute the draft via `InklingDecoderLayer`, save `ref_mtp.json`; C must match
   (catches name-mapping / fused-row / arithmetic bugs).
6. **Acceptance measurement** on the real 975B head — the recipe's ground truth.
   High acceptance (→ 2–3×) confirms the recipe; low means a subtle recipe bug,
   but output stays correct throughout.

## Numerical consistency (watch item)

glm.c needed `SPEC_PIN` so the draft (S=1) and verify (S≥2) forwards use matching
FP accumulation order — near-tie argmax flips otherwise collapse acceptance.
Inkling's matmuls/attention/convs are per-position independent, so the risk is
lower, but confirm draft-vs-verify consistency once the MTP drafter is in.
