#!/usr/bin/env python3
"""Build a tiny random-weight DeepSeek-V4 model + oracle fixture for deepseek.c.

Same flow as make_tiny_laguna.py / make_tiny_qwen35.py: save a small
DeepseekV4ForCausalLM snapshot plus a ref_deepseek.json with
{prompt_ids, full_ids, tf_pred, ppl_ref} that the C engine must reproduce token-for-token.

This is the REAL gate for deepseek.c. tools/dsv4_ref.py is a dependency-free stand-in that
catches transcription bugs anywhere, but only this script can prove agreement with
transformers — run it on a box that has torch and a transformers carrying `deepseek_v4`.

The tiny config exercises every architectural branch of V4-Flash:
- all three attention layer types in one stack: sliding_attention (plain RoPE, window only),
  heavily_compressed_attention (m'=4 here, whole compressed series visible per causality),
  and compressed_sparse_attention (m=2, Lightning Indexer top-k over the compressed series),
- the CSA two-series Ca|Cb overlap windowing, including window 0's missing predecessor,
- shared-KV MQA (one KV head broadcast to every query head, K == V) with per-head sinks and
  the conjugate output rotation that undoes the value's RoPE,
- the grouped low-rank output projection (2 groups -> 8 each -> mixed to hidden),
- mHC hyper-connections at both sublayer sites, with a real 20-iteration Sinkhorn,
- BOTH MoE router kinds, interleaved so the layer indexing is live: hash_moe layers routed
  by the frozen tid2eid table, and top-k layers routed by sqrt(softplus(logits)) with an
  e_score_correction_bias that only affects selection,
- YaRN on the compress rope (factor 4) alongside plain RoPE on the main rope, so the two
  rope tables cannot be conflated.

The prompt length is deliberately coprime with both compress rates, so both compressors
carry a partial window across the prefill -> decode boundary.

Usage: python3 tools/make_tiny_deepseek.py <outdir>
"""
import json
import sys

import torch

try:
    from transformers import DeepseekV4ForCausalLM
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
except ImportError:
    sys.exit("transformers has no DeepSeek-V4 support: pip install -U transformers")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_deepseek"
    torch.manual_seed(0)

    V, D, HD, HEADS = 64, 24, 16, 2
    E, K, MOE_I = 6, 2, 16

    cfg = DeepseekV4Config(
        vocab_size=V,
        hidden_size=D,
        num_hidden_layers=4,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HD,
        q_lora_rank=16,
        partial_rotary_factor=4 / HD,
        moe_intermediate_size=MOE_I,
        n_routed_experts=E,
        n_shared_experts=1,
        num_experts_per_tok=K,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        # all three attention types, and both MoE router kinds, interleaved
        layer_types=["heavily_compressed_attention", "compressed_sparse_attention",
                     "sliding_attention", "compressed_sparse_attention"],
        mlp_layer_types=["hash_moe", "moe", "hash_moe", "moe"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        sliding_window=6,
        o_groups=2,
        o_lora_rank=8,
        # 8, not 2. The indexer score is sum_h w_h * relu(q_h . k_e); with only 2 heads a
        # random model relus BOTH to zero for most (query, entry) pairs, so whole score
        # rows tie at exactly 0.0 and the fixture ends up measuring topk's tie-break
        # rather than the architecture. torch.topk does not define one (it returned the
        # HIGHEST tied index here, a stable index-ascending scan returns the lowest), so
        # that comparison is unwinnable and meaningless. Real V4-Flash has 64 heads,
        # where an exact all-head tie is a 2^-64 event. See docs/deepseek.md.
        index_n_heads=8,
        index_head_dim=8,
        index_topk=2,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1.0e-6,
        rms_norm_eps=1.0e-6,
        hidden_act="silu",
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 4 / HD},
            "compress": {"rope_type": "yarn", "rope_theta": 160000.0, "factor": 4.0,
                         "beta_fast": 32.0, "beta_slow": 1.0,
                         "original_max_position_embeddings": 16,
                         "partial_rotary_factor": 4 / HD},
        },
        eos_token_id=None,
    )

    model = DeepseekV4ForCausalLM(cfg).eval().float()

    # Several V4 parameters are created with torch.empty (sinks, position_bias, the mHC
    # fn/base/scale). Force every one of them live, or an uninitialised tensor makes the
    # fixture non-reproducible. The mHC scale/base get a wider spread than the rest so the
    # sigmoids and the Sinkhorn projection actually vary across tokens instead of sitting
    # at their symmetric fixed point, where a transposed `comb` would be undetectable.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.numel() == 0:
                continue
            s = 0.8 if name.endswith(("_hc.base", "_hc.scale", "hc_base", "hc_scale")) else 0.12
            p.copy_(torch.randn_like(p) * s)
        for name, b in model.named_buffers():
            if name.endswith("tid2eid"):
                b.copy_(torch.randint(0, E, b.shape, dtype=b.dtype))
            elif name.endswith("e_score_correction_bias"):
                # a zeros bias would make the correction path indistinguishable from none
                b.copy_(torch.randn_like(b.float()).to(b.dtype) * 0.3)

    prompt = [7, 42, 19, 3, 58, 14, 21, 60, 9, 33, 47, 25, 11]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 8

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=n_new, do_sample=False, use_cache=True)
        full = gen[0].tolist()
        logits = model(torch.tensor([full], dtype=torch.long)).logits[0]
        tf = logits.argmax(-1).tolist()
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        nxt = torch.tensor(full[1:], dtype=torch.long)
        ppl = float(torch.exp(-logp.gather(1, nxt[:, None]).mean()))

    model.save_pretrained(out, safe_serialization=True)
    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf, "ppl_ref": ppl,
           "logits_last": logits[-1].tolist()}
    with open(f"{out}/ref_deepseek.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_deepseek.json to {out}/  (transformers ppl={ppl:.4f})")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
