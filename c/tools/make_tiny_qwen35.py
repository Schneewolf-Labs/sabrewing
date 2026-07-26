#!/usr/bin/env python3
"""Build a tiny random-weight Qwen3.5-MoE text model + oracle fixture for qwen35.c.

Same flow as make_tiny_laguna.py: save a small Qwen3_5MoeForCausalLM snapshot plus a
ref_qwen35.json with {prompt_ids, full_ids, tf_pred, ppl_ref} that the C engine must
reproduce token-for-token.

The tiny config exercises every architectural branch of the 35B:
- the 3:1 linear_attention / full_attention interleave (so both mixers, and the fact that
  only full-attention layers own a KV cache while linear layers own recurrent state),
- gated delta net: 2 key heads / 4 value heads (so the 2:1 value->key sharing is live),
  causal depthwise conv, gated RMSNorm,
- full attention with a *per-element* output gate (q_proj emits head_dim*2 per head,
  chunked per head -- query and gate interleaved, not block-concatenated),
- partial rotary (partial_rotary_factor 0.5 -> only half of head_dim rotates) with an
  mrope_section, which for text-only input must reduce to plain RoPE,
- QK-RMSNorm over head_dim, GQA 4 query heads / 2 KV heads,
- softmax -> top-k -> renormalize routing (no correction bias, no route scale) and a
  sigmoid-gated shared expert.

seq is chosen longer than conv_kernel_dim so the conv window is exercised past its fill.

Usage: python3 make_tiny_qwen35.py <outdir>
"""
import json
import sys

import torch

try:
    from transformers import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
except ImportError:
    sys.exit("transformers has no Qwen3.5-MoE support: pip install -U 'transformers>=5.3'")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_qwen35"
    torch.manual_seed(0)

    # period-4 [linear, linear, linear, full] like the real full_attention_interval=4
    layer_types = ["linear_attention", "linear_attention", "linear_attention",
                   "full_attention"] * 2

    cfg = Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=8,
        layer_types=layer_types,
        full_attention_interval=4,
        # full attention: GQA 4/2, head_dim > hidden/heads so it is read from config
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        attn_output_gate=True,
        # gated delta net: 4 value heads over 2 key heads (2:1, as in the 35B)
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        mlp_only_layers=[],
        hidden_act="silu",
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        # partial rotary, so the pass-through tail of each head is exercised.
        # mrope_section must sum to rotary_dim/2 = (32*0.5)/2 = 8.
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        eos_token_id=None,
    )

    model = Qwen3_5MoeForCausalLM(cfg).eval().float()
    # Default init leaves the fused expert parameters at zero in some versions, which would
    # make the MoE a no-op and hide routing bugs. Force every parameter to be live.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("A_log", "dt_bias")) or p.numel() == 0:
                continue
            p.copy_(torch.randn_like(p) * 0.12)

    prompt = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 24

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=n_new, do_sample=False, use_cache=True)
        full = gen[0].tolist()
        logits = model(torch.tensor([full], dtype=torch.long)).logits[0]
        tf = logits.argmax(-1).tolist()
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        nxt = torch.tensor(full[1:], dtype=torch.long)
        ppl = float(torch.exp(-logp.gather(1, nxt[:, None]).mean()))

    model.save_pretrained(out, safe_serialization=True)
    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf, "ppl_ref": ppl}
    with open(f"{out}/ref_qwen35.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_qwen35.json to {out}/  (transformers ppl={ppl:.4f})")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
