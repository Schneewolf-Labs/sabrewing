#!/usr/bin/env python3
"""Build a tiny random-weight Laguna text model + oracle fixture for laguna.c.

Stage-A validation, mirroring the Inkling ref.json flow: saves a small
LagunaForCausalLM snapshot (safetensors + config.json) and a ref_laguna.json
with {prompt_ids, full_ids, tf_pred, ppl_ref} from HF transformers, which the C
engine must reproduce token-for-token (run with bits=0 for f32-exact experts).

The tiny config exercises every architectural branch of the big model:
- interleaved global (full_attention) + sliding_attention layers,
- per-layer query-head asymmetry (fewer heads on global, more on sliding),
- a dense layer 0 (mlp_only_layers) + sigmoid-routed MoE layers,
- YaRN partial-rotary RoPE on global layers, plain full RoPE on sliding,
- per-head softplus attention output gate, QK-RMSNorm,
- a shared expert + routed_scaling_factor.

Needs transformers >= 5.7 (native Laguna). Usage: python3 make_tiny_laguna.py <outdir>
"""
import json
import sys

import torch

try:
    from transformers import LagunaForCausalLM, LagunaConfig
except ImportError:
    sys.exit("transformers has no Laguna support: pip install -U 'transformers>=5.7'")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_laguna"
    torch.manual_seed(0)

    # period-4 [full, sliding, sliding, sliding] like the real 1:3 layout,
    # so layers 0 and 4 are global; layer 0 is also the lone dense layer.
    layer_types = ["full_attention", "sliding_attention",
                   "sliding_attention", "sliding_attention"] * 2
    heads_per_layer = [4 if t == "full_attention" else 6 for t in layer_types]

    cfg = LagunaConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,               # global-layer head count
        num_attention_heads_per_layer=heads_per_layer,   # 4 global / 6 sliding
        num_key_value_heads=2,
        head_dim=16,
        gating="per-head",
        hidden_act="silu",
        intermediate_size=96,                # dense layer 0
        sliding_window=8,                    # < seq len, so sliding is exercised
        layer_types=layer_types,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
        num_experts=8,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        # RoPE exactly as the real config: YaRN partial-0.5 on global layers,
        # plain full-rotary theta=10000 on sliding layers.
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        eos_token_id=None,
    )

    model = LagunaForCausalLM(cfg).eval().float()

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
    with open(f"{out}/ref_laguna.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_laguna.json to {out}/  (transformers ppl={ppl:.4f})")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
