#!/usr/bin/env python3
"""Build a tiny GLM-5.3-Flash (transformers `glm5_next`) snapshot plus its oracle.

The reference is transformers' own Glm5NextForConditionalGeneration (text path; the
vision tower is built tiny and never run), so the C engine is checked against an
implementation it shares no code with. The tiny config exercises every text-side
branch of the real model:

- KDA (Kimi Delta Attention) linear layers: per-CHANNEL decay with the safe lower
  bound, short convs, low-rank forget and output gates. HF runs prefill through the
  chunked WY form and decode through the recurrent one, so the engine (recurrent only)
  is checked against both.
- DSA layers: NoPE MLA (kv_lora latent) plus the pooled lightning indexer. The prompt
  is long enough that the indexer actually DROPS pools (more complete pools than
  index_topk / index_kpool), and its length is not a multiple of index_kpool so the
  visible-tail append is live. Two consecutive DSA layers, the second with
  indexer_types "shared", so cross-layer top-k reuse is covered.
- 8 indexer heads, not 2: the score is sum_h w_h relu(q_h . k_pool); with two heads a
  random model relus both to zero for many pools, rows tie, and the fixture would measure
  torch.topk's undefined tie-break instead of the architecture (docs/deepseek.md).
- mHC (hc_mult 4, 20 Sinkhorn iterations), the mean hc head, a dense first layer, and
  sigmoid-routed MoE with a correction bias and the clamped SwiGLU. swiglu_limit is 0.5,
  not the real 10: this model's pre-activations are ~0.6, so at 10 the clamp never fires
  and an engine without it passes (mutation-tested).

The torch-default init leaves several tensors degenerate (A_log zeros, index ape zeros,
compress gate ones, mHC base zeros / scale ones); all of them are re-drawn so a wrong
read of any of them changes the output.

Needs transformers >= 5.16 (glm5_next). Usage: python3 make_tiny_glm53.py <outdir>
"""
import json
import sys

import torch

try:
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
except ImportError:
    sys.exit("transformers has no glm5_next: pip install 'transformers>=5.16'")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_glm53"
    torch.manual_seed(0)

    D, V = 64, 256
    text = dict(
        vocab_size=V, hidden_size=D, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2,
        routed_scaling_factor=2.5, norm_topk_prob=True, n_group=1, topk_group=1,
        kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=0, qk_nope_head_dim=16, v_head_dim=16,
        layer_types=["linear_attention", "indexed_attention", "indexed_attention",
                     "linear_attention", "linear_attention", "indexed_attention"],
        indexer_types=["full", "full", "shared", "full", "full", "full"],
        mlp_layer_types=["dense"] + ["sparse"] * 5,
        index_topk=8, index_kpool=4, index_head_dim=16, index_n_heads=8,
        index_kpool_always_select_tail=True,
        linear_head_dim=16, linear_num_heads=4, linear_conv_kernel_dim=4, linear_lower_bound=-5.0,
        hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
        swiglu_limit=0.5, rms_norm_eps=1e-5, max_position_embeddings=4096,
        tie_word_embeddings=False, pad_token_id=None, eos_token_id=None,
    )
    vision = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=D,
                  projection_intermediate_size=64, patch_size=14, spatial_merge_size=2,
                  temporal_patch_size=2)
    cfg = Glm5NextConfig(text_config=text, vision_config=vision)
    model = Glm5NextForConditionalGeneration(cfg).eval().float()

    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".visual." in name:
                continue
            if name.endswith("A_log"):                       # per-head decay rate
                p.copy_(torch.rand_like(p) * 1.5 - 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif name.endswith("index_kpool_compress_ape"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif name.endswith("index_kpool_compress_gate"):
                p.copy_(torch.randn_like(p) * 0.2)
            elif name.endswith(("_hc.base",)):
                p.copy_(torch.randn_like(p) * 0.5)
            elif name.endswith(("_hc.scale",)):
                p.copy_(torch.rand_like(p) * 1.5 + 0.25)
            elif name.endswith(("_hc.fn",)):
                p.copy_(torch.randn_like(p) * 0.2)
            elif name.endswith("k_norm.bias"):
                p.copy_(torch.randn_like(p) * 0.8)    # a constant key shift only matters through the relu: make it big
            elif "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.1)
            elif name.endswith("conv1d.weight"):
                p.copy_(torch.randn_like(p) * 0.4)
            elif p.dim() >= 2:
                p.copy_(torch.randn_like(p) * 0.08)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):     # selection-only bias
                b.copy_(torch.randn_like(b.float()).to(b.dtype) * 0.3)

    # 21 tokens: 5 complete pools of 4 (> index_topk/kpool = 2, so the indexer drops
    # pools) and a 1-token tail; generation crosses two more pool boundaries.
    prompt = [7, 42, 19, 3, 58, 14, 21, 60, 9, 33, 47, 25, 11, 200, 131, 99, 5, 77, 163, 240, 18]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 10

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
    with open(f"{out}/ref_glm53.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_glm53.json to {out}/  (transformers ppl={ppl:.4f})")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
