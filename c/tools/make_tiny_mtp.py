#!/usr/bin/env python3
"""Build a tiny Inkling MTP-head fixture for inkling.c's MTP draft forward.

transformers drops the MTP head (`_keys_to_ignore_on_load_unexpected`), so there
is no reference from InklingForCausalLM. We fabricate random MTP module weights
in the real `model.mtp.{k}.*` tensor layout, then compute a *teacher-forced*
reference for MTP module 0 by driving transformers' own InklingDecoderLayer — so
the reference shares the block math but pins the C engine's name-mapping, fused
rows, input_proj fusion, and shared-head arithmetic.

Module 0 predicts t_{i+2} from [hidden_norm(h_i) ; embed_norm(emb(t_{i+1}))] where
h_i is the main model's last-layer (pre-final-norm) hidden. t_{i+1} is always a
committed token, so the teacher-forced check has no circular dependency.

Usage: python3 make_tiny_mtp.py <tiny_model_dir> [outdir]
Then:  SNAP=<tiny_model_dir> MTP=<outdir> ./inkling --mtp-oracle <outdir>/ref_mtp.json
"""
import copy
import json
import os
import sys

import torch
from safetensors.torch import save_file

from transformers import InklingForCausalLM, InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer, InklingRMSNorm

STD = 0.05


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "tiny_inkling"
    out = sys.argv[2] if len(sys.argv) > 2 else "tiny_mtp"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(1)

    model = InklingForCausalLM.from_pretrained(src, torch_dtype=torch.float32).eval()
    base = model.config.get_text_config()
    D = base.hidden_size

    NM = 2  # tiny MTP modules
    # MTP block config: dense MLP, global ("hybrid") attention, one layer.
    bcfg = copy.deepcopy(base)
    bcfg.num_hidden_layers = 1
    bcfg.layer_types = ["hybrid"]
    bcfg.mlp_layer_types = ["dense"]
    bcfg.intermediate_size = base.intermediate_size            # dense width

    prompt = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245, 30, 61, 5, 200]
    ids = torch.tensor([prompt], dtype=torch.long)

    with torch.no_grad():
        # MTP module 0's input is the main model's PRE-final-norm hidden (each
        # module re-norms it via its own hidden_norm). Extract it unambiguously by
        # swapping the final norm to identity — matches the C engine's residual
        # stream `x` after the last layer, before final_norm.
        final_norm = model.model.norm
        model.model.norm = torch.nn.Identity()
        h_pre = model.model(ids).last_hidden_state   # [1,S,D], pre-final-norm
        model.model.norm = final_norm
        embed = model.get_input_embeddings()
        lm_head = model.lm_head
        mup = base.logits_mup_width_multiplier

        modules, tensors = [], {}
        h_chain = h_pre
        mod0_pred = None
        for k in range(NM):
            blk = InklingDecoderLayer(bcfg, 0).eval().float()
            for p in blk.parameters():
                p.data.normal_(0, STD)
            in_proj = torch.nn.Linear(2 * D, D, bias=False); in_proj.weight.data.normal_(0, STD)
            e_norm = InklingRMSNorm(D, base.rms_norm_eps); e_norm.weight.data.normal_(1, STD)
            h_norm = InklingRMSNorm(D, base.rms_norm_eps); h_norm.weight.data.normal_(1, STD)

            S = len(prompt)
            # module k teacher-forced over positions 0..S-2 (needs t_{i+1})
            toks = ids[:, 1:]                                  # t_{i+1}, i=0..S-2
            emb = e_norm(embed(toks))                          # [1,S-1,D]
            hn = h_norm(h_chain[:, :S-1, :])                   # [1,S-1,D]
            cat = torch.cat([hn, emb], dim=-1)                 # hidden first
            x = in_proj(cat)
            mask = torch.ones(1, 1, S-1, S-1, dtype=torch.bool).tril()   # True = attend (causal)
            hp = blk(x, attention_mask=mask)                  # [1,S-1,D]
            logits = lm_head(final_norm(hp) / mup)
            pred = logits.argmax(-1)[0].tolist()
            if k == 0:
                mod0_pred = pred
            h_chain = torch.nn.functional.pad(hp, (0, 0, 1, 0))  # shift for next module's h

            # collect weights under model.mtp.k.*
            pref = f"model.mtp.{k}."
            sd = blk.state_dict()
            name_map = {
                "self_attn.q_proj.weight": "self_attn.q_proj.weight",
                "self_attn.k_proj.weight": "self_attn.k_proj.weight",
                "self_attn.v_proj.weight": "self_attn.v_proj.weight",
                "self_attn.r_proj.weight": "self_attn.r_proj.weight",
                "self_attn.o_proj.weight": "self_attn.o_proj.weight",
                "self_attn.q_norm.weight": "self_attn.q_norm.weight",
                "self_attn.k_norm.weight": "self_attn.k_norm.weight",
                "self_attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
                "self_attn.k_sconv.conv1d.weight": "self_attn.k_sconv.conv1d.weight",
                "self_attn.v_sconv.conv1d.weight": "self_attn.v_sconv.conv1d.weight",
                "attn_sconv.conv1d.weight": "attn_sconv.conv1d.weight",
                "mlp_sconv.conv1d.weight": "mlp_sconv.conv1d.weight",
                "input_layernorm.weight": "input_layernorm.weight",
                "post_attention_layernorm.weight": "post_attention_layernorm.weight",
                "mlp.gate_proj.weight": "mlp.gate_proj.weight",
                "mlp.up_proj.weight": "mlp.up_proj.weight",
                "mlp.down_proj.weight": "mlp.down_proj.weight",
            }
            for a, b in name_map.items():
                if a in sd:
                    tensors[pref + b] = sd[a].clone().contiguous()
            tensors[pref + "input_proj.weight"] = in_proj.weight.clone().contiguous()
            tensors[pref + "embed_norm.weight"] = e_norm.weight.clone().contiguous()
            tensors[pref + "hidden_norm.weight"] = h_norm.weight.clone().contiguous()
            if "mlp.global_scale" in sd:
                tensors[pref + "mlp.global_scale"] = sd["mlp.global_scale"].clone().contiguous()

    save_file(tensors, f"{out}/out-mtp.safetensors")
    ref = {"prompt_ids": prompt, "num_mtp_layers": NM, "mtp0_pred": mod0_pred}
    with open(f"{out}/ref_mtp.json", "w") as f:
        json.dump(ref, f)
    print(f"saved {len(tensors)} MTP tensors + ref_mtp.json to {out}/  (NM={NM})")
    print("module-0 teacher-forced preds:", mod0_pred)


if __name__ == "__main__":
    main()
