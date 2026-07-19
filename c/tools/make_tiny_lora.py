#!/usr/bin/env python3
"""Build a random Tinker-format LoRA adapter for the tiny Inkling oracle,
plus a reference fixture the C engine must reproduce with the adapter loaded.

Mirrors the make_tiny_inkling.py flow: the adapter is saved in the exact
tensor naming/fusion structure of a real Tinker checkpoint (shared lora_A on
expert w1/w3, shared lora_B on expert w2, fused dense gate_up, fused shared
experts, lm_head), and the reference tokens come from HF transformers with
every delta merged into the weights. This validates lora.h's two application
paths in one run:

  - resident merge (attn / dense / shared experts / lm_head): the engine
    merges at load, the reference merges here — identical math, and any
    name-mapping or fused-row-order mistake diverges immediately;
  - routed experts: the engine applies the low-rank correction at runtime,
    the reference merges. In f32 (bits=0) these differ only in last-ulp
    rounding order, the same tolerance the base oracle already absorbs
    between torch GEMM and the C dot loops.

Both lora_A and lora_B are random (a zero-init B would make every delta zero
and validate nothing), and alpha != r so the scale factor is exercised.

Usage: python3 make_tiny_lora.py <tiny_model_dir> <adapter_outdir>
Then:  SNAP=<tiny_model_dir> LORA=<adapter_outdir> ./inkling 8 0 <adapter_outdir>/ref_inkling_lora.json
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

try:
    from transformers import InklingForCausalLM
except ImportError:
    sys.exit("transformers has no Inkling support: pip install -U transformers")

R, ALPHA, STD = 4, 8, 0.05


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out = sys.argv[1], sys.argv[2]
    torch.manual_seed(1)

    model = InklingForCausalLM.from_pretrained(model_dir).eval().float()
    cfg = model.config
    sd = model.state_dict()
    D = cfg.hidden_size
    I = cfg.moe_intermediate_size if hasattr(cfg, "moe_intermediate_size") else cfg.intermediate_size
    ns = cfg.n_shared_experts
    E = cfg.n_routed_experts
    scale = ALPHA / R

    def rnd(*shape):
        return (torch.randn(*shape) * STD).float()

    def find(suffix):
        keys = [k for k in sd if k.endswith(suffix)]
        return keys

    adapter = {}   # Tinker-format tensors
    merged = 0

    def add(name, a, b):
        adapter[f"{name}.lora_A.weight"] = a.contiguous()
        adapter[f"{name}.lora_B.weight"] = b.contiguous()

    # attention + dense mlp + shared experts, layer by layer, driven by the
    # engine-visible tensor names so shapes always match the checkpoint
    attn_map = {"q_proj": "wq_du", "k_proj": "wk_dv", "v_proj": "wv_dv",
                "r_proj": "wr_du", "o_proj": "wo_ud"}
    torch.set_grad_enabled(False)   # merges are in-place edits of leaf params
    for li in range(cfg.num_hidden_layers):
        pfx = f"model.layers.{li}."
        tk = f"language_model.layers.{li}"
        for eng, tnk in attn_map.items():
            w = sd[f"{pfx}self_attn.{eng}.weight"]
            a, b = rnd(R, w.shape[1]), rnd(w.shape[0], R)
            add(f"{tk}.attn.{tnk}", a, b)
            w += scale * (b @ a)
            merged += 1
        if f"{pfx}mlp.gate_proj.weight" in sd:
            # dense layer: Tinker fuses gate_up, rows [gate; up]
            g, u = sd[f"{pfx}mlp.gate_proj.weight"], sd[f"{pfx}mlp.up_proj.weight"]
            dI = g.shape[0]
            a, b = rnd(R, D), rnd(2 * dI, R)
            add(f"{tk}.mlp.gate_up_proj", a, b)
            g += scale * (b[:dI] @ a)
            u += scale * (b[dI:] @ a)
            d = sd[f"{pfx}mlp.down_proj.weight"]
            a, b = rnd(R, dI), rnd(D, R)
            add(f"{tk}.mlp.down_proj", a, b)
            d += scale * (b @ a)
            merged += 3
        else:    # sparse layer: shared experts (fused over ns) + routed experts
            shg = sd[[k for k in find("mlp.shared_experts.gate_proj") if k.startswith(pfx)][0]]
            shu = sd[[k for k in find("mlp.shared_experts.up_proj") if k.startswith(pfx)][0]]
            shd = sd[[k for k in find("mlp.shared_experts.down_proj") if k.startswith(pfx)][0]]
            # w1/w3: fused [ns*I, D] view, row-concatenated over shared experts
            for nm, w in (("w1", shg), ("w3", shu)):
                a, b = rnd(R, D), rnd(ns * I, R)
                add(f"{tk}.mlp.shared_experts.{nm}", a, b)
                w.view(ns * I, D)[:] += scale * (b @ a)
            # w2: fused [D, ns*I] view — engine keeps ns consecutive [D, I]
            # blocks, so block j takes A columns [j*I, (j+1)*I)
            a, b = rnd(R, ns * I), rnd(D, R)
            add(f"{tk}.mlp.shared_experts.w2", a, b)
            v = shd.view(ns, D, I)
            for j in range(ns):
                v[j] += scale * (b @ a[:, j * I:(j + 1) * I])
            # routed experts: shared A on w1/w3, shared B on w2 (Tinker layout)
            gup = sd[[k for k in find("mlp.experts.gate_up_proj") if k.startswith(pfx)][0]]   # [E, 2I, D]
            dwn = sd[[k for k in find("mlp.experts.down_proj") if k.startswith(pfx)][0]]      # [E, D, I]
            a1, b1 = rnd(1, R, D), rnd(E, I, R)
            a3, b3 = rnd(1, R, D), rnd(E, I, R)
            a2, b2 = rnd(E, R, I), rnd(1, D, R)
            add(f"{tk}.mlp.experts.w1", a1, b1)
            add(f"{tk}.mlp.experts.w3", a3, b3)
            add(f"{tk}.mlp.experts.w2", a2, b2)
            gup.view(E, 2 * I, D)[:, :I, :] += scale * (b1 @ a1[0])
            gup.view(E, 2 * I, D)[:, I:, :] += scale * (b3 @ a3[0])
            dwn.view(E, D, I)[:] += scale * (b2[0] @ a2)
            merged += 6

    lh = sd["lm_head.weight"]
    a, b = rnd(R, D), rnd(lh.shape[0], R)
    add("language_model.lm_head", a, b)
    lh += scale * (b @ a)
    merged += 1

    os.makedirs(out, exist_ok=True)
    save_file(adapter, f"{out}/adapter_model.safetensors")
    with open(f"{out}/adapter_config.json", "w") as f:
        json.dump({"peft_type": "LORA", "r": R, "lora_alpha": ALPHA,
                   "target_modules": "all-linear", "task_type": "CAUSAL_LM"}, f)

    model.load_state_dict(sd)
    prompt = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245]
    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=24, do_sample=False, use_cache=True)
        full = gen[0].tolist()
        logits = model(torch.tensor([full], dtype=torch.long)).logits[0]
        tf = logits.argmax(-1).tolist()
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        nxt = torch.tensor(full[1:], dtype=torch.long)
        ppl = float(torch.exp(-logp.gather(1, nxt[:, None]).mean()))

    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf, "ppl_ref": ppl}
    with open(f"{out}/ref_inkling_lora.json", "w") as f:
        json.dump(ref, f)
    print(f"saved adapter ({len(adapter)} tensors, {merged} modules merged) + "
          f"ref_inkling_lora.json to {out}/  (transformers ppl={ppl:.4f})")


if __name__ == "__main__":
    main()
