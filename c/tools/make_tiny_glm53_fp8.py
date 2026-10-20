#!/usr/bin/env python3
"""Make an FP8 checkpoint-shaped tiny GLM-5.3 and its exact f32 twin.

  python3 make_tiny_glm53_fp8.py <tiny_glm53> <out_fp8> <out_twin> [<out_twin_i4>]

The real checkpoint stores most matrices as FP8 e4m3 with per-block `_scale_inv`
scales, and convert_glm53_int4.py is the only code that reads that. It can't be run on
the 328 GB original here, so this builds the same shape from the tiny snapshot:

  out_fp8   every matrix the real checkpoint quantizes -> F8_E4M3 + <name>_scale_inv,
            16x16 blocks (several blocks per tiny matrix, so block indexing is live);
            quantization_config written into config.json like the real one
  out_twin  f32, holding EXACTLY the values out_fp8 encodes, plus a fresh
            ref_glm53.json computed by transformers on those weights

Scales are powers of two and every tensor the converter stores bf16 is pre-rounded to
bf16, so `convert --xbits 0` of out_fp8 is lossless: the engine on that container must
pass the twin's transformers oracle exactly as the twin itself does. That is the check
on the converter (names, FP8 block dequant, expert fusion, bf16/f32 policy).

  out_twin_i4  (optional) the twin with every routed expert replaced by its int4
            round-trip (the converter's own quant_int4_rows, dequantized). The int4
            container must then agree with the engine's f32 path on this twin — the check
            on the int4 plumbing (row order, nibble order, per-row scales), separate from
            how much int4 costs, which is measured against the transformers reference.
"""
import json
import math
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_glm53_int4 import F32_RE, quant_int4_rows  # noqa: E402  (the converter's own policy)

BLOCK = 16
FP8_RE = ("_proj.weight", ".wq_b.weight", ".wk.weight", "index_kpool_compress_gate")


def is_fp8(name, t):
    return (t.dim() == 2 and name.startswith("model.language_model.layers.") and
            not F32_RE.search(name) and name.endswith(FP8_RE))


def fp8_blocks(w):
    O, I = w.shape
    bo, bi = math.ceil(O / BLOCK), math.ceil(I / BLOCK)
    sc = torch.empty(bo, bi)
    q = torch.empty(O, I, dtype=torch.float8_e4m3fn)
    for a in range(bo):
        for b in range(bi):
            blk = w[a * BLOCK:(a + 1) * BLOCK, b * BLOCK:(b + 1) * BLOCK]
            amax = float(blk.abs().max()) or 1.0
            s = 2.0 ** math.ceil(math.log2(amax / 448.0))      # power of two: exact in bf16
            sc[a, b] = s
            q[a * BLOCK:(a + 1) * BLOCK, b * BLOCK:(b + 1) * BLOCK] = (blk / s).to(torch.float8_e4m3fn)
    deq = q.float() * sc.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:O, :I]
    return q, sc, deq


def main():
    src, out_fp8, out_twin = sys.argv[1:4]
    for d in (out_fp8, out_twin):
        os.makedirs(d, exist_ok=True)
    sd = load_file(os.path.join(src, "model.safetensors"))
    fp8, twin = {}, {}
    for n, t in sd.items():
        t = t.float()
        if ".visual." in n:
            fp8[n] = twin[n] = t
        elif is_fp8(n, t):
            q, sc, deq = fp8_blocks(t)
            fp8[n], fp8[n + "_scale_inv"], twin[n] = q, sc, deq
        elif F32_RE.search(n):
            fp8[n] = twin[n] = t
        else:                                   # stored bf16 by the converter
            r = t.to(torch.bfloat16).float()
            fp8[n] = r.to(torch.bfloat16)
            twin[n] = r
    save_file(fp8, os.path.join(out_fp8, "model.safetensors"))
    save_file(twin, os.path.join(out_twin, "model.safetensors"))
    cfg = json.load(open(os.path.join(src, "config.json")))
    for d in (out_fp8, out_twin):
        c = dict(cfg)
        if d == out_fp8:
            c["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                        "activation_scheme": "dynamic", "weight_block_size": [BLOCK, BLOCK]}
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=2)
        for fn in ("generation_config.json", "tokenizer.json"):
            if os.path.exists(os.path.join(src, fn)):
                shutil.copy(os.path.join(src, fn), d)

    # transformers' own answer on the twin's weights
    from transformers import Glm5NextForConditionalGeneration
    ref = json.load(open(os.path.join(src, "ref_glm53.json")))
    model = Glm5NextForConditionalGeneration.from_pretrained(out_twin, torch_dtype=torch.float32).eval()
    prompt = ref["prompt_ids"]
    n_new = len(ref["full_ids"]) - len(prompt)
    with torch.no_grad():
        full = model.generate(torch.tensor([prompt]), max_new_tokens=n_new, do_sample=False)[0].tolist()
        logits = model(torch.tensor([full])).logits[0]
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        ppl = float(torch.exp(-logp.gather(1, torch.tensor(full[1:])[:, None]).mean()))
    new = {"prompt_ids": prompt, "full_ids": full, "tf_pred": logits.argmax(-1).tolist(),
           "ppl_ref": ppl, "logits_last": logits[-1].tolist()}
    for d in (out_fp8, out_twin):
        json.dump(new, open(os.path.join(d, "ref_glm53.json"), "w"))
    if len(sys.argv) > 4:
        out_i4 = sys.argv[4]
        os.makedirs(out_i4, exist_ok=True)
        tw = dict(twin)
        for n, t in tw.items():
            if ".mlp.experts." in n:
                q, sc = quant_int4_rows(t.numpy())
                lo = (q & 0x0F).astype("int32") - 8
                hi = (q >> 4).astype("int32") - 8
                deq = torch.empty_like(t)
                deq[:, 0::2] = torch.from_numpy(lo).float()
                deq[:, 1::2] = torch.from_numpy(hi).float()
                tw[n] = deq * torch.from_numpy(sc)[:, None]
        save_file(tw, os.path.join(out_i4, "model.safetensors"))
        for fn in ("config.json", "generation_config.json", "tokenizer.json", "ref_glm53.json"):
            if os.path.exists(os.path.join(out_twin, fn)):
                shutil.copy(os.path.join(out_twin, fn), out_i4)
    nq = sum(1 for n in fp8 if n.endswith("_scale_inv"))
    print(f"{nq} FP8 tensors ({BLOCK}x{BLOCK} blocks) -> {out_fp8}; exact f32 twin + transformers ref -> "
          f"{out_twin} (ppl {ppl:.4f})")


if __name__ == "__main__":
    main()
