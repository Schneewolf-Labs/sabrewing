#!/usr/bin/env python3
"""Convert the GLM-5.3-Flash checkpoint (zai-org/GLM-5.3-Flash) into a container
glm53.c reads: bf16 residents + int4 routed experts, one file per layer.

The checkpoint ships FP8 e4m3 weights with 128x128 block scales (`<name>_scale_inv`,
block size from quantization_config.weight_block_size) for most matrices. Every FP8
tensor is dequantized here; nothing FP8 reaches the engine.

  routed experts (97% of the bytes)  per-expert gate/up/down -> fused per layer:
      <P>layers.{i}.mlp.experts.gate_up_proj  int4 u8 [E*2I, D/2] (+ .qs f32 [E*2I])
      <P>layers.{i}.mlp.experts.down_proj     int4 u8 [E*D,  I/2] (+ .qs f32 [E*D])
    gate_up is block-concatenated per expert (gate rows, then up rows), which is
    F.linear(x, gate_up_proj[e]).chunk(2); --xbits 0 keeps them f32 instead.
  small or numerically sensitive -> f32: norms, mHC (hc_*), KDA dt_bias / A_log /
    convs, the router and its correction bias, the indexer's ape and k-norm bias.
  everything else -> bf16 (the dense set is ~9.7B params, ~18 GB).

Names stay the checkpoint's own (`model.language_model.` prefix, per-expert originals
replaced by the fused int4 pair). The vision tower and the MTP layer (layer index
num_hidden_layers) are dropped: glm53.c is text-only and does not draft.

Usage:
  python3 convert_glm53_int4.py --indir <hf snapshot> --outdir <container> [--xbits 4|0]
  python3 convert_glm53_int4.py --selftest
"""
import argparse
import json
import os
import re
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

P = "model.language_model."
F32_RE = re.compile(r"(norm\.weight$|norm\.bias$|\.hc_(attn|ffn)_(fn|base|scale)$|dt_bias$|A_log$|"
                    r"conv1d\.weight$|\.mlp\.gate\.weight$|e_score_correction_bias$|"
                    r"index_kpool_compress_ape$)")


def quant_int4_rows(w):
    """w: f32 [O,I], I even -> (packed u8 [O,I/2], f32 scales [O]); low nibble = even
    column, high nibble = odd column, offset +8 (laguna's container, moe_quant.h)."""
    O, I = w.shape
    assert I % 2 == 0, f"row width {I} must be even for int4"
    s = np.abs(w).max(axis=1, keepdims=True) / 7.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (lo | (hi << 4)), s[:, 0].astype(np.float32)


def shard_map(indir):
    idx = os.path.join(indir, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {n: os.path.join(indir, s) for n, s in json.load(open(idx))["weight_map"].items()}
    single = os.path.join(indir, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {n: single for n in f.keys()}


class Pool:
    """Lazily opened shard handles; get() returns f32, dequantizing FP8 blocks."""
    def __init__(self, smap, block):
        self.smap, self.block, self.h = smap, block, {}

    def raw(self, name):
        p = self.smap[name]
        if p not in self.h:
            self.h[p] = safe_open(p, framework="pt")
        return self.h[p].get_tensor(name)

    def get(self, name):
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            sc = self.raw(name + "_scale_inv").float()
            bo, bi = self.block
            O, I = t.shape
            sc = sc.repeat_interleave(bo, 0).repeat_interleave(bi, 1)[:O, :I]
            return t.float() * sc
        return t.float()

    def close(self):
        self.h.clear()


def convert(indir, outdir, xbits):
    os.makedirs(outdir, exist_ok=True)
    root = json.load(open(os.path.join(indir, "config.json")))
    cfg = root.get("text_config", root)
    L = cfg["num_hidden_layers"]
    E, I, D = cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"]
    qc = root.get("quantization_config") or cfg.get("quantization_config") or {}
    block = tuple(qc.get("weight_block_size", (128, 128)))
    smap = shard_map(indir)
    pool = Pool(smap, block)
    names = [n for n in smap if n.startswith(P) and not n.endswith("_scale_inv")]
    if not names:
        raise SystemExit(f"no {P}* tensors in {indir}")

    def put(out, n):
        w = pool.get(n)
        out[n] = w.contiguous() if F32_RE.search(n) else w.to(torch.bfloat16).contiguous()

    # top-level: embedding, final norm, lm_head
    out = {}
    for n in (P + "embed_tokens.weight", P + "norm.weight", "lm_head.weight"):
        if n in smap:
            put(out, n)
    save_file(out, os.path.join(outdir, "out-top.safetensors"))
    print("out-top.safetensors", flush=True)

    expert_re = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    for li in range(L):
        t0 = time.time()
        pre = f"{P}layers.{li}."
        mine = [n for n in names if n.startswith(pre)]
        out = {}
        experts = [n for n in mine if expert_re.search(n)]
        for n in mine:
            if n not in experts:
                put(out, n)
        if experts:
            ep = f"{pre}mlp.experts"
            if xbits == 4:
                gu_q = np.empty((E * 2 * I, D // 2), np.uint8); gu_s = np.empty(E * 2 * I, np.float32)
                dn_q = np.empty((E * D, I // 2), np.uint8);     dn_s = np.empty(E * D, np.float32)
            else:
                gu = np.empty((E, 2 * I, D), np.float32); dn = np.empty((E, D, I), np.float32)
            for e in range(E):
                g = pool.get(f"{ep}.{e}.gate_proj.weight").numpy()
                u = pool.get(f"{ep}.{e}.up_proj.weight").numpy()
                d = pool.get(f"{ep}.{e}.down_proj.weight").numpy()
                if xbits == 4:
                    q, s = quant_int4_rows(np.concatenate([g, u], 0))
                    gu_q[e * 2 * I:(e + 1) * 2 * I], gu_s[e * 2 * I:(e + 1) * 2 * I] = q, s
                    q, s = quant_int4_rows(d)
                    dn_q[e * D:(e + 1) * D], dn_s[e * D:(e + 1) * D] = q, s
                else:
                    gu[e] = np.concatenate([g, u], 0); dn[e] = d
            if xbits == 4:
                out[f"{ep}.gate_up_proj"] = torch.from_numpy(gu_q.reshape(-1))
                out[f"{ep}.gate_up_proj.qs"] = torch.from_numpy(gu_s)
                out[f"{ep}.down_proj"] = torch.from_numpy(dn_q.reshape(-1))
                out[f"{ep}.down_proj.qs"] = torch.from_numpy(dn_s)
            else:
                out[f"{ep}.gate_up_proj"] = torch.from_numpy(gu)
                out[f"{ep}.down_proj"] = torch.from_numpy(dn)
        dst = os.path.join(outdir, f"out-layer-{li:03d}.safetensors")
        save_file(out, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        pool.close()
        print(f"out-layer-{li:03d}.safetensors ({os.path.getsize(dst) / 1e9:.2f} GB) in {time.time() - t0:.0f}s",
              flush=True)

    for fn in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(indir, fn)):
            shutil.copy(os.path.join(indir, fn), outdir)
    print(f"conversion complete -> {outdir}")


def selftest():
    w = np.random.randn(16, 32).astype(np.float32)
    q, s = quant_int4_rows(w)
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = (q >> 4).astype(np.int32) - 8
    deq = np.empty_like(w); deq[:, 0::2] = lo; deq[:, 1::2] = hi; deq *= s[:, None]
    assert np.array_equal(deq, np.clip(np.rint(w / s[:, None]), -8, 7) * s[:, None])
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--xbits", type=int, default=4, choices=[0, 4])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir required")
    convert(a.indir, a.outdir, a.xbits)


if __name__ == "__main__":
    main()
