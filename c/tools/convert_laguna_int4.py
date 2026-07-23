#!/usr/bin/env python3
"""Convert the Poolside Laguna BF16 checkpoint into a colibri snapshot readable
by laguna.c.

Laguna already ships HF-native tensor names, so there is no name mapping and no
de-interleaving (unlike Inkling). The only restructuring is the routed experts:
the checkpoint stores them per-expert (``experts.{e}.gate_proj/up_proj/down_proj``,
possibly split across shards), and laguna.c wants them fused per layer:

  model.layers.{i}.mlp.experts.gate_up_proj  packed-int4 u8  [E*2I, D/2]  (+ .qs f32 [E*2I])
  model.layers.{i}.mlp.experts.down_proj     packed-int4 u8  [E*D,  I/2]  (+ .qs f32 [E*D])

gate_up is block-concatenated per expert (all gate rows, then all up rows), which
is exactly ``F.linear(x, gate_up_proj[e]).chunk(2)`` — no interleaving.

Output policy (Stage A), matching inkling's container:
  - routed experts (>95% of params) -> int4 (or --xbits 8) + `.qs` f32 row scales
  - norms, router weight, e_score_correction_bias -> f32
  - everything else (attn q/k/v/o/g, q/k_norm, dense layer-0 MLP, shared expert,
    embed, lm_head) -> bf16 passthrough (st_read_f32 converts on load)

Modes:
  --indir DIR --outdir DIR [--watch]   convert a local snapshot (index-driven,
        per-layer; --watch polls while `hf download` is still running)
  --selftest                           numpy round-trip unit test
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quant_int8_rows(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return q.view(np.uint8), s[:, 0].astype(np.float32)


def quant_int4_rows(w):
    """w: f32 [O,I], I even -> (packed u8 [O,I/2], f32 scales [O]).
    Low nibble = even column, high nibble = odd column, offset +8."""
    O, I = w.shape
    assert I % 2 == 0, f"row width {I} must be even for int4"
    s = np.abs(w).max(axis=1, keepdims=True) / 7.0
    s[s < 1e-8] = 1e-8
    q = np.clip(np.rint(w / s), -8, 7).astype(np.int32)
    lo = (q[:, 0::2] + 8).astype(np.uint8)
    hi = (q[:, 1::2] + 8).astype(np.uint8)
    return (lo | (hi << 4)), s[:, 0].astype(np.float32)


# names kept f32 in the container (small / numerically sensitive)
F32_RE = re.compile(r"(norm\.weight$|\.mlp\.gate\.weight$|e_score_correction_bias$)")


def build_shard_map(indir):
    """tensor name -> shard path (from the index, or the single-file snapshot)."""
    idx = os.path.join(indir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {n: os.path.join(indir, s) for n, s in wm.items()}
    single = os.path.join(indir, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, framework="pt") as f:
            return {n: single for n in f.keys()}
    raise SystemExit(f"no safetensors found in {indir}")


class ShardPool:
    """Lazily-opened safe_open handles, reused across a layer's tensor reads."""
    def __init__(self, smap):
        self.smap = smap
        self.h = {}

    def get(self, name):
        p = self.smap.get(name)
        if p is None:
            raise KeyError(name)
        if p not in self.h:
            self.h[p] = safe_open(p, framework="pt")
        return self.h[p].get_tensor(name)

    def has(self, name):
        return name in self.smap

    def close(self):
        self.h.clear()


def quant_experts(pool, layer, E, xbits, out):
    """Fuse per-expert gate/up -> [E*2I, D/2] int4, down -> [E*D, I/2] int4."""
    from concurrent.futures import ThreadPoolExecutor
    pre = f"model.layers.{layer}.mlp.experts"
    # discover shapes from expert 0
    g0 = pool.get(f"{pre}.0.gate_proj.weight")
    d0 = pool.get(f"{pre}.0.down_proj.weight")
    I, D = g0.shape                      # gate/up: [I, D]
    Dd, Id = d0.shape                    # down: [D, I]
    assert Dd == D and Id == I
    R_gu, C_gu = 2 * I, D
    if xbits == 4:
        gu_q = np.empty((E * R_gu, C_gu // 2), np.uint8)
        dn_q = np.empty((E * D, I // 2), np.uint8)
    else:
        gu_q = np.empty((E * R_gu, C_gu), np.uint8)
        dn_q = np.empty((E * D, I), np.uint8)
    gu_s = np.empty(E * R_gu, np.float32)
    dn_s = np.empty(E * D, np.float32)
    quant = quant_int4_rows if xbits == 4 else quant_int8_rows

    def work(e, g, u, d):
        gu = np.concatenate([g, u], axis=0)          # [2I, D] block-concat
        q, s = quant(gu); gu_q[e * R_gu:(e + 1) * R_gu] = q; gu_s[e * R_gu:(e + 1) * R_gu] = s
        q, s = quant(d);  dn_q[e * D:(e + 1) * D] = q;       dn_s[e * D:(e + 1) * D] = s

    # safetensors handles are NOT safe under concurrent get_tensor, and experts
    # of one layer may span shards — so READ serially on the main thread, and
    # fan the numpy quant (the CPU bottleneck, releases the GIL) out in waves of
    # 16 to keep peak RAM ~1 GB of pending f32 slices.
    with ThreadPoolExecutor(max_workers=8) as ex:
        for base in range(0, E, 16):
            futs = []
            for e in range(base, min(base + 16, E)):
                g = pool.get(f"{pre}.{e}.gate_proj.weight").float().numpy()
                u = pool.get(f"{pre}.{e}.up_proj.weight").float().numpy()
                d = pool.get(f"{pre}.{e}.down_proj.weight").float().numpy()
                futs.append(ex.submit(work, e, g, u, d))
            for fu in futs:
                fu.result()
    out[f"{pre}.gate_up_proj"] = torch.from_numpy(gu_q.reshape(-1))
    out[f"{pre}.gate_up_proj.qs"] = torch.from_numpy(gu_s)
    out[f"{pre}.down_proj"] = torch.from_numpy(dn_q.reshape(-1))
    out[f"{pre}.down_proj.qs"] = torch.from_numpy(dn_s)


def convert(indir, outdir, xbits, watch=False):
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(os.path.join(indir, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    E = cfg["num_experts"]
    mlp_only = set(cfg.get("mlp_only_layers", [0]))

    while True:
        smap = build_shard_map(indir)
        have = set(smap)
        # top-level tensors (embed / final norm / lm_head)
        top_done = os.path.exists(os.path.join(outdir, "out-top.safetensors"))
        top_ready = all(n in have for n in ("model.embed_tokens.weight", "model.norm.weight"))
        if not top_done and top_ready:
            pool = ShardPool(smap); out = {}
            for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                if pool.has(n):
                    t = pool.get(n)
                    out[n] = t.float() if F32_RE.search(n) else t
            save_file(out, os.path.join(outdir, "out-top.safetensors.tmp"))
            os.replace(os.path.join(outdir, "out-top.safetensors.tmp"),
                       os.path.join(outdir, "out-top.safetensors"))
            pool.close(); print("out-top.safetensors", flush=True)

        for li in range(n_layers):
            dst = os.path.join(outdir, f"out-layer-{li:03d}.safetensors")
            if os.path.exists(dst):
                continue
            pre = f"model.layers.{li}"
            is_moe = li not in mlp_only
            # ready only when every tensor this layer needs has downloaded
            need = [f"{pre}.input_layernorm.weight", f"{pre}.post_attention_layernorm.weight",
                    f"{pre}.self_attn.q_proj.weight", f"{pre}.self_attn.o_proj.weight"]
            if is_moe:
                need += [f"{pre}.mlp.experts.{E-1}.down_proj.weight", f"{pre}.mlp.gate.weight"]
            else:
                need += [f"{pre}.mlp.down_proj.weight"]
            if not all(n in have for n in need):
                continue
            t0 = time.time()
            pool = ShardPool(smap); out = {}
            # attention + norms + gates (residents)
            for suf in ("input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                        "self_attn.g_proj.weight", "self_attn.q_norm.weight",
                        "self_attn.k_norm.weight"):
                n = f"{pre}.{suf}"
                if pool.has(n):
                    t = pool.get(n); out[n] = t.float() if F32_RE.search(n) else t
            if is_moe:
                for suf in ("mlp.gate.weight", "mlp.experts.e_score_correction_bias",
                            "mlp.shared_expert.gate_proj.weight",
                            "mlp.shared_expert.up_proj.weight",
                            "mlp.shared_expert.down_proj.weight"):
                    n = f"{pre}.{suf}"
                    if pool.has(n):
                        t = pool.get(n); out[n] = t.float() if F32_RE.search(n) else t
                if xbits:
                    quant_experts(pool, li, E, xbits, out)
                else:
                    for e in range(E):
                        for suf in ("gate_proj", "up_proj", "down_proj"):
                            n = f"{pre}.mlp.experts.{e}.{suf}.weight"; out[n] = pool.get(n)
            else:
                for suf in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                    n = f"{pre}.{suf}"; out[n] = pool.get(n)
            save_file(out, dst + ".tmp"); os.replace(dst + ".tmp", dst)
            pool.close()
            print(f"out-layer-{li:03d}.safetensors ({os.path.getsize(dst)/1e9:.2f}G) "
                  f"in {time.time()-t0:.0f}s", flush=True)

        done = os.path.exists(os.path.join(outdir, "out-top.safetensors")) and all(
            os.path.exists(os.path.join(outdir, f"out-layer-{li:03d}.safetensors")) for li in range(n_layers))
        if done:
            break
        if not watch:
            break
        time.sleep(60)

    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"conversion complete -> {outdir}")


def selftest():
    w = np.random.randn(16, 32).astype(np.float32)
    q, s = quant_int4_rows(w)
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = ((q >> 4) & 0x0F).astype(np.int32) - 8
    deq = np.empty_like(w); deq[:, 0::2] = lo; deq[:, 1::2] = hi; deq *= s[:, None]
    ref = np.clip(np.rint(w / s[:, None]), -8, 7) * s[:, None]
    assert np.array_equal(deq, ref), "int4 round-trip mismatch"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--xbits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir required")
    convert(a.indir, a.outdir, a.xbits, watch=a.watch)


if __name__ == "__main__":
    main()
