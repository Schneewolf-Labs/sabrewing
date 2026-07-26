#!/usr/bin/env python3
"""Convert the Qwen3.5-MoE BF16 checkpoint into a sabrewing snapshot.

Qwen3.5-35B-A3B is a *multimodal* checkpoint (`Qwen3_5MoeForConditionalGeneration`) whose
text tower is the interesting part: 40 layers, 30 of them gated-delta-net linear attention
(see docs/linear-attention.md), 10 full attention, 256 experts top-8 everywhere.

Three things differ from convert_laguna_int4.py and each one was verified against the actual
checkpoint shapes rather than assumed:

1. **Names are nested under `model.language_model.`** (the multimodal wrapper). This converter
   strips that prefix so the container looks like every other engine's — `model.layers.N.*`,
   `model.embed_tokens.weight`, `model.norm.weight` — and a future qwen35.c can reuse the same
   loader shapes as laguna.c. `lm_head.*` and `mtp.*` are already top-level and pass through.

2. **Routed experts are already fused into 3D tensors** and, happily, in exactly the layout
   laguna.c wants:
       gate_up_proj [E, 2I, D]   used as F.linear(x, W[e]).chunk(2, dim=-1)
       down_proj    [E, D, I]
   so `gate_up_proj[e]` is nn.Linear-oriented [2I, D] with all gate rows then all up rows —
   the same block-concat laguna.c already assumes. No transpose, no de-interleaving. Verified:
   the shipped tensors are [256, 1024, 2048] and [256, 2048, 512] with I=512, D=2048.

3. **The MTP head uses the OLD per-expert layout** (`mtp.layers.0.mlp.experts.{e}.gate_proj`)
   while the main layers use the fused 3D form. Same checkpoint, two conventions — so both
   read paths exist here. The MTP head is one full-attention decoder layer plus `fc`,
   `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `norm`; it is a purpose-trained draft head
   and the reason to keep it is speculative decoding, which is the only lever that beats the
   per-token weight-read floor at batch 1.

Output policy, matching laguna's container:
  - routed experts (>90% of params) -> int4 (or --xbits 8) + `.qs` f32 row scales
  - f32: every norm, the router, `A_log`, `dt_bias`, `conv1d.weight`, `shared_expert_gate`
    (small and numerically load-bearing -- A_log/dt_bias feed exp/softplus, and the conv is a
    4-tap filter where rounding is not amortized over a wide reduction)
  - bf16 passthrough: everything else (linear_attn + self_attn projections, shared expert,
    embed, lm_head)

The vision tower is **skipped** by default: ~1.1 GB of weights with no engine support yet.
`--vision` passes it through bf16 into out-visual.safetensors so a later pass can use it.

Layout on disk: out-top / out-layer-NNN / out-mtp / (out-visual), same per-layer sharding as
laguna so --watch can convert while `hf download` is still running.

Usage:
  python3 tools/convert_qwen35_int4.py --indir DIR --outdir DIR [--xbits 4] [--watch] [--vision]
  python3 tools/convert_qwen35_int4.py --selftest
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

SRC_PREFIX = "model.language_model."
DST_PREFIX = "model."


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


# Kept f32 in the container: small, and each one feeds a nonlinearity or a narrow
# reduction where a bf16 rounding is not averaged away.
F32_RE = re.compile(
    r"(norm\.weight$|norm\.bias$|\.mlp\.gate\.weight$|\.A_log$|\.dt_bias$"
    r"|\.conv1d\.weight$|\.conv1d\.bias$|shared_expert_gate\.weight$)"
)


def dst_name(n):
    return DST_PREFIX + n[len(SRC_PREFIX):] if n.startswith(SRC_PREFIX) else n


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

    def _handle(self, name):
        p = self.smap.get(name)
        if p is None:
            raise KeyError(name)
        if p not in self.h:
            self.h[p] = safe_open(p, framework="pt")
        return self.h[p]

    def get(self, name):
        return self._handle(name).get_tensor(name)

    def slice(self, name):
        """Partial reads, so a 1 GB fused expert tensor never lands whole in RAM."""
        return self._handle(name).get_slice(name)

    def has(self, name):
        return name in self.smap

    def close(self):
        self.h.clear()


def _emit(out, pre_dst, gu_q, gu_s, dn_q, dn_s):
    out[f"{pre_dst}.gate_up_proj"] = torch.from_numpy(gu_q.reshape(-1))
    out[f"{pre_dst}.gate_up_proj.qs"] = torch.from_numpy(gu_s)
    out[f"{pre_dst}.down_proj"] = torch.from_numpy(dn_q.reshape(-1))
    out[f"{pre_dst}.down_proj.qs"] = torch.from_numpy(dn_s)


def _alloc(E, I, D, xbits):
    w = 2 if xbits == 4 else 1           # columns per stored byte
    gu_q = np.empty((E * 2 * I, D // w), np.uint8)
    dn_q = np.empty((E * D, I // w), np.uint8)
    return gu_q, np.empty(E * 2 * I, np.float32), dn_q, np.empty(E * D, np.float32)


def quant_experts_fused(pool, src_pre, dst_pre, E, xbits, out, wave=16):
    """Qwen3.5 main layers: gate_up_proj [E,2I,D], down_proj [E,D,I] -> int4 rows.

    gate_up_proj[e] is already nn.Linear-oriented [2I, D] with gate rows first, which is
    exactly the engine's block-concat expectation -- so this is a row-quantize, not a
    relayout. Read in waves so peak RAM stays ~a few hundred MB of f32 rather than the
    2 GB the whole f32 expansion of one layer would take."""
    gu_sl = pool.slice(f"{src_pre}.gate_up_proj")
    dn_sl = pool.slice(f"{src_pre}.down_proj")
    Eg, twoI, D = gu_sl.get_shape()
    Ed, Dd, I = dn_sl.get_shape()
    assert Eg == Ed == E, f"expert count {Eg}/{Ed} != config {E}"
    assert twoI == 2 * I and Dd == D, f"shape mismatch gate_up{gu_sl.get_shape()} down{dn_sl.get_shape()}"
    quant = quant_int4_rows if xbits == 4 else quant_int8_rows
    gu_q, gu_s, dn_q, dn_s = _alloc(E, I, D, xbits)
    R = 2 * I
    for e0 in range(0, E, wave):
        e1 = min(e0 + wave, E)
        gu = gu_sl[e0:e1].float().numpy()          # [w, 2I, D]
        dn = dn_sl[e0:e1].float().numpy()          # [w, D,  I]
        for j, e in enumerate(range(e0, e1)):
            q, s = quant(gu[j]); gu_q[e * R:(e + 1) * R] = q; gu_s[e * R:(e + 1) * R] = s
            q, s = quant(dn[j]); dn_q[e * D:(e + 1) * D] = q; dn_s[e * D:(e + 1) * D] = s
    _emit(out, dst_pre, gu_q, gu_s, dn_q, dn_s)
    return I, D


def quant_experts_per_expert(pool, src_pre, dst_pre, E, xbits, out):
    """MTP head: experts.{e}.{gate,up,down}_proj.weight -> the same fused int4 blobs.

    The checkpoint uses this older layout for `mtp.*` only. gate/up are concatenated here to
    produce the identical container the fused path emits, so the engine sees one format."""
    g0 = pool.get(f"{src_pre}.0.gate_proj.weight")
    I, D = g0.shape
    quant = quant_int4_rows if xbits == 4 else quant_int8_rows
    gu_q, gu_s, dn_q, dn_s = _alloc(E, I, D, xbits)
    R = 2 * I
    for e in range(E):
        g = pool.get(f"{src_pre}.{e}.gate_proj.weight").float().numpy()
        u = pool.get(f"{src_pre}.{e}.up_proj.weight").float().numpy()
        d = pool.get(f"{src_pre}.{e}.down_proj.weight").float().numpy()
        gu = np.concatenate([g, u], axis=0)        # [2I, D], gate rows then up rows
        q, s = quant(gu); gu_q[e * R:(e + 1) * R] = q; gu_s[e * R:(e + 1) * R] = s
        q, s = quant(d);  dn_q[e * D:(e + 1) * D] = q; dn_s[e * D:(e + 1) * D] = s
    _emit(out, dst_pre, gu_q, gu_s, dn_q, dn_s)
    return I, D


def copy_plain(pool, out, names):
    """bf16 passthrough, or f32 for the load-bearing small tensors. conv1d weights are
    squeezed from Conv1d's [C,1,K] to [C,K] since the engine treats them as depthwise rows."""
    for n in names:
        if not pool.has(n):
            continue
        t = pool.get(n)
        if n.endswith(".conv1d.weight") and t.dim() == 3:
            t = t.squeeze(1)
        out[dst_name(n)] = t.float() if F32_RE.search(n) else t


def layer_tensor_names(pre, kind):
    """Every non-expert tensor a layer owns, by mixer kind."""
    common = [f"{pre}.input_layernorm.weight", f"{pre}.post_attention_layernorm.weight",
              f"{pre}.mlp.gate.weight", f"{pre}.mlp.shared_expert_gate.weight",
              f"{pre}.mlp.shared_expert.gate_proj.weight",
              f"{pre}.mlp.shared_expert.up_proj.weight",
              f"{pre}.mlp.shared_expert.down_proj.weight"]
    if kind == "linear_attention":
        return common + [f"{pre}.linear_attn.{s}" for s in (
            "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
            "conv1d.weight", "conv1d.bias", "dt_bias", "A_log", "norm.weight",
            "out_proj.weight")]
    return common + [f"{pre}.self_attn.{s}" for s in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight")]


def convert(indir, outdir, xbits, watch=False, vision=False):
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(os.path.join(indir, "config.json")))
    tc = cfg.get("text_config", cfg)
    n_layers = tc["num_hidden_layers"]
    E = tc["num_experts"]
    layer_types = tc.get("layer_types")
    if not layer_types:
        iv = tc.get("full_attention_interval", 4)
        layer_types = ["full_attention" if (i + 1) % iv == 0 else "linear_attention"
                       for i in range(n_layers)]
    n_lin = sum(t == "linear_attention" for t in layer_types)
    print(f"{n_layers} layers ({n_lin} linear_attention, {n_layers - n_lin} full_attention), "
          f"{E} experts top-{tc['num_experts_per_tok']}, hidden={tc['hidden_size']}, "
          f"xbits={xbits}", flush=True)

    while True:
        smap = build_shard_map(indir)
        have = set(smap)

        top = os.path.join(outdir, "out-top.safetensors")
        top_names = [SRC_PREFIX + "embed_tokens.weight", SRC_PREFIX + "norm.weight",
                     "lm_head.weight"]
        if not os.path.exists(top) and all(n in have for n in top_names[:2]):
            pool = ShardPool(smap); out = {}
            copy_plain(pool, out, top_names)
            save_file(out, top + ".tmp"); os.replace(top + ".tmp", top)
            pool.close(); print(f"out-top.safetensors ({os.path.getsize(top)/1e9:.2f}G)", flush=True)

        for li in range(n_layers):
            dst = os.path.join(outdir, f"out-layer-{li:03d}.safetensors")
            if os.path.exists(dst):
                continue
            src_pre = f"{SRC_PREFIX}layers.{li}"
            kind = layer_types[li]
            names = layer_tensor_names(src_pre, kind)
            # ready only when the expert blobs and the mixer's own weights have landed
            fused = f"{src_pre}.mlp.experts.gate_up_proj" in have
            # save_pretrained writes the per-expert layout even for checkpoints whose
            # runtime module holds them fused, so a locally-saved model (e.g. the tiny
            # oracle from make_tiny_qwen35.py) arrives in the older shape. Accept either.
            need = [f"{src_pre}.input_layernorm.weight"]
            need += [f"{src_pre}.mlp.experts.gate_up_proj", f"{src_pre}.mlp.experts.down_proj"] \
                if fused else [f"{src_pre}.mlp.experts.{E-1}.down_proj.weight"]
            need += [f"{src_pre}.linear_attn.in_proj_qkv.weight"] if kind == "linear_attention" \
                else [f"{src_pre}.self_attn.q_proj.weight"]
            if not all(n in have for n in need):
                continue
            t0 = time.time()
            pool = ShardPool(smap); out = {}
            copy_plain(pool, out, names)
            if xbits:
                (quant_experts_fused if fused else quant_experts_per_expert)(
                    pool, f"{src_pre}.mlp.experts", f"model.layers.{li}.mlp.experts",
                    E, xbits, out)
            elif fused:
                for suf in ("gate_up_proj", "down_proj"):
                    n = f"{src_pre}.mlp.experts.{suf}"
                    out[dst_name(n)] = pool.get(n)
            else:
                for e in range(E):
                    for suf in ("gate_proj", "up_proj", "down_proj"):
                        n = f"{src_pre}.mlp.experts.{e}.{suf}.weight"
                        out[dst_name(n)] = pool.get(n)
            save_file(out, dst + ".tmp"); os.replace(dst + ".tmp", dst)
            pool.close()
            print(f"out-layer-{li:03d}.safetensors [{kind:17s}] "
                  f"({os.path.getsize(dst)/1e9:.2f}G) in {time.time()-t0:.0f}s", flush=True)

        # MTP draft head: one full-attention decoder layer + fc/norms, per-expert MoE layout
        mtp = os.path.join(outdir, "out-mtp.safetensors")
        if not os.path.exists(mtp) and "mtp.fc.weight" in have:
            t0 = time.time()
            pool = ShardPool(smap); out = {}
            copy_plain(pool, out, ["mtp.fc.weight", "mtp.norm.weight",
                                   "mtp.pre_fc_norm_embedding.weight",
                                   "mtp.pre_fc_norm_hidden.weight"]
                                  + layer_tensor_names("mtp.layers.0", "full_attention"))
            if xbits:
                quant_experts_per_expert(pool, "mtp.layers.0.mlp.experts",
                                         "mtp.layers.0.mlp.experts", E, xbits, out)
            save_file(out, mtp + ".tmp"); os.replace(mtp + ".tmp", mtp)
            pool.close()
            print(f"out-mtp.safetensors ({os.path.getsize(mtp)/1e9:.2f}G) in "
                  f"{time.time()-t0:.0f}s", flush=True)

        vis = os.path.join(outdir, "out-visual.safetensors")
        vis_names = sorted(n for n in have if n.startswith("model.visual."))
        if vision and vis_names and not os.path.exists(vis):
            pool = ShardPool(smap); out = {}
            copy_plain(pool, out, vis_names)
            save_file(out, vis + ".tmp"); os.replace(vis + ".tmp", vis)
            pool.close(); print(f"out-visual.safetensors ({os.path.getsize(vis)/1e9:.2f}G)", flush=True)
        elif vis_names and not vision:
            print(f"skipped {len(vis_names)} model.visual.* tensors (no engine support; "
                  f"--vision to keep them)", flush=True)

        done = os.path.exists(top) and all(
            os.path.exists(os.path.join(outdir, f"out-layer-{li:03d}.safetensors"))
            for li in range(n_layers))
        if done or not watch:
            break
        time.sleep(60)

    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "special_tokens_map.json", "chat_template.jinja",
               "generation_config.json"):
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    tot = sum(os.path.getsize(os.path.join(outdir, f)) for f in os.listdir(outdir)
              if f.endswith(".safetensors"))
    print(f"conversion complete -> {outdir}  ({tot/1e9:.1f} GB)")


def selftest():
    # int4 round-trip: the packed nibbles must dequantize to the same values the
    # naive rint/clip reference produces.
    w = np.random.randn(16, 32).astype(np.float32)
    q, s = quant_int4_rows(w)
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = ((q >> 4) & 0x0F).astype(np.int32) - 8
    deq = np.empty_like(w); deq[:, 0::2] = lo; deq[:, 1::2] = hi; deq *= s[:, None]
    ref = np.clip(np.rint(w / s[:, None]), -8, 7) * s[:, None]
    assert np.array_equal(deq, ref), "int4 round-trip mismatch"

    # Orientation: the two expert layouts in this one checkpoint must produce byte-identical
    # containers. This is the assumption the whole converter rests on, so it gets a test
    # rather than a comment -- a silent transpose here would only show up as a broken model.
    E, I, D = 3, 8, 6
    g = [np.random.randn(I, D).astype(np.float32) for _ in range(E)]
    u = [np.random.randn(I, D).astype(np.float32) for _ in range(E)]
    d = [np.random.randn(D, I).astype(np.float32) for _ in range(E)]
    fused_gu = np.stack([np.concatenate([g[e], u[e]], axis=0) for e in range(E)])  # [E,2I,D]
    fused_dn = np.stack(d)                                                          # [E,D,I]
    a_gu, a_gs, a_dn, a_ds = _alloc(E, I, D, 4)
    R = 2 * I
    for e in range(E):
        q1, s1 = quant_int4_rows(fused_gu[e]); a_gu[e*R:(e+1)*R] = q1; a_gs[e*R:(e+1)*R] = s1
        q1, s1 = quant_int4_rows(fused_dn[e]); a_dn[e*D:(e+1)*D] = q1; a_ds[e*D:(e+1)*D] = s1
    b_gu, b_gs, b_dn, b_ds = _alloc(E, I, D, 4)
    for e in range(E):
        gu = np.concatenate([g[e], u[e]], axis=0)
        q1, s1 = quant_int4_rows(gu); b_gu[e*R:(e+1)*R] = q1; b_gs[e*R:(e+1)*R] = s1
        q1, s1 = quant_int4_rows(d[e]); b_dn[e*D:(e+1)*D] = q1; b_ds[e*D:(e+1)*D] = s1
    assert np.array_equal(a_gu, b_gu) and np.array_equal(a_gs, b_gs), "gate_up layouts differ"
    assert np.array_equal(a_dn, b_dn) and np.array_equal(a_ds, b_ds), "down layouts differ"

    # gate rows come first: F.linear(x, gate_up[e]).chunk(2,-1)[0] must be the gate half.
    x = np.random.randn(D).astype(np.float32)
    y = fused_gu[0] @ x
    assert np.allclose(y[:I], g[0] @ x) and np.allclose(y[I:], u[0] @ x), "gate/up order"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--xbits", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--vision", action="store_true", help="also emit the vision tower (bf16)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.indir or not a.outdir:
        ap.error("--indir and --outdir required")
    convert(a.indir, a.outdir, a.xbits, watch=a.watch, vision=a.vision)


if __name__ == "__main__":
    main()
