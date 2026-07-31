#!/usr/bin/env python3
"""Convert the shipped DeepSeek-V4-Flash checkpoint into a snapshot deepseek.c reads.

Two things have to change; neither is a requantization.

1. NAMES. The HF repo ships DeepSeek's own inference layout (`layers.0.attn.wq_a`),
   not the transformers layout deepseek.c was written against
   (`model.layers.0.self_attn.q_a_proj`). MAP below is the whole translation, and
   it is the inverse of the repo's own inference/convert.py.

2. LAYOUT. Routed experts ship per-expert (`ffn.experts.{e}.w1/w2/w3`); the engine
   wants them fused per layer, gate rows then up rows:

     model.layers.{i}.mlp.experts.gate_up_proj      u8  [E*2I, D/2]   fp4 e2m1 codes
     model.layers.{i}.mlp.experts.gate_up_proj.es   u8  [E*2I, D/32]  ue8m0 block scales
     model.layers.{i}.mlp.experts.down_proj         u8  [E*D,  I/2]
     model.layers.{i}.mlp.experts.down_proj.es      u8  [E*D,  I/32]

Expert bytes are copied VERBATIM. The checkpoint is already fp4 (e2m1) with a
per-32-block ue8m0 scale, and sabrewing's int4 container is uniform 4-bit with one
scale per row — re-gridding onto it costs 17% relative error (9.7% even at matched
group-32 granularity) because e2m1's levels are non-uniform. Measured on real layer-0
experts, taking the shipped fp4 weights as ground truth:

    int4 per-row 17.0% | int4 group-128 11.8% | int4 group-32 9.7% | fp4 verbatim 0%

So the experts keep their own format and moe_quant.h grew matmul_fp4_k to read it.
The nibble convention already matches (low = even column, high = odd column), which
is why this is a byte copy and not a repack. It also makes conversion I/O-bound
instead of a dequant/requant pass over 277B parameters.

Everything else is dequantized to bf16 (st.h converts on load):
  - dense fp8: e4m3 codes * 2^(ue8m0 scale - 127) over 128x128 blocks
  - bf16/f32 tensors pass through

The `mtp.*` tensors are SKIPPED. That is the DSpark speculative-decoding module
(config `dspark_*`, 3 extra layers); vLLM only runs it behind an opt-in flag, and
this engine does not implement it. The 43 main layers stand alone without it.

Usage:
  python3 tools/convert_deepseek_fp4.py --indir <hf snapshot> --outdir <snapshot>
  python3 tools/convert_deepseek_fp4.py --selftest
"""
import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

# e2m1 code -> value. Index is the raw nibble; must match moe_quant.h's moe_fp4_lut.
FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

FP8_BLOCK = 128     # dense e4m3 scale block (both axes)


# ---------------------------------------------------------------- safetensors reader
class Shards:
    """Minimal safetensors reader: no torch, no safetensors package, no mmap of the
    whole 167 GB. Reads exactly the byte range a tensor occupies."""

    def __init__(self, indir):
        self.index = {}         # name -> (path, dtype, shape, begin, end)
        for path in sorted(glob.glob(os.path.join(indir, "*.safetensors"))):
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.index[k] = (path, v["dtype"], tuple(v["shape"]), base + a, base + b)

    def has(self, name):
        return name in self.index

    def shape(self, name):
        return self.index[name][2]

    def raw(self, name):
        path, _dt, _sh, a, b = self.index[name]
        with open(path, "rb") as fh:
            fh.seek(a)
            return fh.read(b - a)

    def array(self, name):
        """-> numpy array in the file's own dtype (fp8/ue8m0 come back as uint8)."""
        path, dt, sh, a, b = self.index[name]
        buf = self.raw(name)
        npdt = {"F32": np.float32, "F16": np.float16, "I64": np.int64,
                "I32": np.int32, "I8": np.uint8, "U8": np.uint8,
                "BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8}[dt]
        return np.frombuffer(buf, dtype=npdt).reshape(sh), dt


# ---------------------------------------------------------------- dtype helpers
def ue8m0(e):
    """ue8m0 byte -> 2^(e-127). e=0 is the denormal 2^-127, 0xFF is the format NaN."""
    out = np.ldexp(np.ones_like(e, dtype=np.float32), e.astype(np.int32) - 127)
    return np.where(e == 0xFF, np.float32(0.0), out)


def e4m3_to_f32(b):
    """float8_e4m3fn (S.EEEE.MMM, bias 7, no inf, 0xFF/0x7F = NaN) -> float32."""
    b = b.astype(np.uint32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    man = b & 0x7
    # normal: 2^(exp-7) * (1 + man/8);  subnormal (exp==0): 2^-6 * (man/8)
    val = np.where(exp == 0,
                   np.ldexp(man.astype(np.float32) / 8.0, -6),
                   np.ldexp(1.0 + man.astype(np.float32) / 8.0, exp.astype(np.int32) - 7))
    val = np.where((exp == 0xF) & (man == 0x7), np.float32(np.nan), val)
    return np.where(sign == 1, -val, val).astype(np.float32)


def f32_to_bf16(x):
    """round-to-nearest-even f32 -> bf16 (stored as uint16)."""
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def dequant_fp8_block(w_u8, s_u8):
    """e4m3 [O,I] with ue8m0 [O/128, I/128] block scales -> f32 [O,I]."""
    O, I = w_u8.shape
    x = e4m3_to_f32(w_u8)
    s = ue8m0(s_u8)
    # expand each block scale over its 128x128 tile (last tile may be short)
    s = np.repeat(np.repeat(s, FP8_BLOCK, axis=0), FP8_BLOCK, axis=1)[:O, :I]
    return x * s


# ---------------------------------------------------------------- name mapping
# native suffix -> transformers suffix, applied under `model.layers.{i}.`
MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight",
    "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight",
    "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight",
    "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.attn_sink": "self_attn.sinks",
    "hc_attn_fn": "attn_hc.fn",
    "hc_attn_base": "attn_hc.base",
    "hc_attn_scale": "attn_hc.scale",
    "hc_ffn_fn": "ffn_hc.fn",
    "hc_ffn_base": "ffn_hc.base",
    "hc_ffn_scale": "ffn_hc.scale",
    "ffn.gate.weight": "mlp.gate.weight",
    "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid",
    "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    # outer compressor (CSA and HCA layers)
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    # lightning indexer (CSA layers). NOTE the native order is indexer.compressor.*,
    # the transformers order is compressor.indexer.* — the nesting is swapped.
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight":
        "self_attn.compressor.indexer.scorer.weights_proj.weight",
}

TOP_MAP = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "head.weight": "head.weight",          # deepseek.c accepts head.weight directly
    "hc_head_fn": "model.hc_head.hc_fn",
    "hc_head_base": "model.hc_head.hc_base",
    "hc_head_scale": "model.hc_head.hc_scale",
}

# these stay f32 in the container (tiny, and the engine reads them as f32 anyway)
F32_KEEP = re.compile(r"(hc_fn|hc_base|hc_scale|attn_hc\.|ffn_hc\.|sinks|position_bias)")


def load_dense(sh, name):
    """Any non-expert tensor -> (numpy array ready to save, dtype tag)."""
    arr, dt = sh.array(name)
    if dt == "F8_E4M3":
        s, _ = sh.array(name.replace(".weight", ".scale"))
        return dequant_fp8_block(arr, s), "f32"
    if dt == "BF16":
        return arr, "bf16"          # already bf16, pass the raw u16 through
    if dt == "I64":
        return arr, "i64"
    return arr.astype(np.float32), "f32"


# ---------------------------------------------------------------- safetensors writer
DT_TAG = {"bf16": "BF16", "f32": "F32", "u8": "U8", "i64": "I64"}


def save_file(path, tensors):
    """tensors: name -> (numpy array, tag). Written in insertion order."""
    header, offset, blobs = {}, 0, []
    for name, (arr, tag) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": DT_TAG[tag], "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)          # pad to 8 bytes
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)
    os.replace(tmp, path)


# ---------------------------------------------------------------- expert fusion
def fuse_experts(sh, li, E, D, I, out):
    """Concatenate the per-expert fp4 blobs into the two fused tensors, verbatim."""
    pre = f"layers.{li}.ffn.experts"
    gu = np.empty((E * 2 * I, D // 2), dtype=np.uint8)
    gu_s = np.empty((E * 2 * I, D // 32), dtype=np.uint8)
    dn = np.empty((E * D, I // 2), dtype=np.uint8)
    dn_s = np.empty((E * D, I // 32), dtype=np.uint8)
    for e in range(E):
        for j, w in enumerate(("w1", "w3")):        # gate rows, then up rows
            a, dt = sh.array(f"{pre}.{e}.{w}.weight")
            s, _ = sh.array(f"{pre}.{e}.{w}.scale")
            if dt not in ("I8", "U8"):
                sys.exit(f"{pre}.{e}.{w}.weight is {dt}, expected packed fp4 (I8). "
                         "This converter only handles the fp4 expert container.")
            r0 = e * 2 * I + j * I
            gu[r0:r0 + I] = a
            gu_s[r0:r0 + I] = s
        a, _ = sh.array(f"{pre}.{e}.w2.weight")     # down
        s, _ = sh.array(f"{pre}.{e}.w2.scale")
        r0 = e * D
        dn[r0:r0 + D] = a
        dn_s[r0:r0 + D] = s
    p = f"model.layers.{li}.mlp.experts"
    out[f"{p}.gate_up_proj"] = (gu, "u8")
    out[f"{p}.gate_up_proj.es"] = (gu_s, "u8")
    out[f"{p}.down_proj"] = (dn, "u8")
    out[f"{p}.down_proj.es"] = (dn_s, "u8")


# ---------------------------------------------------------------- driver
def convert(indir, outdir, layers=None):
    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(os.path.join(indir, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    E = cfg["n_routed_experts"]
    D = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    want = range(n_layers) if layers is None else layers

    print(f"[convert] {n_layers} layers, {E} experts, D={D} I={I}", flush=True)
    sh = Shards(indir)
    skipped_mtp = sum(1 for k in sh.index if k.startswith("mtp."))
    print(f"[convert] {len(sh.index)} tensors, skipping {skipped_mtp} mtp.* "
          f"(DSpark speculative module, not implemented)", flush=True)

    dst = os.path.join(outdir, "out-top.safetensors")
    if not os.path.exists(dst):
        out = {}
        for src, tgt in TOP_MAP.items():
            if not sh.has(src):
                continue
            arr, tag = load_dense(sh, src)
            if tag == "f32" and not F32_KEEP.search(tgt):
                arr, tag = f32_to_bf16(arr), "bf16"
            out[tgt] = (arr, tag)
        save_file(dst, out)
        print(f"  out-top.safetensors ({os.path.getsize(dst)/1e9:.2f} GB)", flush=True)

    for li in want:
        dst = os.path.join(outdir, f"out-layer-{li:03d}.safetensors")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        out = {}
        for suf, tgt in MAP.items():
            src = f"layers.{li}.{suf}"
            if not sh.has(src):
                continue
            arr, tag = load_dense(sh, src)
            full = f"model.layers.{li}.{tgt}"
            if tag == "f32" and not F32_KEEP.search(full):
                arr, tag = f32_to_bf16(arr), "bf16"
            out[full] = (arr, tag)
        fuse_experts(sh, li, E, D, I, out)
        save_file(dst, out)
        print(f"  out-layer-{li:03d}.safetensors ({os.path.getsize(dst)/1e9:.2f} GB) "
              f"in {time.time()-t0:.0f}s", flush=True)

    for fn in ("config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json"):
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"[convert] done -> {outdir}", flush=True)


# ---------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    # ue8m0 / e4m3 decode against hand-computed values
    assert ue8m0(np.array([127], np.uint8))[0] == 1.0
    assert ue8m0(np.array([128], np.uint8))[0] == 2.0
    assert ue8m0(np.array([126], np.uint8))[0] == 0.5
    assert e4m3_to_f32(np.array([0x38], np.uint8))[0] == 1.0     # exp=7 man=0
    assert e4m3_to_f32(np.array([0xB8], np.uint8))[0] == -1.0
    assert e4m3_to_f32(np.array([0x3C], np.uint8))[0] == 1.5     # 1 + 4/8
    assert e4m3_to_f32(np.array([0x00], np.uint8))[0] == 0.0

    # fp4 nibble order: low nibble is the EVEN column (matches moe_quant.h)
    codes = rng.integers(0, 16, size=(4, 8), dtype=np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    lo = FP4_LUT[packed & 0x0F]
    hi = FP4_LUT[(packed >> 4) & 0x0F]
    un = np.empty((4, 8), np.float32)
    un[:, 0::2], un[:, 1::2] = lo, hi
    assert np.array_equal(un, FP4_LUT[codes]), "nibble order mismatch"

    # bf16 round trip is exact for values already representable
    x = f32_to_bf16(np.array([1.0, -2.5, 0.0, 1e-8], np.float32))
    assert np.allclose(bf16_to_f32(x), [1.0, -2.5, 0.0, 1e-8], rtol=1e-2)

    # fp8 block dequant vs an explicit loop
    O, I = 256, 256
    w = rng.integers(0, 255, size=(O, I), dtype=np.uint8)
    s = rng.integers(120, 134, size=(O // 128, I // 128), dtype=np.uint8)
    got = dequant_fp8_block(w, s)
    for o in (0, 5, 200):
        for i in (0, 130, 255):
            exp = e4m3_to_f32(w[o:o+1, i:i+1])[0, 0] * ue8m0(s[o//128:o//128+1, i//128:i//128+1])[0, 0]
            assert np.isnan(exp) or got[o, i] == exp, (o, i, got[o, i], exp)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir")
    ap.add_argument("--outdir")
    ap.add_argument("--layers", help="comma/range subset, e.g. 0-3 (debugging)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not a.indir or not a.outdir:
            ap.error("--indir and --outdir are required")
        sel = None
        if a.layers:
            sel = []
            for part in a.layers.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    sel += list(range(int(lo), int(hi) + 1))
                else:
                    sel.append(int(part))
        convert(a.indir, a.outdir, sel)
