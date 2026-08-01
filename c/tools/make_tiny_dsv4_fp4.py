#!/usr/bin/env python3
"""Rewrite a tiny f32 DeepSeek-V4 snapshot as an fp4/bf16 container.

The real container (tools/convert_deepseek_fp4.py) can only be exercised by the
167 GB checkpoint, which no test can depend on. This builds the same container
shape out of the tiny snapshot the dependency-free harness already produces, so
`deepseek.c`'s fp4 expert path and bf16 resident tier get run by `make
deepseek-fp4` in under a second:

  experts   f32 -> fp4 e2m1 + per-32-block ue8m0 scales, fused per layer
  dense 2D  f32 -> bf16
  the rest  unchanged

Unlike the real conversion this one is LOSSY — the source is f32, not already-fp4 —
so comparing it to the ORIGINAL f32 run would only measure quantization error on
random weights, which says nothing about the container. Instead this also writes a
second snapshot holding the DEQUANTIZED values as plain f32. Both then describe the
exact same weights, so the fp4 path and the f32 path must agree to float noise, and
any disagreement is a real plumbing bug: fused gate/up row order, nibble order,
block-scale stride, or the kernel's decode.

Usage: python3 tools/make_tiny_dsv4_fp4.py <in-snapshot> <out-fp4> <out-dequantized>
"""
import json
import os
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_deepseek_fp4 import (FP4_LUT, Shards, f32_to_bf16, save_file)

# e2m1 magnitudes, ascending; index into FP4_LUT is code = (sign << 3) | mag_index
MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def quant_fp4_blocks(w, block=32):
    """f32 [O,I] -> (packed u8 [O,I/2], ue8m0 u8 [O,I/block]).

    Power-of-two block scale from amax (what the reference quantizer does), then
    round each value to the nearest e2m1 level."""
    O, I = w.shape
    assert I % block == 0, (f"row width {I} must be a multiple of {block}: the fp4 "
        "container indexes block scales with stride I/32. Build the source snapshot "
        "with DSV4_D / DSV4_MOE_I set to multiples of 32.")
    v = w.reshape(O, I // block, block)
    amax = np.abs(v).max(axis=2)
    # scale = 2^ceil(log2(amax/6)); amax==0 -> exponent 127 (scale 1.0)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.ceil(np.log2(amax / 6.0)) + 127, 127)
    e = np.clip(e, 1, 254).astype(np.uint8)
    scale = np.ldexp(1.0, e.astype(np.int32) - 127).astype(np.float32)
    x = v / scale[:, :, None]
    mag = np.abs(x)
    # nearest magnitude on the e2m1 ladder
    idx = np.abs(mag[..., None] - MAGS[None, None, None, :]).argmin(axis=-1)
    code = (idx + np.where(x < 0, 8, 0)).astype(np.uint8).reshape(O, I)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, e.astype(np.uint8)


def dequant_fp4(packed, e, block=32):
    """inverse of quant_fp4_blocks -> f32 [O, I]"""
    lo = FP4_LUT[packed & 0x0F]
    hi = FP4_LUT[(packed >> 4) & 0x0F]
    O, half = packed.shape
    x = np.empty((O, half * 2), np.float32)
    x[:, 0::2], x[:, 1::2] = lo, hi
    scale = np.ldexp(1.0, e.astype(np.int32) - 127).astype(np.float32)
    return x * np.repeat(scale, block, axis=1)


def main():
    src_dir, dst_dir, deq_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(deq_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    E = cfg["n_routed_experts"]
    D = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    sh = Shards(src_dir)

    # anything matching these stays f32 (norms, biases, mHC, sinks, tables)
    def keep_f32(name):
        return any(k in name for k in ("norm", "sink", "position_bias", "hc_", "_hc.",
                                       "tid2eid", "correction_bias", "scale", "base"))

    out, deq = {}, {}
    for name in sh.index:
        if ".mlp.experts." in name:
            continue
        arr, dt = sh.array(name)
        if dt == "I64":
            out[name] = deq[name] = (arr, "i64")
        elif dt == "BF16":
            out[name] = deq[name] = (arr, "bf16")
        elif keep_f32(name) or arr.ndim < 2:
            out[name] = deq[name] = (arr.astype(np.float32), "f32")
        else:
            # bf16 in BOTH: the resident tier is what differs, and bf16 widened to f32
            # is exact, so the two snapshots still hold identical values.
            out[name] = deq[name] = (f32_to_bf16(arr.astype(np.float32)), "bf16")

    for li in range(n_layers):
        pre = f"model.layers.{li}.mlp.experts"
        fused = f"{pre}.gate_up_proj"
        if sh.has(fused):                      # already-fused f32 [E, 2I, D]
            gu_src = sh.array(fused)[0].reshape(E * 2 * I, D)
            dn_src = sh.array(f"{pre}.down_proj")[0].reshape(E * D, I)
        else:                                  # per-expert w1/w2/w3
            gu_src = np.empty((E * 2 * I, D), np.float32)
            dn_src = np.empty((E * D, I), np.float32)
            for e in range(E):
                for j, w in enumerate(("w1", "w3")):
                    a = sh.array(f"{pre}.{e}.{w}.weight")[0].astype(np.float32)
                    gu_src[e * 2 * I + j * I: e * 2 * I + (j + 1) * I] = a
                dn_src[e * D:(e + 1) * D] = sh.array(f"{pre}.{e}.w2.weight")[0]
        gu, gu_e = quant_fp4_blocks(gu_src.astype(np.float32))
        dn, dn_e = quant_fp4_blocks(dn_src.astype(np.float32))
        out[fused] = (gu, "u8")
        out[fused + ".es"] = (gu_e, "u8")
        out[f"{pre}.down_proj"] = (dn, "u8")
        out[f"{pre}.down_proj.es"] = (dn_e, "u8")
        deq[fused] = (dequant_fp4(gu, gu_e).reshape(E, 2 * I, D), "f32")
        deq[f"{pre}.down_proj"] = (dequant_fp4(dn, dn_e).reshape(E, D, I), "f32")

    save_file(os.path.join(dst_dir, "model.safetensors"), out)
    save_file(os.path.join(deq_dir, "model.safetensors"), deq)
    for fn in ("config.json", "generation_config.json", "ref_deepseek.json",
               "tokenizer.json", "tokenizer_config.json"):
        s = os.path.join(src_dir, fn)
        if os.path.exists(s):
            shutil.copy(s, dst_dir)
            shutil.copy(s, deq_dir)
    sz = os.path.getsize(os.path.join(dst_dir, "model.safetensors"))
    print(f"wrote fp4/bf16 container -> {dst_dir} ({sz/1e6:.1f} MB)")
    print(f"wrote dequantized f32 twin -> {deq_dir} "
          f"({os.path.getsize(os.path.join(deq_dir, 'model.safetensors'))/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
