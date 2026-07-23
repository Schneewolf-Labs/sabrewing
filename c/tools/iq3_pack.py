#!/usr/bin/env python3
"""fmt=6 (E8/IQ3 lattice container) index codec — #452 ladder step 2.

Note: fmt=5 is taken by the int3 dual-plane container (#132); this lattice
container is fmt=6.

The ablation (#453) proved the SCHEME: an IQ3_XXS-style codebook plus rotation
matches our simulated E8 ball (51.5% vs 51.5% on OLMoE). That code quantizes to
lattice points and keeps floats. This module produces the DEPLOYABLE bytes and
reads them back, so the container, the converter and the decode kernels all
agree on one layout.

Layout - one 256-weight super-block, 98 bytes, 3.0625 bpw:

    [0  .. 63]  uint8  grid index per 4-dim magnitude block   (64 blocks)
    [64 .. 95]  uint32 x8, one per 32-weight sub-block:
                  bits  0..20  three 7-bit sign words (8 weights each,
                               bit i set => weight i negative; the 8th sign
                               is implied by odd parity)
                  bits 21..27  the fourth 7-bit sign word
                  bits 28..31  4-bit sub-scale code
    [96 .. 97]  fp16   super-scale d

    value(w) = d * (0.5 + code) * 0.5 * grid[idx][j] * 0.5 * sign

The last 0.5 is the half-unit convention of the published grid (magnitudes are
stored doubled: 4,12,...,62 mean 2.0,6.0,...,31.0).

Odd-parity signs: llama.cpp stores 7 of every 8 signs and derives the 8th so the
product of the eight is +1. The encoder therefore flips the smallest-magnitude
weight of any block whose true signs violate that — the same cost the ablation
priced in, now applied for real.
"""
import json
import os
import numpy as np

QK = 256                      # weights per super-block
SUB = 32                      # weights per sub-block (one uint32 of signs+scale)
BLOCK_BYTES = QK // 4 + (QK // SUB) * 4 + 2      # 64 + 32 + 2 = 98

_GRID = None


def grid():
    """[256,4] float32 magnitudes in weight units (published table is doubled)."""
    global _GRID
    if _GRID is None:
        path = os.path.join(os.path.dirname(__file__), "iq3xxs_grid.json")
        _GRID = np.asarray(json.load(open(path)), dtype=np.float32) * 0.5
    return _GRID


def _nearest(mag4):
    """[N,4] magnitudes -> [N] grid indices, argmin ||m-g||^2 without cdist."""
    g = grid()
    g2 = (g * g).sum(1)
    out = np.empty(len(mag4), dtype=np.uint8)
    for i in range(0, len(mag4), 1 << 16):          # bounded working set
        c = mag4[i:i + (1 << 16)]
        out[i:i + len(c)] = np.argmin(g2 - 2.0 * (c @ g.T), axis=1).astype(np.uint8)
    return out


def encode(x):
    """float32 [..., K] (K % 256 == 0) -> packed uint8 [..., K//256 * 98]."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    K = x.shape[-1]
    if K % QK:
        raise ValueError(f"fmt=5 needs K % {QK} == 0, got {K}")
    rows = x.reshape(-1, K)
    nsb = K // QK
    out = np.empty((len(rows), nsb * BLOCK_BYTES), dtype=np.uint8)

    for sb in range(nsb):
        blk = rows[:, sb * QK:(sb + 1) * QK]                     # [R,256]
        sign = np.where(blk < 0, -1.0, 1.0).astype(np.float32)
        mag = np.abs(blk)

        # parity fix: flip the smallest magnitude of every 8 whose product is -1
        s8 = sign.reshape(len(rows), QK // 8, 8)
        m8 = mag.reshape(len(rows), QK // 8, 8)
        viol = s8.prod(-1) < 0                                   # [R,32]
        amin = m8.argmin(-1)
        r, b = np.nonzero(viol)
        s8[r, b, amin[r, b]] *= -1.0
        sign = s8.reshape(len(rows), QK)

        base = sb * BLOCK_BYTES
        # super-scale: RMS anchor, same statistic the ablation searches around
        d = np.sqrt((mag * mag).mean(-1, keepdims=True)) / 20.0 + 1e-12
        out[:, base + 96:base + 98] = d.astype(np.float16).view(np.uint8)
        d = d.astype(np.float16).astype(np.float32)              # encode what we store

        g = grid()
        for ib in range(QK // SUB):
            m = mag[:, ib * SUB:(ib + 1) * SUB]                  # [R,32]
            best_err = None
            best = None
            for code in range(16):
                db = d * (0.5 + code) * 0.5
                q = (m / np.maximum(db, 1e-20)).reshape(-1, 4)
                idx = _nearest(q)
                rec = g[idx].reshape(len(rows), SUB) * db
                err = ((rec - m) ** 2).sum(-1, keepdims=True)
                if best_err is None:
                    best_err, best = err, (idx.reshape(len(rows), SUB // 4), code)
                else:
                    take = (err < best_err)[:, 0]
                    if take.any():
                        keep_idx, keep_code = best
                        ni = idx.reshape(len(rows), SUB // 4)
                        keep_idx = np.where(take[:, None], ni, keep_idx)
                        # per-row code: store alongside, resolved below
                        keep_code = np.where(take, code, keep_code) if isinstance(
                            keep_code, np.ndarray) else np.where(
                            take, code, np.full(len(rows), keep_code))
                        best = (keep_idx, keep_code)
                        best_err = np.where(take[:, None], err, best_err)
            bidx, bcode = best
            if not isinstance(bcode, np.ndarray):
                bcode = np.full(len(rows), bcode)
            out[:, base + ib * 8:base + (ib + 1) * 8] = bidx.astype(np.uint8)

            # signs: four 7-bit words for this sub-block + the 4-bit code
            s = sign[:, ib * SUB:(ib + 1) * SUB].reshape(len(rows), 4, 8)
            neg = (s < 0).astype(np.uint32)
            word = np.zeros(len(rows), dtype=np.uint32)
            for l in range(4):
                seven = np.zeros(len(rows), dtype=np.uint32)
                for j in range(7):
                    seven |= neg[:, l, j] << j
                word |= seven << (7 * l)
            word |= (bcode.astype(np.uint32) & 0xF) << 28
            off = base + QK // 4 + ib * 4
            out[:, off:off + 4] = word.view(np.uint8).reshape(len(rows), 4) if False else \
                np.ascontiguousarray(word).view(np.uint8).reshape(len(rows), 4)
    return out.reshape(*x.shape[:-1], nsb * BLOCK_BYTES)


def decode(packed, K):
    """packed uint8 [..., K//256*98] -> float32 [..., K]. The kernels' reference."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    nsb = K // QK
    rows = packed.reshape(-1, nsb * BLOCK_BYTES)
    out = np.empty((len(rows), K), dtype=np.float32)
    g = grid()
    for sb in range(nsb):
        base = sb * BLOCK_BYTES
        d = rows[:, base + 96:base + 98].copy().view(np.float16).astype(np.float32)
        for ib in range(QK // SUB):
            idx = rows[:, base + ib * 8:base + (ib + 1) * 8]                  # [R,8]
            off = base + QK // 4 + ib * 4
            word = np.ascontiguousarray(rows[:, off:off + 4]).view(np.uint32).reshape(-1)
            code = (word >> 28) & 0xF
            db = d[:, 0] * (0.5 + code) * 0.5                                 # [R]
            mag = g[idx].reshape(len(rows), SUB)                              # [R,32]
            sgn = np.ones((len(rows), 4, 8), dtype=np.float32)
            for l in range(4):
                seven = (word >> (7 * l)) & 0x7F
                par = 0
                for j in range(7):
                    bit = (seven >> j) & 1
                    sgn[:, l, j] = np.where(bit == 1, -1.0, 1.0)
                    par ^= bit
                sgn[:, l, 7] = np.where(par == 1, -1.0, 1.0)   # odd parity closes the block
            out[:, sb * QK + ib * SUB:sb * QK + (ib + 1) * SUB] = \
                mag * sgn.reshape(len(rows), SUB) * db[:, None]
    return out.reshape(*packed.shape[:-1], K)


def bpw():
    return BLOCK_BYTES * 8 / QK


# ---- rotation (converter side of quant.h e8_signs / e8_rot_rows) -------------
#
# Q = D @ H / sqrt(n) per power-of-two block; W@Q on weight rows here equals
# Q^T x on activations in the engine (sign-flip then FWHT — one routine both
# sides). The sign diagonal D is REGENERATED from the block size, never stored:
# both sides draw the same xorshift64* stream seeded 417+n, and the kernel
# fixture (make_e8_fixture.py) pins the agreement.

def signs(n):
    """[n] float32 in {+1,-1} — bit-exact mirror of quant.h e8_signs()."""
    s = np.uint64(417 + n)
    out = np.empty((n + 7) // 8, dtype=np.uint8)
    with np.errstate(over="ignore"):
        for i in range(len(out)):
            s ^= s >> np.uint64(12)
            s ^= (s << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            s ^= s >> np.uint64(27)
            out[i] = np.uint8(((s * np.uint64(2685821657736338717)) &
                               np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(56))
    bits = np.unpackbits(out, bitorder="little")[:n]
    return np.where(bits == 1, -1.0, 1.0).astype(np.float32)


def rot_blocks(dim):
    """Block tiling: each block is the largest power of two dividing the
    remainder (its lowest set bit) — 6144 -> [2048, 4096], 1536 -> [512, 1024].
    Blocks over 32768 halve, mirroring the engine's sign-buffer cap."""
    out, rem = [], dim
    while rem:
        b = rem & (-rem)
        while b > 32768:
            b >>= 1
        out.append(b)
        rem -= b
    return out


def _fwht_rows(blk, b):
    """In-place-style FWHT over rows of [..., b] (b a power of two)."""
    h = 1
    while h < b:
        blk = blk.reshape(-1, b // (2 * h), 2, h)
        u, v = blk[:, :, 0, :].copy(), blk[:, :, 1, :].copy()
        blk[:, :, 0, :] = u + v
        blk[:, :, 1, :] = u - v
        blk = blk.reshape(-1, b)
        h <<= 1
    return blk / np.sqrt(b, dtype=np.float32)


def rotate_rows(x):
    """Apply the rotation to rows of float32 [..., dim] (weights W@Q or
    activations Q^T x — the transform is the same). Returns a new array."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    dim = x.shape[-1]
    rows = x.reshape(-1, dim).copy()
    off = 0
    for b in rot_blocks(dim):
        rows[:, off:off + b] = _fwht_rows(rows[:, off:off + b] * signs(b)[None, :], b)
        off += b
    return rows.reshape(x.shape)


def unrotate_rows(x):
    """Inverse of rotate_rows (Q v back to v): FWHT first, then the sign flip.
    Eval-side only — the engine always works in the rotated space."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    dim = x.shape[-1]
    rows = x.reshape(-1, dim).copy()
    off = 0
    for b in rot_blocks(dim):
        rows[:, off:off + b] = _fwht_rows(rows[:, off:off + b].copy(), b) * signs(b)[None, :]
        off += b
    return rows.reshape(x.shape)
