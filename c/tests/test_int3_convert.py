"""quant_int3_g64 (tools/convert_fp8_to_int4.py): pack layout + round-trip.

Decodes the packed bytes with an independent NumPy decoder implementing the
fmt=5 spec (16B low plane / 8B high plane per 64-group, v+4, per-group f32
scale) and checks the dequantized result equals the reference
quantize-dequantize (same math as quant_ablation._quant_last_dim(3, 64)).
The C side of the same layout is covered by tests/test_int3.c.
"""
import os, sys, unittest
try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("numpy not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from convert_fp8_to_int4 import quant_int3_g64


def decode(packed, scales, O, I, group=64):
    ng = (I + group - 1) // group
    b = packed.reshape(O, ng, 24)
    lo, hi = b[:, :, :16], b[:, :, 16:]
    k = np.arange(group)
    lov = (lo[:, :, k >> 2] >> ((k & 3) * 2)[None, None, :]) & 3
    hiv = (hi[:, :, k >> 3] >> (k & 7)[None, None, :]) & 1
    v = (lov | (hiv << 2)).astype(np.int64) - 4
    dq = v.astype(np.float64) * scales.reshape(O, ng, 1).astype(np.float64)
    return dq.reshape(O, ng * group)[:, :I]


def reference(w, group=64):
    """same math as quant_int3_g64 (which works in f32), replayed exactly, then
    dequantized in f64 so it matches decode() bit for bit"""
    O, I = w.shape
    ng = (I + group - 1) // group
    pad = ng * group - I
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = wp.reshape(O, ng, group)
    s = np.maximum(np.abs(g).max(axis=2, keepdims=True) / 3.0, 1e-8).astype(np.float32)
    q = np.clip(np.rint(g / s), -4, 3).astype(np.int64)
    return (q.astype(np.float64) * s.astype(np.float64)).reshape(O, ng * group)[:, :I]


class Int3ConvertTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(7)
        for I in (64, 128, 100, 65, 7168):
            w = (rng.standard_normal((5, I)) * 0.05).astype(np.float32)
            w[0, 3] = 1.7; w[2, min(5, I - 1)] = -2.2
            packed, scales = quant_int3_g64(w)
            ng = (I + 63) // 64
            self.assertEqual(packed.size, 5 * ng * 24)
            self.assertEqual(scales.size, 5 * ng)
            np.testing.assert_allclose(decode(packed, scales, 5, I),
                                       reference(w), rtol=0, atol=0)

    def test_outliers_beat_row_int4(self):
        rng = np.random.default_rng(11)
        w = (rng.standard_normal((8, 1024)) * 0.02).astype(np.float32)
        for o in range(8): w[o, (o * 37) % 1024] = 1.5
        packed, scales = quant_int3_g64(w)
        e3 = float(((decode(packed, scales, 8, 1024) - w) ** 2).mean())
        s4 = np.maximum(np.abs(w).max(axis=1, keepdims=True) / 7.0, 1e-8)
        w4 = np.clip(np.rint(w / s4), -8, 7) * s4
        e4 = float(((w4 - w) ** 2).mean())
        self.assertLess(e3, e4)


if __name__ == "__main__":
    unittest.main()
