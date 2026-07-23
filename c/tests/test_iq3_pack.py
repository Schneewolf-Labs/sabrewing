"""fmt=6 codec oracle (#452 ladder step 2).

Checks the properties the container, the converter and the decode kernels all
depend on: exact byte budget, deterministic encode, decode agreeing with a
straight-from-the-spec reader, sign parity closure, and reconstruction quality
matching the ablation that chose this scheme (#453).
"""
import os
import sys
import unittest

# The runtime path is dependency-free by design and CI keeps it that way, so the
# offline-tooling tests skip rather than fail where numpy is absent.
np = None
P = None
try:
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import iq3_pack as P
except ImportError:  # pragma: no cover - exercised only on dependency-free CI
    pass


def ref_decode(packed, K):
    """Independent reader written straight from the layout comment — deliberately
    naive and loop-based, so a shared bug in the vectorized path shows up."""
    g = P.grid()
    nsb = K // P.QK
    rows = packed.reshape(-1, nsb * P.BLOCK_BYTES)
    out = np.zeros((len(rows), K), dtype=np.float32)
    for r in range(len(rows)):
        for sb in range(nsb):
            base = sb * P.BLOCK_BYTES
            d = float(rows[r, base + 96:base + 98].copy().view(np.float16)[0])
            for ib in range(P.QK // P.SUB):
                word = int(np.ascontiguousarray(
                    rows[r, base + 64 + ib * 4:base + 64 + ib * 4 + 4]).view(np.uint32)[0])
                db = d * (0.5 + ((word >> 28) & 0xF)) * 0.5
                for l in range(4):
                    seven = (word >> (7 * l)) & 0x7F
                    bits = [(seven >> j) & 1 for j in range(7)]
                    bits.append(sum(bits) & 1)          # odd parity closes the 8th
                    idx = int(rows[r, base + ib * 8 + l * 2 + 0])
                    idx2 = int(rows[r, base + ib * 8 + l * 2 + 1])
                    mags = list(g[idx]) + list(g[idx2])
                    for j in range(8):
                        pos = sb * P.QK + ib * P.SUB + l * 8 + j
                        out[r, pos] = mags[j] * db * (-1.0 if bits[j] else 1.0)
    return out


@unittest.skipIf(P is None, "numpy not available (offline-tooling test)")
class TestIq3Pack(unittest.TestCase):
    def setUp(self):
        np.random.seed(4242)
        self.x = (np.random.randn(6, 1024) * 0.05).astype(np.float32)

    def test_byte_budget(self):
        self.assertEqual(P.BLOCK_BYTES, 98)
        self.assertAlmostEqual(P.bpw(), 3.0625, places=6)
        packed = P.encode(self.x)
        self.assertEqual(packed.shape, (6, 1024 // P.QK * 98))
        self.assertEqual(packed.dtype, np.uint8)

    def test_encode_is_deterministic(self):
        self.assertTrue(np.array_equal(P.encode(self.x), P.encode(self.x)))

    def test_decode_matches_spec_reader(self):
        packed = P.encode(self.x)
        fast = P.decode(packed, 1024)
        slow = ref_decode(packed, 1024)
        self.assertTrue(np.allclose(fast, slow, rtol=1e-6, atol=1e-8),
                        f"max |Δ| = {np.abs(fast - slow).max()}")

    def test_sign_parity_closes(self):
        """Every 8-weight block must have an even number of negatives — that is
        what lets the 8th sign be derived instead of stored."""
        y = P.decode(P.encode(self.x), 1024)
        neg = (y < 0).reshape(-1, 8).sum(-1)
        self.assertTrue(np.all(neg % 2 == 0), "a block stored odd negatives")

    def test_reconstruction_quality(self):
        y = P.decode(P.encode(self.x), 1024)
        rel = np.sqrt(((y - self.x) ** 2).mean()) / np.sqrt((self.x ** 2).mean())
        # the torch model that won the #453 A/B measures ~0.195 on this input class
        self.assertLess(rel, 0.25, f"rel-RMSE {rel:.4f} — worse than the chosen scheme")
        self.assertGreater(rel, 0.05, f"rel-RMSE {rel:.4f} — implausibly good, check the test")

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            P.encode(np.zeros((2, 300), dtype=np.float32))

    def test_signs_spec_frozen(self):
        """First bytes of the n=256 stream, computed once by hand from the
        xorshift64* spec. If this fails, the PRNG drifted and every rotated
        container in the wild decodes to garbage — the constants are the spec."""
        s = P.signs(256)
        self.assertEqual(s.shape, (256,))
        self.assertTrue(set(np.unique(s)) <= {-1.0, 1.0})
        self.assertEqual(list((s[:16] < 0).astype(int)),
                         [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        self.assertFalse(np.array_equal(P.signs(256), P.signs(512)[:256]),
                         "streams for different sizes must differ (seed = 417+n)")

    def test_rot_blocks_tiling(self):
        self.assertEqual(P.rot_blocks(2048), [2048])
        self.assertEqual(P.rot_blocks(6144), [2048, 4096])
        self.assertEqual(P.rot_blocks(1536), [512, 1024])
        for dim in (768, 1536, 6144):
            self.assertEqual(sum(P.rot_blocks(dim)), dim)

    def test_rotation_is_orthogonal(self):
        """Q preserves norms and cancels between W@Q and Q^T x: the rotated
        matmul must reproduce the unrotated one to float precision."""
        w = (np.random.randn(8, 1536) * 0.05).astype(np.float32)
        x = np.random.randn(1536).astype(np.float32)
        wr, xr = P.rotate_rows(w), P.rotate_rows(x[None, :])[0]
        self.assertTrue(np.allclose(np.linalg.norm(wr, axis=1),
                                    np.linalg.norm(w, axis=1), rtol=1e-5))
        self.assertTrue(np.allclose(wr @ xr, w @ x, rtol=1e-4, atol=1e-5),
                        f"max |Δ| = {np.abs(wr @ xr - w @ x).max()}")

    def test_unrotate_inverts_rotate(self):
        w = (np.random.randn(4, 768) * 0.05).astype(np.float32)
        back = P.unrotate_rows(P.rotate_rows(w))
        self.assertTrue(np.allclose(back, w, rtol=1e-5, atol=1e-7),
                        f"max |Δ| = {np.abs(back - w).max()}")


if __name__ == "__main__":
    unittest.main()
