"""Tests for the heat-tiered scheme path in tools/quant_ablation.py
(docs/heat-quant-design.md: rank experts by routing mass, hot -> int4, cold -> int2-gN).

torch is a dev-only dependency of the ablation tool (the engine stays dependency-free),
so this module skips itself when torch is absent — same stance as the tool.
"""
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("torch not installed (dev-only dep of quant_ablation.py)")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import quant_ablation as qa


class TestParse(unittest.TestCase):
    def test_heat_grammar(self):
        self.assertEqual(qa.parse_heat("heat10-int4+int2-g64"), (10, (4, 0), (2, 64)))
        self.assertEqual(qa.parse_heat("heat2-int4-g32+int3"), (2, (4, 32), (3, 0)))
        self.assertIsNone(qa.parse_heat("int4-g64"))       # uniform schemes fall through
        self.assertIsNone(qa.parse_heat("heat10-bogus"))
        with self.assertRaises(SystemExit):                 # main()'s fail-fast path
            qa.parse_scheme("heat10-bogus")


# A minimal module tree with the names the tool keys on: model.layers.<L>.mlp.gate
# (router, must stay float) and model.layers.<L>.mlp.experts.* in BOTH layouts the
# tool supports — fused 3D [E, in, out] (transformers 5.x / GLM) and per-expert 2D
# nn.Linear (older transformers OLMoE).
E, D, I = 4, 16, 8


class FusedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, E, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(E, D, 2 * I))
        self.experts.down_proj = nn.Parameter(torch.randn(E, I, D))


class PerExpertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, E, bias=False)
        self.experts = nn.ModuleList(
            [nn.ModuleDict({"gate_proj": nn.Linear(D, I, bias=False),
                            "down_proj": nn.Linear(I, D, bias=False)}) for _ in range(E)])


def make_model(mlp_cls):
    layer = nn.Module()
    layer.mlp = mlp_cls()
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    model.config = types.SimpleNamespace(num_experts_per_tok=2)

    def forward(ids):
        g = torch.Generator().manual_seed(int(ids.sum()))
        x = torch.randn(ids.shape[-1], D, generator=g)
        return layer.mlp.gate(x)

    model.forward = forward
    return model


class StubTok:
    def __call__(self, text, add_special_tokens=False):
        g = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
        return types.SimpleNamespace(input_ids=torch.randint(0, 99, (16,), generator=g).tolist())


class TestCalibration(unittest.TestCase):
    def test_mass_shape_and_bounds(self):
        torch.manual_seed(0)
        model = make_model(FusedMoE)
        mass = qa.measure_expert_mass(model, StubTok(), [f"c{i}" for i in range(4)], "cpu")
        self.assertEqual(list(mass), [0])
        m = mass[0]
        self.assertEqual(m.shape, (E,))
        self.assertGreater(m.sum().item(), 0)
        # 4 ctxs * 16 tokens, top-2 softmax probs sum to at most 1 per token
        self.assertLessEqual(m.sum().item(), 64 + 1e-6)

    def test_no_gate_is_loud(self):
        model = nn.Linear(4, 4)
        model.config = types.SimpleNamespace(num_experts_per_tok=2)
        with self.assertRaises(SystemExit):
            qa.measure_expert_mass(model, StubTok(), ["c"], "cpu")


class TestApplyHeat(unittest.TestCase):
    # hand-built mass: hot = {0, 2}
    MASS = {0: torch.tensor([5.0, 1.0, 4.0, 0.5], dtype=torch.float64)}

    def check_router_untouched(self, model, before):
        name = "model.layers.0.mlp.gate.weight"
        self.assertTrue(torch.equal(dict(model.named_parameters())[name], before[name]))

    def test_fused_3d(self):
        torch.manual_seed(1)
        model = make_model(FusedMoE)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        qa.apply_heat_scheme(model, "heat2-int4+int2-g8", self.MASS)
        params = dict(model.named_parameters())
        for name in ("model.layers.0.mlp.experts.gate_up_proj",
                     "model.layers.0.mlp.experts.down_proj"):
            qh = qa.quantize_param(before[name].float(), 4, 0)
            qc = qa.quantize_param(before[name].float(), 2, 8)
            for e in range(E):
                want = qh[e] if e in (0, 2) else qc[e]
                self.assertTrue(torch.equal(params[name][e], want), f"{name}[{e}] wrong tier")
        self.check_router_untouched(model, before)

    def test_per_expert_2d(self):
        torch.manual_seed(2)
        model = make_model(PerExpertMLP)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        qa.apply_heat_scheme(model, "heat2-int4+int2-g8", self.MASS)
        for name, p in model.named_parameters():
            em = qa.EXPERT_NAME_RE.search(name)
            if not em:
                continue
            cold = int(em.group(1)) not in (0, 2)
            want = qa.quantize_param(before[name].float(), *((2, 8) if cold else (4, 0)))
            self.assertTrue(torch.equal(p, want.to(p.dtype)), f"{name} wrong tier (cold={cold})")
        self.check_router_untouched(model, before)


class TestTinyOlmoe(unittest.TestCase):
    """End-to-end on a real (tiny, random) OLMoE: hooks find the real gate modules,
    the fused expert tensors get per-expert tiers, everything else goes hot."""

    def test_end_to_end(self):
        try:
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("transformers without OLMoE support")
        torch.manual_seed(0)
        cfg = OlmoeConfig(hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
                          num_experts=8, num_experts_per_tok=2, max_position_embeddings=64,
                          eos_token_id=None)
        model = OlmoeForCausalLM(cfg).eval()

        class Tok:
            def __call__(self, text, add_special_tokens=False):
                g = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
                return types.SimpleNamespace(
                    input_ids=torch.randint(0, 128, (16,), generator=g).tolist())

        mass = qa.measure_expert_mass(model, Tok(), [f"ctx{i}" for i in range(4)], "cpu")
        self.assertEqual(sorted(mass), [0, 1])
        hot = {L: set(m.topk(2).indices.tolist()) for L, m in mass.items()}

        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        n, qp, tp = qa.apply_heat_scheme(model, "heat2-int4+int2-g16", mass)
        self.assertGreater(100 * qp / tp, 90)       # the coverage bar main() enforces

        for name, p in model.named_parameters():
            if p.ndim < 2 or qa.is_router(name):
                self.assertTrue(torch.equal(p, before[name]), f"{name} should be untouched")
            elif p.ndim == 3 and ".experts." in name:
                L = int(qa.LAYER_RE.search(name).group(1))
                qh = qa.quantize_param(before[name].float(), 4, 0).to(p.dtype)
                qc = qa.quantize_param(before[name].float(), 2, 16).to(p.dtype)
                for e in range(p.shape[0]):
                    want = qh[e] if e in hot[L] else qc[e]
                    self.assertTrue(torch.equal(p[e], want), f"{name}[{e}] wrong tier")
            else:
                want = qa.quantize_param(before[name].float(), 4, 0).to(p.dtype)
                self.assertTrue(torch.equal(p, want), f"{name} should be hot int4")


if __name__ == "__main__":
    unittest.main()
