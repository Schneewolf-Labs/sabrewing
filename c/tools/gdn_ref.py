#!/usr/bin/env python3
"""Dump a reference for Qwen3.5's gated delta net (the linear-attention layer).

The novel part of qwen3_5_moe is not the MoE — that maps onto our existing descriptor —
it is the GatedDeltaNet token mixer used by 30 of 40 layers. This writes a fixture with
random weights/inputs and the outputs transformers computes, so the C implementation can be
checked against the reference *before* an engine exists to run it in. Same idea as
kernel_check.c: validate the piece, not the whole model.

Usage: python3 tools/gdn_ref.py [out.npz]
"""
import sys

import numpy as np
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    torch_recurrent_gated_delta_rule,
)


class Cfg:
    """Small stand-in for Qwen3_5MoeTextConfig with the real head geometry ratios."""

    hidden_size = 64
    linear_num_key_heads = 2
    linear_num_value_heads = 4          # 2 value heads per key head, as in the 35B
    linear_key_head_dim = 16
    linear_value_head_dim = 16
    linear_conv_kernel_dim = 4
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    dtype = torch.float32


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "gdn_ref.npz"
    torch.manual_seed(1234)
    cfg = Cfg()
    layer = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).to(torch.float32).eval()

    # random but bounded weights: the recurrence is sensitive to the decay term, and
    # default zero-init on some projections would hide sign errors
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name in ("A_log", "dt_bias"):
                continue
            p.copy_(torch.randn_like(p) * 0.15)

    seq = 12
    x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float32) * 0.5

    with torch.no_grad():
        y = layer(hidden_states=x, cache_params=None, cache_position=None, attention_mask=None)

        # Also dump the recurrence in isolation (post-conv, post-projection), so a C bug can
        # be localized to the projections/conv or to the delta rule itself.
        mixed = layer.in_proj_qkv(x).transpose(1, 2)
        mixed = torch.nn.functional.silu(layer.conv1d(mixed)[:, :, :seq]).transpose(1, 2)
        q, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
        q = q.reshape(1, seq, -1, layer.head_k_dim)
        k = k.reshape(1, seq, -1, layer.head_k_dim)
        v = v.reshape(1, seq, -1, layer.head_v_dim)
        b = layer.in_proj_b(x)
        a = layer.in_proj_a(x)
        beta = b.sigmoid()
        g = -layer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer.dt_bias)
        rep = layer.num_v_heads // layer.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        core, _ = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        z = layer.in_proj_z(x).reshape(1, seq, -1, layer.head_v_dim)

    np.savez(
        out,
        # geometry
        hidden=np.int32(cfg.hidden_size), seq=np.int32(seq),
        n_k_heads=np.int32(cfg.linear_num_key_heads), n_v_heads=np.int32(cfg.linear_num_value_heads),
        k_dim=np.int32(cfg.linear_key_head_dim), v_dim=np.int32(cfg.linear_value_head_dim),
        conv_k=np.int32(cfg.linear_conv_kernel_dim), eps=np.float32(cfg.rms_norm_eps),
        # weights
        w_qkv=layer.in_proj_qkv.weight.detach().numpy(),
        w_z=layer.in_proj_z.weight.detach().numpy(),
        w_b=layer.in_proj_b.weight.detach().numpy(),
        w_a=layer.in_proj_a.weight.detach().numpy(),
        conv_w=layer.conv1d.weight.detach().numpy().squeeze(1),
        conv_b=(layer.conv1d.bias.detach().numpy() if layer.conv1d.bias is not None
                else np.zeros(layer.conv_dim, dtype=np.float32)),
        dt_bias=layer.dt_bias.detach().numpy(),
        A_log=layer.A_log.detach().numpy(),
        norm_w=layer.norm.weight.detach().numpy(),
        w_out=layer.out_proj.weight.detach().numpy(),
        # input and expected outputs
        x=x.numpy(),
        core_ref=core.numpy(),      # recurrence output, pre gated-norm
        z_ref=z.numpy(),
        y_ref=y.numpy(),            # full layer output
    )
    print(f"wrote {out}: seq={seq} hidden={cfg.hidden_size} "
          f"k_heads={cfg.linear_num_key_heads} v_heads={cfg.linear_num_value_heads} "
          f"k_dim={cfg.linear_key_head_dim} v_dim={cfg.linear_value_head_dim}")
    print(f"  |y_ref| mean={np.abs(y.numpy()).mean():.5f} max={np.abs(y.numpy()).max():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
