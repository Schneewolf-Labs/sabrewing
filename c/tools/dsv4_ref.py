#!/usr/bin/env python3
"""Build a tiny DeepSeek-V4 snapshot with random weights and a pure-Python reference
forward pass, so deepseek.c can be cross-checked WITHOUT torch, transformers or a GPU.

Why this exists alongside tools/make_tiny_deepseek.py: the transformers oracle is the real
gate, but it needs a box with torch + a transformers carrying `deepseek_v4`. This script
needs nothing but CPython, so the V4-only machinery — mHC + Sinkhorn-Knopp, the CSA
two-series overlap windowing, the HCA compressor, the Lightning Indexer's top-k with its
causal `-1` sentinel, shared-KV attention with sinks and the conjugate output rotation,
grouped output projection, hash-MoE vs top-k routing, and YaRN on the compress rope — is
exercised on every machine, including CI.

What it does and does NOT prove:
  DOES  — catch transcription bugs in deepseek.c: the two implementations are written
          independently against the same reference semantics, and disagree loudly.
  DOES  — catch prefill/decode divergence: `full_ids` is produced by re-running the whole
          prefix from scratch each step, so the C engine's incremental compressor buffers,
          sliding ring and indexer state have to reproduce a stateless recompute.
  DOES NOT — prove agreement with transformers. A shared misreading of the reference would
          satisfy both. Only `make deepseek-test` settles that.

Writes <out>/config.json, <out>/model.safetensors and <out>/ref_deepseek.json, in the same
format the transformers oracle uses, so:  SNAP=<out> ./deepseek <out>/ref_deepseek.json

Usage: python3 tools/dsv4_ref.py <outdir>
"""
import json
import math
import random
import struct
import sys

# ---------------------------------------------------------------- tiny architecture
V, D, NL = 64, 24, 4
HEADS, HD, QLORA = 2, 16, 16
ROPE_DIM = 4                      # partial_rotary_factor = 4/16
O_GROUPS, O_RANK = 2, 8
E, K, MOE_I = 6, 2, 16
HC, SINK_ITERS, HC_EPS = 4, 20, 1.0e-6
IDX_HEADS, IDX_DIM, IDX_TOPK = 2, 8, 2
WIN = 6
M_CSA, M_HCA = 2, 4
EPS = 1.0e-6
LIMIT = 10.0
ROUTE_SCALE = 1.5
THETA_MAIN, THETA_COMP = 10000.0, 160000.0
YARN_FACTOR, YARN_ORIG = 4.0, 16
BETA_FAST, BETA_SLOW = 32.0, 1.0

LAYER_TYPES = ["heavily_compressed_attention", "compressed_sparse_attention",
               "sliding_attention", "compressed_sparse_attention"]
MLP_TYPES = ["hash_moe", "moe", "hash_moe", "moe"]

rng = random.Random(20260731)


def rnd(n, s=0.12):
    return [rng.gauss(0.0, 1.0) * s for _ in range(n)]


def rnd_norm(n):
    return [1.0 + rng.gauss(0.0, 1.0) * 0.1 for _ in range(n)]


# ---------------------------------------------------------------- linear algebra
def matvec(W, x, out_n, in_n):
    """y[o] = sum_i W[o*in_n+i]*x[i] — HF Linear convention (W is [out, in], row-major)."""
    return [sum(W[o * in_n + i] * x[i] for i in range(in_n)) for o in range(out_n)]


def rms(x, w, eps=EPS):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * inv * w[i] for i, v in enumerate(x)]


def rms_plain(x, eps=EPS):
    inv = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * inv for v in x]


def softmax(v):
    m = max(v)
    e = [math.exp(t - m) if t != float("-inf") else 0.0 for t in v]
    s = sum(e)
    return [t / s for t in e]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x / (1.0 + math.exp(-x))


def softplus(x):
    return x if x > 20.0 else math.log1p(math.exp(x))


# ---------------------------------------------------------------- rope
def inv_default(theta, dim):
    return [1.0 / (theta ** ((2 * j) / dim)) for j in range(dim // 2)]


def inv_yarn(theta, dim, factor, orig, beta_fast, beta_slow):
    """transformers' _compute_yarn_parameters: blend extrapolation and interpolation
    frequencies over a linear ramp between the two correction dims."""
    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(theta))
    lo, hi = math.floor(corr(beta_fast)), math.ceil(corr(beta_slow))
    lo, hi = max(lo, 0), min(hi, dim - 1)
    if hi == lo:
        hi = lo + 0.001
    out = []
    for j in range(dim // 2):
        pf = theta ** ((2 * j) / dim)
        extrap, interp = 1.0 / pf, 1.0 / (factor * pf)
        ramp = min(1.0, max(0.0, (j - lo) / (hi - lo)))
        ext = 1.0 - ramp
        out.append(interp * (1 - ext) + extrap * ext)
    return out


INV_MAIN = inv_default(THETA_MAIN, ROPE_DIM)
INV_COMP = inv_yarn(THETA_COMP, ROPE_DIM, YARN_FACTOR, YARN_ORIG, BETA_FAST, BETA_SLOW)

# YaRN's attention_factor (mscale) multiplies BOTH cos and sin, so it is a per-rotation
# GAIN, not just a phase. transformers computes 0.1*ln(factor)+1 when the config does not
# override it. This harness used to pin it to 1.0 in the config it emits, which is why it
# could not catch the engine dropping it: with mscale == 1 the two behaviours coincide.
# The real V4-Flash compress rope has factor 16 -> 1.2773, so exercise a non-unit value.
MS_MAIN = 1.0
MS_COMP = 0.1 * math.log(YARN_FACTOR) + 1.0 if YARN_FACTOR > 1 else 1.0


def rope(vec, pos, inv, sgn=1.0, ms=1.0):
    """Interleaved RoPE on the TRAILING len(inv)*2 channels; leading channels pass through."""
    rd = len(inv) * 2
    out = list(vec)
    base = len(vec) - rd
    for j in range(len(inv)):
        ang = pos * inv[j]
        c, s = math.cos(ang) * ms, sgn * math.sin(ang) * ms
        a, b = out[base + 2 * j], out[base + 2 * j + 1]
        out[base + 2 * j] = a * c - b * s
        out[base + 2 * j + 1] = b * c + a * s
    return out


# ---------------------------------------------------------------- weights
def build_weights():
    W = {}
    W["model.embed_tokens.weight"] = rnd(V * D)
    W["model.norm.weight"] = rnd_norm(D)
    W["lm_head.weight"] = rnd(V * D)
    W["model.hc_head.hc_fn"] = rnd(HC * HC * D, 0.5)
    W["model.hc_head.hc_base"] = rnd(HC, 0.8)
    W["model.hc_head.hc_scale"] = rnd(1, 0.8)

    for i in range(NL):
        p = f"model.layers.{i}."
        W[p + "input_layernorm.weight"] = rnd_norm(D)
        W[p + "post_attention_layernorm.weight"] = rnd_norm(D)
        W[p + "self_attn.q_a_proj.weight"] = rnd(QLORA * D)
        W[p + "self_attn.q_a_norm.weight"] = rnd_norm(QLORA)
        W[p + "self_attn.q_b_proj.weight"] = rnd(HEADS * HD * QLORA)
        W[p + "self_attn.kv_proj.weight"] = rnd(HD * D)
        W[p + "self_attn.kv_norm.weight"] = rnd_norm(HD)
        W[p + "self_attn.o_a_proj.weight"] = rnd(O_GROUPS * O_RANK * (HEADS * HD // O_GROUPS))
        W[p + "self_attn.o_b_proj.weight"] = rnd(D * O_GROUPS * O_RANK)
        W[p + "self_attn.sinks"] = rnd(HEADS, 0.5)
        for site in ("attn_hc", "ffn_hc"):
            W[p + site + ".fn"] = rnd((2 + HC) * HC * HC * D, 0.5)
            W[p + site + ".base"] = rnd((2 + HC) * HC, 0.8)
            W[p + site + ".scale"] = rnd(3, 0.8)
        W[p + "mlp.gate.weight"] = rnd(E * D)
        if MLP_TYPES[i] == "hash_moe":
            W[p + "mlp.gate.tid2eid"] = [rng.randrange(E) for _ in range(V * K)]
        else:
            W[p + "mlp.gate.e_score_correction_bias"] = rnd(E, 0.3)
        W[p + "mlp.experts.gate_up_proj"] = rnd(E * 2 * MOE_I * D)
        W[p + "mlp.experts.down_proj"] = rnd(E * D * MOE_I)
        W[p + "mlp.shared_experts.gate_proj.weight"] = rnd(MOE_I * D)
        W[p + "mlp.shared_experts.up_proj.weight"] = rnd(MOE_I * D)
        W[p + "mlp.shared_experts.down_proj.weight"] = rnd(D * MOE_I)
        lt = LAYER_TYPES[i]
        if lt != "sliding_attention":
            m = M_CSA if lt == "compressed_sparse_attention" else M_HCA
            wid = 2 * HD if lt == "compressed_sparse_attention" else HD
            W[p + "self_attn.compressor.kv_proj.weight"] = rnd(wid * D)
            W[p + "self_attn.compressor.gate_proj.weight"] = rnd(wid * D)
            W[p + "self_attn.compressor.position_bias"] = rnd(m * wid, 0.5)
            W[p + "self_attn.compressor.kv_norm.weight"] = rnd_norm(HD)
        if lt == "compressed_sparse_attention":
            q = p + "self_attn.compressor.indexer."
            W[q + "kv_proj.weight"] = rnd(2 * IDX_DIM * D)
            W[q + "gate_proj.weight"] = rnd(2 * IDX_DIM * D)
            W[q + "position_bias"] = rnd(M_CSA * 2 * IDX_DIM, 0.5)
            W[q + "kv_norm.weight"] = rnd_norm(IDX_DIM)
            W[q + "q_b_proj.weight"] = rnd(IDX_HEADS * IDX_DIM * QLORA)
            W[q + "scorer.weights_proj.weight"] = rnd(IDX_HEADS * D)
    return W


SHAPES = {}


def shape_of(name, n):
    """Shapes only matter to safetensors' header bookkeeping and to any tool that reads
    the snapshot back; the engine indexes flat, so a faithful 2D/3D shape is enough."""
    if name in SHAPES:
        return SHAPES[name]
    return [n]


def register_shapes():
    S = SHAPES
    S["model.embed_tokens.weight"] = [V, D]
    S["lm_head.weight"] = [V, D]
    S["model.hc_head.hc_fn"] = [HC, HC * D]
    for i in range(NL):
        p = f"model.layers.{i}."
        S[p + "self_attn.q_a_proj.weight"] = [QLORA, D]
        S[p + "self_attn.q_b_proj.weight"] = [HEADS * HD, QLORA]
        S[p + "self_attn.kv_proj.weight"] = [HD, D]
        S[p + "self_attn.o_a_proj.weight"] = [O_GROUPS * O_RANK, HEADS * HD // O_GROUPS]
        S[p + "self_attn.o_b_proj.weight"] = [D, O_GROUPS * O_RANK]
        for site in ("attn_hc", "ffn_hc"):
            S[p + site + ".fn"] = [(2 + HC) * HC, HC * D]
        S[p + "mlp.gate.weight"] = [E, D]
        if MLP_TYPES[i] == "hash_moe":
            S[p + "mlp.gate.tid2eid"] = [V, K]
        S[p + "mlp.experts.gate_up_proj"] = [E, 2 * MOE_I, D]
        S[p + "mlp.experts.down_proj"] = [E, D, MOE_I]
        S[p + "mlp.shared_experts.gate_proj.weight"] = [MOE_I, D]
        S[p + "mlp.shared_experts.up_proj.weight"] = [MOE_I, D]
        S[p + "mlp.shared_experts.down_proj.weight"] = [D, MOE_I]
        lt = LAYER_TYPES[i]
        if lt != "sliding_attention":
            m = M_CSA if lt == "compressed_sparse_attention" else M_HCA
            wid = 2 * HD if lt == "compressed_sparse_attention" else HD
            S[p + "self_attn.compressor.kv_proj.weight"] = [wid, D]
            S[p + "self_attn.compressor.gate_proj.weight"] = [wid, D]
            S[p + "self_attn.compressor.position_bias"] = [m, wid]
        if lt == "compressed_sparse_attention":
            q = p + "self_attn.compressor.indexer."
            S[q + "kv_proj.weight"] = [2 * IDX_DIM, D]
            S[q + "gate_proj.weight"] = [2 * IDX_DIM, D]
            S[q + "position_bias"] = [M_CSA, 2 * IDX_DIM]
            S[q + "q_b_proj.weight"] = [IDX_HEADS * IDX_DIM, QLORA]
            S[q + "scorer.weights_proj.weight"] = [IDX_HEADS, D]


def write_safetensors(path, W):
    register_shapes()
    header, blobs, off = {}, [], 0
    for name in sorted(W):
        vals = W[name]
        if name.endswith("tid2eid"):
            raw = struct.pack("<%dq" % len(vals), *vals)
            dt = "I64"
        else:
            raw = struct.pack("<%df" % len(vals), *[float(v) for v in vals])
            dt = "F32"
        header[name] = {"dtype": dt, "shape": shape_of(name, len(vals)),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def write_config(path):
    cfg = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "vocab_size": V, "hidden_size": D, "num_hidden_layers": NL,
        "num_attention_heads": HEADS, "num_key_value_heads": 1, "head_dim": HD,
        "q_lora_rank": QLORA, "partial_rotary_factor": ROPE_DIM / HD,
        "moe_intermediate_size": MOE_I, "n_routed_experts": E, "n_shared_experts": 1,
        "num_experts_per_tok": K, "scoring_func": "sqrtsoftplus", "norm_topk_prob": True,
        "routed_scaling_factor": ROUTE_SCALE, "swiglu_limit": LIMIT,
        "layer_types": LAYER_TYPES, "mlp_layer_types": MLP_TYPES,
        "compress_rates": {"compressed_sparse_attention": M_CSA,
                           "heavily_compressed_attention": M_HCA},
        "sliding_window": WIN, "o_groups": O_GROUPS, "o_lora_rank": O_RANK,
        "index_n_heads": IDX_HEADS, "index_head_dim": IDX_DIM, "index_topk": IDX_TOPK,
        "hc_mult": HC, "hc_sinkhorn_iters": SINK_ITERS, "hc_eps": HC_EPS,
        "rms_norm_eps": EPS, "hidden_act": "silu",
        "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "rope_theta": THETA_MAIN, "compress_rope_theta": THETA_COMP,
        "rope_parameters": {
            "main": {"rope_type": "default", "rope_theta": THETA_MAIN,
                     "partial_rotary_factor": ROPE_DIM / HD},
            "compress": {"rope_type": "yarn", "rope_theta": THETA_COMP,
                         "factor": YARN_FACTOR, "beta_fast": BETA_FAST,
                         "beta_slow": BETA_SLOW,
                         "original_max_position_embeddings": YARN_ORIG,
                         "partial_rotary_factor": ROPE_DIM / HD},
        },
        "eos_token_id": None, "bos_token_id": 0,
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)


# ---------------------------------------------------------------- the model
def hc_site(W, prefix, streams):
    """mHC mapping: returns (post, comb, collapsed) for one token's HC streams."""
    fn, base, scale = W[prefix + ".fn"], W[prefix + ".base"], W[prefix + ".scale"]
    flat = rms_plain([v for s in streams for v in s])
    mix = matvec(fn, flat, (2 + HC) * HC, HC * D)
    pre = [sigmoid(mix[k] * scale[0] + base[k]) + HC_EPS for k in range(HC)]
    post = [2 * sigmoid(mix[HC + k] * scale[1] + base[HC + k]) for k in range(HC)]

    comb = []
    for j in range(HC):
        row = [mix[2 * HC + j * HC + k] * scale[2] + base[2 * HC + j * HC + k] for k in range(HC)]
        comb.append([v + HC_EPS for v in softmax(row)])
    # Sinkhorn-Knopp: first a COLUMN normalisation, then (iters-1) x (row, column).
    for k in range(HC):
        s = sum(comb[j][k] for j in range(HC)) + HC_EPS
        for j in range(HC):
            comb[j][k] /= s
    for _ in range(SINK_ITERS - 1):
        for j in range(HC):
            s = sum(comb[j]) + HC_EPS
            comb[j] = [v / s for v in comb[j]]
        for k in range(HC):
            s = sum(comb[j][k] for j in range(HC)) + HC_EPS
            for j in range(HC):
                comb[j][k] /= s
    collapsed = [sum(pre[k] * streams[k][i] for k in range(HC)) for i in range(D)]
    return post, comb, collapsed


def hc_merge(streams, post, comb, y):
    """streams[k] = post[k]*y + sum_j comb[j][k]*streams[j]  (comb consumed transposed)."""
    return [[post[k] * y[i] + sum(comb[j][k] * streams[j][i] for j in range(HC))
             for i in range(D)] for k in range(HC)]


def compress(W, prefix, hidden, m, dim, overlap, norm_key):
    """Emit one compressed entry per closed window of `m` source tokens.

    overlap=False (HCA): each window pools its own `m` tokens.
    overlap=True  (CSA/indexer): kv_proj emits 2*dim per token — Ca | Cb. Entry w pools
    window w-1's Ca with window w's Cb over 2m slots (width 2m, stride m); window 0's
    first half has no predecessor, so it gets kv 0 / gate -inf, i.e. softmax weight 0.
    """
    wid = 2 * dim if overlap else dim
    kvp, gtp = W[prefix + "kv_proj.weight"], W[prefix + "gate_proj.weight"]
    bias, nw = W[prefix + "position_bias"], W[prefix + norm_key]
    kv = [matvec(kvp, h, wid, D) for h in hidden]
    gate = [matvec(gtp, h, wid, D) for h in hidden]
    nwin = len(hidden) // m
    out = []
    for w in range(nwin):
        if not overlap:
            sk = [kv[w * m + j][:dim] for j in range(m)]
            sg = [[gate[w * m + j][d] + bias[j * wid + d] for d in range(dim)] for j in range(m)]
        else:
            first_k, first_g = [], []
            for j in range(m):
                if w > 0:
                    first_k.append(kv[(w - 1) * m + j][:dim])
                    first_g.append([gate[(w - 1) * m + j][d] + bias[j * wid + d] for d in range(dim)])
                else:
                    first_k.append([0.0] * dim)
                    first_g.append([float("-inf")] * dim)
            sk = first_k + [kv[w * m + j][dim:] for j in range(m)]
            sg = first_g + [[gate[w * m + j][dim + d] + bias[j * wid + dim + d]
                             for d in range(dim)] for j in range(m)]
        slots = len(sk)
        # per-CHANNEL softmax across the window slots, then the gated sum
        ent = []
        for d in range(dim):
            p = softmax([sg[j][d] for j in range(slots)])
            ent.append(sum(p[j] * sk[j][d] for j in range(slots)))
        out.append(rope(rms(ent, nw), w * m, INV_COMP, ms=MS_COMP))
    return out


def attention(W, i, hidden, positions):
    p = f"model.layers.{i}."
    lt = LAYER_TYPES[i]
    inv = INV_MAIN if lt == "sliding_attention" else INV_COMP
    ms = MS_MAIN if lt == "sliding_attention" else MS_COMP
    n = len(hidden)

    qres = [rms(matvec(W[p + "self_attn.q_a_proj.weight"], h, QLORA, D),
                W[p + "self_attn.q_a_norm.weight"]) for h in hidden]
    q = []
    for t in range(n):
        full = matvec(W[p + "self_attn.q_b_proj.weight"], qres[t], HEADS * HD, QLORA)
        q.append([rope(rms_plain(full[h * HD:(h + 1) * HD]), positions[t], inv, ms=ms)
                  for h in range(HEADS)])
    kv = [rope(rms(matvec(W[p + "self_attn.kv_proj.weight"], h, HD, D),
                   W[p + "self_attn.kv_norm.weight"]), positions[t], inv, ms=ms)
          for t, h in enumerate(hidden)]

    comp, vis = [], [[] for _ in range(n)]
    if lt != "sliding_attention":
        m = M_CSA if lt == "compressed_sparse_attention" else M_HCA
        comp = compress(W, p + "self_attn.compressor.", hidden, m, HD,
                        lt == "compressed_sparse_attention", "kv_norm.weight")
        if lt == "heavily_compressed_attention":
            for t in range(n):
                thr = (positions[t] + 1) // m
                vis[t] = [e for e in range(len(comp)) if e < thr]
        else:
            qi = p + "self_attn.compressor.indexer."
            ic = compress(W, qi, hidden, m, IDX_DIM, True, "kv_norm.weight")
            wsc, ssc = IDX_HEADS ** -0.5, IDX_DIM ** -0.5
            for t in range(n):
                iq = matvec(W[qi + "q_b_proj.weight"], qres[t], IDX_HEADS * IDX_DIM, QLORA)
                iq = [rope(iq[h * IDX_DIM:(h + 1) * IDX_DIM], positions[t], INV_COMP, ms=MS_COMP)
                      for h in range(IDX_HEADS)]
                iw = matvec(W[qi + "scorer.weights_proj.weight"], hidden[t], IDX_HEADS, D)
                thr = (positions[t] + 1) // m
                sc = []
                for e in range(len(ic)):
                    a = 0.0
                    for h in range(IDX_HEADS):
                        dot = sum(iq[h][d] * ic[e][d] for d in range(IDX_DIM))
                        if dot > 0:
                            a += (iw[h] * wsc) * (dot * ssc)
                    sc.append(a if e < thr else float("-inf"))
                order = sorted(range(len(sc)), key=lambda e: (-sc[e], e))
                vis[t] = sorted(e for e in order[:min(IDX_TOPK, len(sc))] if e < thr)

    scale = HD ** -0.5
    sinks = W[p + "self_attn.sinks"]
    out_rows = []
    for t in range(n):
        pos = positions[t]
        heads = []
        for h in range(HEADS):
            keys = [(kv[j], j) for j in range(n) if positions[j] <= pos and pos - positions[j] < WIN]
            keys += [(comp[e], None) for e in vis[t]]
            logits = [sum(q[t][h][d] * kk[d] for d in range(HD)) * scale for kk, _ in keys]
            mx = max(logits + [sinks[h]])
            ex = [math.exp(v - mx) for v in logits]
            den = sum(ex) + math.exp(sinks[h] - mx)          # the sink is dropped after
            ah = [sum(ex[j] / den * keys[j][0][d] for j in range(len(keys))) for d in range(HD)]
            heads.append(rope(ah, pos, inv, sgn=-1.0, ms=ms))       # K == V: undo the value rotation
        flat = [v for hv in heads for v in hv]
        gin = HEADS * HD // O_GROUPS
        grp = []
        for g in range(O_GROUPS):
            grp += matvec(W[p + "self_attn.o_a_proj.weight"][g * O_RANK * gin:(g + 1) * O_RANK * gin],
                          flat[g * gin:(g + 1) * gin], O_RANK, gin)
        out_rows.append(matvec(W[p + "self_attn.o_b_proj.weight"], grp, D, O_GROUPS * O_RANK))
    return out_rows


def moe(W, i, x, tok):
    p = f"model.layers.{i}."
    logits = matvec(W[p + "mlp.gate.weight"], x, E, D)
    score = [math.sqrt(softplus(v)) for v in logits]
    if MLP_TYPES[i] == "hash_moe":
        sel = list(W[p + "mlp.gate.tid2eid"][tok * K:(tok + 1) * K])
    else:
        bias = W[p + "mlp.gate.e_score_correction_bias"]
        sel = sorted(range(E), key=lambda e: (-(score[e] + bias[e]), e))[:K]
    w = [score[e] for e in sel]
    w = [v / (sum(w) + 1e-20) * ROUTE_SCALE for v in w]

    acc = [0.0] * D
    order = sorted(range(K), key=lambda a: sel[a])
    for a in order:
        e = sel[a]
        gu = W[p + "mlp.experts.gate_up_proj"][e * 2 * MOE_I * D:(e + 1) * 2 * MOE_I * D]
        dn = W[p + "mlp.experts.down_proj"][e * D * MOE_I:(e + 1) * D * MOE_I]
        h = matvec(gu, x, 2 * MOE_I, D)
        act = [silu(min(h[j], LIMIT)) * min(max(h[MOE_I + j], -LIMIT), LIMIT) for j in range(MOE_I)]
        y = matvec(dn, act, D, MOE_I)
        for d in range(D):
            acc[d] += w[a] * y[d]

    g = matvec(W[p + "mlp.shared_experts.gate_proj.weight"], x, MOE_I, D)
    u = matvec(W[p + "mlp.shared_experts.up_proj.weight"], x, MOE_I, D)
    act = [silu(min(g[j], LIMIT)) * min(max(u[j], -LIMIT), LIMIT) for j in range(MOE_I)]
    sh = matvec(W[p + "mlp.shared_experts.down_proj.weight"], act, D, MOE_I)
    return [acc[d] + sh[d] for d in range(D)]


def forward(W, ids):
    """Stateless full-prefix forward. Returns logits for every position."""
    n = len(ids)
    positions = list(range(n))
    streams = [[list(W["model.embed_tokens.weight"][t * D:(t + 1) * D]) for _ in range(HC)]
               for t in ids]
    for i in range(NL):
        p = f"model.layers.{i}."
        sites = [hc_site(W, p + "attn_hc", streams[t]) for t in range(n)]
        xn = [rms(sites[t][2], W[p + "input_layernorm.weight"]) for t in range(n)]
        att = attention(W, i, xn, positions)
        streams = [hc_merge(streams[t], sites[t][0], sites[t][1], att[t]) for t in range(n)]

        sites = [hc_site(W, p + "ffn_hc", streams[t]) for t in range(n)]
        xn = [rms(sites[t][2], W[p + "post_attention_layernorm.weight"]) for t in range(n)]
        mo = [moe(W, i, xn[t], ids[t]) for t in range(n)]
        streams = [hc_merge(streams[t], sites[t][0], sites[t][1], mo[t]) for t in range(n)]

    out = []
    for t in range(n):
        flat = rms_plain([v for s in streams[t] for v in s])
        mix = matvec(W["model.hc_head.hc_fn"], flat, HC, HC * D)
        pre = [sigmoid(mix[k] * W["model.hc_head.hc_scale"][0] + W["model.hc_head.hc_base"][k])
               + HC_EPS for k in range(HC)]
        hv = [sum(pre[k] * streams[t][k][d] for k in range(HC)) for d in range(D)]
        hv = rms(hv, W["model.norm.weight"])
        out.append(matvec(W["lm_head.weight"], hv, V, D))
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_dsv4_py"
    import os
    os.makedirs(out, exist_ok=True)

    W = build_weights()
    write_safetensors(f"{out}/model.safetensors", W)
    write_config(f"{out}/config.json")

    # 13 tokens: divisible by neither compress rate, so both compressors carry a partial
    # window across the prefill -> decode boundary (the state the buffers exist for).
    prompt = [7, 42, 19, 3, 58, 14, 21, 60, 9, 33, 47, 25, 11]
    n_new = 6
    full = list(prompt)
    for _ in range(n_new):
        lg = forward(W, full)[-1]
        full.append(max(range(V), key=lambda v: lg[v]))
        print(f"  generated {len(full) - len(prompt)}/{n_new}: {full[-1]}", file=sys.stderr)

    lg = forward(W, full)
    tf = [max(range(V), key=lambda v: row[v]) for row in lg]
    nll = 0.0
    for t in range(len(full) - 1):
        row = lg[t]
        m = max(row)
        s = sum(math.exp(v - m) for v in row)
        nll -= (row[full[t + 1]] - m) - math.log(s)
    ppl = math.exp(nll / (len(full) - 1))

    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf, "ppl_ref": ppl,
           "logits_last": lg[-1],
           "_note": "pure-Python reference (tools/dsv4_ref.py), NOT transformers"}
    with open(f"{out}/ref_deepseek.json", "w") as f:
        json.dump(ref, f)
    print(f"wrote {out}/ (python ppl={ppl:.4f}) continuation: {full[len(prompt):]}")


if __name__ == "__main__":
    main()
