# Laguna (Poolside Laguna-S-2.1, 118B/8B MoE) on sabrewing

[Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) is Poolside's agentic
coding model — 118B total, ~8B active, 48 layers, 256 experts (top-10) + 1 shared.
Unlike Inkling's 975B (streamed from NVMe), Laguna's int4 weights **fit a single
48 GB GPU**, so it's the flagship for the A6000 tier and batched serving.

Engine: `c/laguna.c` over the shared MoE runtime (`c/moe_*.h`). Architecture: a
Qwen2-MoE / GLM / DeepSeek-V3 hybrid — GQA with interleaved 512-sliding / global
attention, per-head QK-RMSNorm, a per-head **softplus attention output gate**, RoPE
(YaRN partial-rotary on global layers, plain full rotary on sliding), a dense layer
0 + sigmoid loss-free-routed MoE (routed ×2.5, shared unscaled). No short-convs
(those are Inkling's).

## Quickstart

```sh
hf download nbeerbower/Laguna-S-2.1-colibri-int4 --local-dir ~/models/laguna_i4
cd c
make laguna CUDA=1 ARCH=native        # GPU tier; `make laguna` for CPU-only
SNAP=~/models/laguna_i4 ./laguna -p "def is_prime(n):" -n 128
```

Laguna is an instruct/RL model — raw base-completion prompts degenerate. Use the
chat template (`-f` with a rendered prompt, or the gateway):

```
〈|EOS|〉<system>…</system>\n<user>…</user>\n<assistant></think>
```

`</think>` right after `<assistant>` disables reasoning; `<think>` enables it. The
gateway (`openai_server.py --arch laguna`) applies this automatically.

## Container format

Produced by `c/tools/convert_laguna_int4.py` from the bf16 HF checkpoint:
- **Routed experts → int4**, per-layer fused: `experts.gate_up_proj` packed
  `[E·2I, D/2]` (gate rows then up rows — block-concat, not interleaved) + `.qs`
  f32 per-row scales; `experts.down_proj` `[E·D, I/2]` + `.qs`.
- **Residents → bf16** (attn q/k/v/o/g, qk-norm, dense layer-0 MLP, shared expert,
  embed, lm_head); norms / router / `e_score_correction_bias` stay f32.
- `laguna.c` auto-detects the int4 container (`experts.gate_up_proj.qs` present).

## A6000 GPU tier

`make laguna CUDA=1` brings up the A6000 (`backend_cuda_laguna.cu`):
- **bf16 residents → VRAM** (~8 GB); resident matmuls run on-device.
- **Expert VRAM cache**: each MoE layer's int4 expert blob (~1.2 GB) uploads
  greedily until VRAM fills — 35–37 of 47 layers fit; the rest use the fast CPU
  int4 path. Fused GPU expert kernel + batched per-layer submission (one sync).
- **`RES8=1`** stores residents int8 in VRAM (~4 GB) → ~2 more expert layers fit
  and half the resident VRAM bandwidth (~lossless).

The routed experts total ~57 GB int4, which does not fit 48 GB — so some layers
stay on the CPU. That capacity gap is the single-stream bottleneck.

## Speed levers (all opt-in; the oracle stays exact)

| Knob | Effect |
|---|---|
| `CUDA=1` build | residents + expert cache on the A6000 |
| `RES8=1` | int8 residents in VRAM (fits more experts, +bandwidth) |
| `LAG_IDOT=1` | CPU experts via int8-VNNI (~+19% on the CPU tier; ~0.4% quant noise) |
| `BATCH=N` | batched-decode throughput bench (N copies of the prompt) |
| `CUDA_HEADROOM_MB`, `CUDA_EXPERT_GB` | tune the expert-cache VRAM budget |
| `LAG_GPU_MINEL` | min weight size to offload a resident matmul (default 0 = all) |
| `NOGPU=1`, `GPU_DEV=n` | disable GPU / pick device |

## Batched serving (the throughput win)

Single-stream decode is bandwidth/latency-bound (~12–14 tok/s). Decoding many
requests together reads the resident weights **once for the whole batch**
(`forward_batch`), so aggregate throughput scales:

```
BATCH=16 SNAP=~/models/laguna_i4 ./laguna -f chat.txt -n 24
```

| Batch | Aggregate tok/s | Speedup |
|---|---|---|
| 1 | 11.5 | 1.0× |
| 8 | 28.9 | 2.5× |
| 16 | 30.9 | 2.7× |

Every stream is bit-identical to single-stream (per-row identical to `forward` at
S=1). Throughput saturates ~batch 16 because experts are still computed per-token;
group-by-expert batching is the next lever.

## Chat serving (OpenAI + Anthropic gateway)

```sh
python3 openai_server.py --arch laguna --engine ./laguna --model ~/models/laguna_i4
# http://127.0.0.1:8000 — /v1/chat/completions, /v1/messages, streaming, web UI
```

`--arch laguna` is auto-detected from `config.json` (`model_type: laguna`). The
gateway renders Laguna's chat template and splits reasoning between
`<think>`/`</think>`; model id `laguna-s-2.1-colibri`.

## Validation

- **Token-exact oracle**: `c/tools/make_tiny_laguna.py` builds a tiny random-init
  `LagunaForCausalLM` (needs `transformers >= 5.7`) and a reference; the engine
  matches it teacher-forced **36/36**, greedy **24/24**, perplexity within 0.00%.

  ```sh
  SNAP=/path/to/tiny_laguna ./laguna ref_laguna.json
  ```

- **Quant kernels** (not oracle-reachable — the oracle runs f32): `make kernel-check`
  validates the int4/bf16/int8 kernels against a double-precision dequant reference.
- The real 118B int4 model reproduces its CPU output byte-identically under the GPU
  tier and the batched path (residual float noise only in the mixed CPU/GPU expert
  case, within quant tolerance).
