# sabrewing

**Trillion-parameter open models on hardware you own.** A pure-C inference engine
running [Thinking Machines Inkling](https://huggingface.co/thinkingmachines/Inkling)
(975B MoE) on a single workstation — expert weights streamed from NVMe, hot experts
cached in RAM, dense weights on GPU if you have one, and an OpenAI-compatible API
in front. No Python in the inference path, no frameworks, no cluster.

*A sabrewing (**Campylopterus**) is one of the largest hummingbirds. This project is a
fork of [JustVugg/colibrì](https://github.com/JustVugg/colibri) — colibrì being Italian
for hummingbird — grown to carry heavier models. It was built on a machine named
`sabre`, and the name was sitting right there.*

## What it does

- **Inkling 975B / 41B-active**, text generation, token-exact against the HF
  transformers reference (validated by a tiny-model oracle in four numeric modes,
  including the CUDA path)
- **Runs in ~120 GB RAM** (CPU-only) — int4 experts + bf16 residents; a 24 GB+
  NVIDIA GPU moves residents to VRAM and speeds decode substantially
- **OpenAI-compatible API** (`/v1/chat/completions`, streaming, temperature/top-p)
  with Inkling's chat template, plus a browser chat UI
- **Cache warming that learns**: expert usage history accumulates across runs and
  pins your workload's hot experts at startup (all toggleable)

## Quickstart

```sh
# converted weights (int4 experts + bf16 residents, ~469 GiB)
hf download nbeerbower/Inkling-colibri-int4 --local-dir ~/models/inkling_i4

git clone git@github.com:Schneewolf-Labs/sabrewing.git && cd sabrewing/c
make inkling                 # pure CPU
make inkling CUDA=1          # + NVIDIA GPU for resident weights (~37 GB VRAM,
                             #   or partial upload on smaller cards)

# one-shot generation
SNAP=~/models/inkling_i4 ./inkling -p "The capital of France is" -n 64

# OpenAI API + web chat UI on http://127.0.0.1:8000
python3 openai_server.py --engine ./inkling --model ~/models/inkling_i4 --cap 0
```

See [`docs/inkling.md`](docs/inkling.md) for cache-warming knobs, the container
format, and benchmark methodology.

## Performance (Ryzen 9 7900 · 187 GB DDR5 · RTX A6000 · NVMe)

| Configuration | Prefill (5 tok) | Decode | Cache hit |
|---|---|---|---|
| first light | 150 s | 0.06 tok/s | ~0% |
| tuned CPU | 21 s | 0.25 tok/s | 82% |
| + GPU residents | 18 s | 0.32 tok/s | 84% |
| + deep pins, workload-trained | **1.9 s** | **2.51 tok/s** | 100% |
| steady state (novel prompts) | 35 s | 0.25 tok/s | 82% |

Decode speed is a function of cache warmth, and the cache learns your workload —
the steady-state number climbs toward the trained number with use. Phase-profiled
honestly: at high hit rates the bottleneck is CPU expert matmul, which is the
roadmap's next target.

## Roadmap

- LoRA adapter serving (fine-tune on [Tinker](https://thinkingmachines.ai/tinker),
  run the adapter here)
- mmap expert path for RAM ≥ model machines (512 GB box = every prompt runs warm)
- GPU expert tier + MTP speculative decoding
- Perplexity harness (quantization quality, measured not vibed)
- The rest of the hummingbird catalog as open trillion-scale models keep landing

## Relationship to colibrì

sabrewing is a friendly fork, not a rival. The GLM-5.2 engine (`c/glm.c`) that
colibrì is built around ships here unchanged and working; the substrate this fork
stands on — expert streaming, the int4 container, the oracle-validation culture —
is colibrì's design, and improvements that belong upstream go upstream
([tokenizer](https://github.com/JustVugg/colibri/pull/330),
[I/O robustness](https://github.com/JustVugg/colibri/pull/331),
[profiling](https://github.com/JustVugg/colibri/pull/232) — merged, and the
[Inkling engine itself](https://github.com/JustVugg/colibri/pull/312) is offered
there too). Licensed [Apache 2.0](LICENSE), same as upstream and same as the model.

---

A [Schneewolf Labs](https://huggingface.co/schneewolflabs) project.
