# Efficiency suite — regression tests + optimization dossier

Two layers:

1. **`test_inefficiency.py`** — tiny-model *asserted* regression tests. Fast
   (~0.15s/run), gate CI, catch breakage. Run as part of `make test`.
2. **`test_efficiency_report.py`** — an *opt-in optimization dossier* for a real
   model. Runs every instrumentation flag, prints a 9-section report answering
   *what is doing what, when, with what, is it inefficient, how to improve*.
   Never fails CI (it's a report, not a gate).

## The dossier (what you run when optimizing)

```bash
# CPU-only (safe, fast to validate):
COLI_EFFICIENCY_MODEL=../glm52_i4_g64 make efficiency-report

# CUDA (dense + expert tiers — needs a CUDA build, see below):
COLI_EFFICIENCY_MODEL=../glm52_i4_g64 COLI_EFFICIENCY_CUDA=1 make efficiency-report
```

It turns ON every observability flag the engine supports — `PROF=1`,
`COLI_CUDA_PROFILE=1`, `CACHE_ROUTE=1` (auto-unlocks `route_agree`/`route_kl`),
`DISK_SPLIT=1`, `LOOKA=1` — so nothing the engine can tell you is left dark.
None of these change the computed output; they only add telemetry.

The 9 sections, and the question each answers:

| § | section | answers |
|---|---|---|
| 1 | PROVENANCE | what is running, on what CPU/backend, with what effective config |
| 2 | THROUGHPUT | tok/s + forward-latency p50/p90/p99/max (is the tail healthy?) |
| 3 | WHERE TIME GOES | the 5 PROFILE phases as % of decode + absolute seconds + verdict |
| 3a | ATTENTION BREAKDOWN | attention split into projection/RoPE, score-softmax-value, output |
| 4 | EXPERT CACHE | hit %, experts-loaded/token vs baseline topk |
| 5 | DISK I/O | GB fetched, MB/token, GB/s, read-service vs felt-wait, phase split |
| 5a | DISK-LOAD SPLIT | loads by decode phase (draft/absorb/verify) + MTP-vs-main bytes |
| 6 | ROUTING QUALITY | route_agree %, route_kl, cache swaps |
| 6a | ROUTING PREDICTABILITY | LOOKAHEAD recall per predictor (which prefetch wins) |
| 7 | SPECULATION | tokens/forward, MTP acceptance % |
| 8 | GPU TIERS | resident tensors, expert tier (count/GB/calls), H2D/kernel/D2H ms |

Every line that crosses an advisory threshold is marked `[FLAG]` with the
concrete lever to pull (raise RAM_GB, add PIN_GB, try DIRECT=1, lower CTX, …),
and all flags repeat in a summary at the end.

## Tunable thresholds

The `IS IT INEFFICIENT?` lines are advisory constants at the top of
`test_efficiency_report.py`:

| constant | default | meaning |
|---|---|---|
| `DISK_WAIT_DOMINANT` | 0.40 | >40% decode waiting on expert reads → I/O-bound |
| `LOW_HIT_RATE` | 0.30 | <30% cache hit → thrashing |
| `LOW_ROUTE_AGREE` | 0.80 | <80% routing overlap → prefetch guessing wrong |
| `HIGH_TAIL_RATIO` | 3.0 | p99 > 3× p50 → decode stalls |
| `LOW_MTP_ACCEPT` | 0.20 | <20% MTP acceptance → draft decoder is dead weight |

The tiny-model asserted floors live in `tools/efficiency.py` (`TINY_TOK_S_FLOOR`,
`MAX_DISK_WAIT_SHARE`, `MIN_CPU_CUDA_AGREEMENT`).

## The regression tests (what gates CI)

`test_inefficiency.py` runs on the bundled `glm_tiny` model and asserts:

- telemetry parses (no format drift)
- tiny tok/s ≥ floor (throughput regression)
- PROFILE phases present and non-negative (accounting sanity)
- disk-wait not dominant on a resident model (I/O-path regression)
- CPU determinism (two greedy runs agree)
- **CUDA** (skip unless CUDA built): init path, dense uploads VRAM, CPU-vs-CUDA
  argmax agreement ≥ 70% (kernel-correctness guard)

```bash
make efficiency        # tiny CPU tests
make efficiency-cuda   # tiny CUDA tests (needs CUDA build)
```

## CUDA build prerequisite

The default `make glm.exe` builds **without** CUDA. The CUDA tests and the CUDA
dossier need a host built with `-DCOLI_CUDA` plus the runtime DLL:

```bash
make clean && make glm.exe CUDA_DLL=1 && make cuda-dll
```

`make efficiency-cuda` auto-skips with a clear message if the host is CPU-only
(it scans the binary for the "CPU-only" marker the engine embeds).

## Files

- `tools/efficiency.py` — shared harness: `parse_run()` (captures every
  telemetry signal), `run_engine()`, thresholds. Reuses `PROFILE_RE`/`SPEED_RE`
  from `tools/benchmark_cuda_fixture.py`.
- `tests/test_inefficiency.py` — tiny-model asserted tests (CPU + CUDA).
- `tests/test_efficiency_report.py` — the opt-in optimization dossier.
