#!/usr/bin/env python3
"""A/B two OpenAI-compatible endpoints on the axes that decide an agent workload.

RUN THE ENGINES ONE AT A TIME. Both want most of a 48 GB card, and a resident second engine
distorts everything (see the benchmarking rules in CLAUDE.md). Stop one, verify VRAM is back
to idle, start the other, and wait for load average to settle before each run.

    python3 tools/bench_endpoints.py http://127.0.0.1:8099 qwen3.5-35b-a3b-sabrewing sabrewing
    python3 tools/bench_endpoints.py http://127.0.0.1:8080 model.gguf                llamacpp


Thinking is DISABLED on both engines: reasoning-token volume varies wildly between models and
would dominate any decode comparison. Separates TTFT (a prefill proxy) from the inter-token rate, because the two engines
differ almost entirely in the former. Every prompt carries a unique nonce so a warm server
cannot serve it from its prefix cache -- except the explicit `warm` case, which repeats a
prompt on purpose to measure exactly how much that cache is worth.

Usage: bench_ab.py <base_url> <model> <label>
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
FILLER = ("The runtime keeps a per-sequence recurrent state for each linear-attention layer "
          "and a conventional key/value cache only for the periodic full-attention layers. ")
NONCE = [0]


def prompt_of(approx_tokens, nonce=True):
    """~1.35 tokens per word for this filler; nonce defeats prefix caching."""
    reps = max(1, int(approx_tokens / 27))
    head = f"[request {NONCE[0]}] " if nonce else ""
    if nonce:
        NONCE[0] += 1
    return head + FILLER * reps + "\n\nReply with exactly one word: acknowledged"


def stream(prompt, max_tokens):
    """Returns (ttft_s, total_s, n_chunks)."""
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0, "stream": True,
                         # sabrewing's gateway reads enable_thinking; llama.cpp reads
                         # chat_template_kwargs. Send both so ONE harness drives both, and
                         # neither engine spends its budget on reasoning tokens.
                         "enable_thinking": False,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            try:
                d = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            # Count ONLY content deltas. The gateway's keepalive pump emits a
            # reasoning_content "." every 10s during a long prefill precisely so clients do
            # not time out -- counting it made every TTFT come out at exactly 10.01s.
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, time.time() - t0, n


def blocking(prompt, max_tokens):
    """Non-streaming; returns (wall_s, usage, timings). llama.cpp reports authoritative
    prompt_per_second / predicted_per_second in `timings`, which cross-checks TTFT."""
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0,
                         "enable_thinking": False,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    return time.time() - t0, d.get("usage") or {}, d.get("timings") or {}


res = {"label": LABEL, "prefill": {}, "decode": {}, "warm": {}, "concurrent": {}}
print(f"=== {LABEL} ===", flush=True)

# 1. cold prefill: TTFT at several prompt sizes, unique prompts
for approx in (512, 2048, 4096):
    p = prompt_of(approx)
    words = len(p.split())
    ttft, tot, n = stream(p, 8)
    w2, usage, tm = blocking(prompt_of(approx), 8)
    ptok = usage.get("prompt_tokens")
    pps = tm.get("prompt_per_second")
    res["prefill"][approx] = {"ttft_s": ttft, "words": words, "prompt_tokens": ptok,
                              "engine_prompt_per_s": pps}
    extra = f"  [engine: {ptok} tok, {pps:.1f} tok/s]" if pps else f"  [{ptok} prompt tok]"
    print(f"  cold prefill ~{approx:5d} tok: TTFT {ttft:7.2f}s{extra}", flush=True)

# 2. decode rate: long generation from a modest prompt
p = prompt_of(2048)
p = p.replace("Reply with exactly one word: acknowledged",
              "Now write a detailed 200-word explanation of caching in your own words.")
ttft, tot, n = stream(p, 200)
dec = (n - 1) / (tot - ttft) if (n > 1 and tot > ttft) else 0
res["decode"] = {"ttft_s": ttft, "total_s": tot, "chunks": n, "tok_per_s": dec}
print(f"  decode: {n} chunks, TTFT {ttft:.2f}s, then {dec:.2f} tok/s", flush=True)

# 3. warm: the SAME prompt twice — how much is the prefix cache worth?
p = prompt_of(2048, nonce=False)
t1, _, _ = stream(p, 8)
t2, _, _ = stream(p, 8)
res["warm"] = {"first_ttft_s": t1, "second_ttft_s": t2,
               "speedup": (t1 / t2) if t2 else None}
print(f"  warm repeat ~2048 tok: 1st TTFT {t1:.2f}s, 2nd {t2:.2f}s  ({t1/t2 if t2 else 0:.1f}x)",
      flush=True)

# 4. concurrency: 4 distinct requests at once, aggregate output rate
def worker(i, out):
    pp = prompt_of(2048)
    pp = pp.replace("Reply with exactly one word: acknowledged",
                    "Write a 60-word note about memory bandwidth.")
    out[i] = stream(pp, 64)

out = [None] * 4
ths = [threading.Thread(target=worker, args=(i, out)) for i in range(4)]
t0 = time.time()
for t in ths:
    t.join() if False else t.start()
for t in ths:
    t.join()
wall = time.time() - t0
tot_chunks = sum(o[2] for o in out if o)
res["concurrent"] = {"streams": 4, "wall_s": wall, "chunks": tot_chunks,
                     "aggregate_tok_per_s": tot_chunks / wall}
print(f"  4 concurrent: {tot_chunks} chunks in {wall:.2f}s = "
      f"{tot_chunks/wall:.2f} tok/s aggregate", flush=True)

with open(f"/tmp/bench_{LABEL}.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"  -> /tmp/bench_{LABEL}.json", flush=True)
