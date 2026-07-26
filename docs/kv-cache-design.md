# Where our KV cache should go: unified cells

Our KV cache is **per-slot and fully reserved**. Each serve slot allocates its own buffers
sized for `CTX_MAX`, and each shared prefix snapshot allocates another whole cache. That is
simple, it is why prefix reuse could be built in an afternoon, and it wastes most of the
memory it takes.

llama.cpp's `unified` cache and vLLM's paged attention are two points on the same design:
stop giving each sequence its own buffer, and let sequences *share cells*.

## What llama.cpp does

`llama_kv_cache` takes a `unified` flag, and the only thing it changes is how many streams
of cells exist:

```cpp
n_stream(unified ? 1 : n_seq_max)
```

Unified: **one** cell array for every sequence. Non-unified: one per sequence, which is what
we do today.

The interesting part is `llama-kv-cells.h`, where each cell carries a **bitset of sequence
ids** rather than belonging to one sequence:

```cpp
bool seq_rm  (uint32_t i, llama_seq_id seq_id);   // seq[i].reset(seq_id)
bool seq_keep(uint32_t i, llama_seq_id seq_id);   // returns true if the cell became empty
```

So a cell holding position *p* of a shared preamble can be referenced by every conversation
that shares that preamble. Prefix sharing becomes setting a bit. Nothing is copied, and the
preamble is stored **once** for the machine.

vLLM takes the same idea further: fixed-size blocks of tokens, a per-sequence block table,
copy-on-write when sequences diverge, and content hashing so a shared prefix is *discovered*
rather than declared. Our `psnap_*` pool is a crude version of that last part — hash-keyed,
but copying whole caches instead of sharing blocks.

## What it would buy us, in our numbers

At `CTX_MAX=32768`, one slot's KV is **3.52 GB** (12 global layers × full context + 36
sliding layers × a 1024-slot ring). Today:

| configuration | reserved |
|---|---|
| 6 slots, fully reserved | **21.1 GB** |
| + 4 prefix snapshots (each a whole cache) | **+14.1 GB** |
| six 4k-token conversations, actual tokens in use | ~2.6 GB |

So we reserve **~35 GB to hold ~2.6 GB of real context**, and a shared preamble is stored
once per snapshot *and* again in every slot that restored it. A unified pool sized by total
tokens rather than slots × context would fit an order of magnitude more concurrent
conversations in the same RAM — which is the entire point of the agent-fleet workload, and
the reason the box has 187 GB.

## What makes it work here, and the one hard part

Cheap, because we already have the pieces:

- Routing and scheduling are already host-side, so a cell table is plain C the scheduler can
  own — no device round-trip to consult it (the sync advantage over llama.cpp noted in
  `llamacpp-notes.md` applies here too).
- Prefix *matching* already exists (`psnap_match`, the longest-common-prefix walk in
  `slot_admit`); it would set bits instead of memcpy'ing buffers.
- The sliding-window ring already proves we can index K/V by something other than absolute
  position, and the `hist_len == kv.len` invariant is the accounting discipline that made
  reuse safe.

The hard part is attention. Every one of our SDPA paths — `sdpa_head`, `sdpa_group`,
`sdpa_head_q8` — walks **contiguous** positions `t0..qpos` (optionally ring-masked). Unified
cells scatter a sequence's positions across the pool, so attention has to read an **index
list** (vLLM's block table) instead of a range. That touches:

- the three SDPA entry points, which grow a `const int32_t *cells` parameter;
- the GQA-grouped path, whose whole value is streaming one KV head's window once — with
  scattered cells the win survives only if blocks are large enough to stay sequential, which
  is the argument for block granularity (16-32 positions) rather than per-cell indirection;
- the ring logic, which unified cells replace outright — sliding layers become "the last
  `window` blocks of this sequence's table", which is cleaner than the
  `next_pow2(window + span)` rule we derived the hard way.

## Sequencing, if we do it

1. **Blocks, not cells.** Fixed 32-position blocks: an index list per (sequence, layer),
   and attention reads block-contiguous memory so the grouped path keeps its locality.
2. **Attention by block table.** Add the parameter to the three SDPA paths, keep the
   contiguous fast path for the single-block case, and gate on the oracle at every step —
   the ring change taught us this is exactly where a silent correctness bug hides (36/36 →
   4/36 from one off-by-one in what the pass may overwrite).
3. **Share on match.** Replace the snapshot pool: on admission, walk the prompt against a
   content-hashed block index and reference existing blocks instead of copying. Prefix reuse
   then costs nothing and works across every conversation at once.
4. **Free by refcount.** A block dies when the last sequence referencing it retires, which
   also removes `LAG_PREFIX_SNAPS` as a tuning knob.

Steps 1-2 are the risky ones and buy nothing on their own; step 3 is where the memory and
the fleet-scale prefix sharing arrive. Worth doing in that order anyway, because step 2
without the oracle green is a silent-wrong-answer machine.
