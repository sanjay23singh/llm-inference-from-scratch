# Step 1: Slot engine and ragged forward

**Problem.** Part 2 ended with three loose ends: the KV cache grows with `torch.cat` (a full copy every step),
every sequence in a batch must have the same length (padding), and a sequence can never join or leave a running
batch. All three come from one design choice: the batch is a single `(B, T)` tensor with a single cache tensor.

**Design.** Two changes, both in `batchlib/slot_engine.py`:

| Change | What it removes |
| --- | --- |
| `SlotKV`: one preallocated tensor per layer `(n_slots, heads, max_len, head_dim)`; a request owns a slot; new K/V rows are written in place | the per-step `torch.cat` copy; sequences can join/leave independently |
| `ragged_forward`: all tokens of all sequences concatenated into one `(N, C)` tensor, each with its own position and slot | padding; lets one step mix a 1-token decode with a 90-token prefill chunk |

**The one idea to hold on to.** Weights are shared by every sequence; K/V is not.

```
per layer:   QKV proj  ->  attention  ->  out proj  ->  MLP
              [N tokens]    [per sequence]   [N tokens]    [N tokens]
              weights read   reads ITS OWN    weights read  weights read
              ONCE           K/V only         ONCE          ONCE
```

Batching saves weight traffic, which is why decode (memory-bound) gets N tokens for about the price of one.
It does nothing for attention traffic: each sequence still reads its own K/V. That is why K/V, not weights, becomes
the bottleneck as batch x context grows (Module 3), and why real engines write a fused kernel that loops over
sequences inside one launch. This repo loops in Python: same bytes moved, more launch overhead.

**Why logits only for the last position.** Step 3 computed `(T, 50257)` logits during prefill and discarded all but
the last row. `lm_head` is the largest single matrix (~154 MB in fp32). Now only rows that need a next token go through it.

## Run

```bash
python batching/tests/test_correctness.py          # offline, seconds: engine == Step 3 model; every policy == single-request tokens
python batching/tests/test_correctness.py --real   # also against pretrained GPT-2 (needs the HF cache)
python batching/step1_slot_engine/bench_engine.py  # ~1 min on a Mac
```

Outputs: `results/engine_bench.json`, `results/engine_bench.png`.

## What the benchmark shows
* **A. the append alone**: `torch.cat` copies `2 x layers x B x (S+1) x hidden x 4 B` per step (hundreds of MB at B=16,
  S=250, i.e. comparable to the weights themselves) vs. writing one row in place.
* **B. whole decode step**: Step 3 (batched attention, cat cache) vs the slot engine (in-place cache, per-sequence
  attention loop). The loop costs launch overhead; the cat costs bandwidth. Read which one wins on your machine.

## Cost of this design (set up for Module 3)
Every slot reserves `max_len` tokens whether the request uses 30 or 300. At 73.7 KB/token (GPT-2 small, fp32) a
256-token slot is 18.9 MB; a request that uses 100 tokens wastes 61% of it. That is internal fragmentation and it is
what PagedAttention removes.

## You should be able to explain
1. Why is writing K/V in place cheaper than `torch.cat`, and how does the saving scale with batch and context?
2. In a ragged step, which operations are shared across sequences and which are per sequence? Why?
3. Why does the causal mask need a `start` offset, and what goes wrong without it for a prefill chunk?
4. Why can we skip `lm_head` for most rows?
