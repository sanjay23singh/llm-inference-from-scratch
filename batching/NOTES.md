# Module 2: Batching

From a tidy `(B, T)` batch to a mini vLLM scheduler. Every strategy runs on **one engine** and **one serving loop**;
a strategy is a ~30-line `Policy`. So any difference between two results is the schedule and nothing else.

```
batching/
  batchlib/          engine, serving loop, workload, stats, cost model (shared)
    slot_engine.py     SlotKV (preallocated per-slot cache) + ragged_forward (selective batching)
    scheduler.py       the loop: decide step -> run -> update; virtual clock
    request.py         Req / Work: "prefill vs decode" is just n_tokens
    workload.py        Poisson arrivals, lognormal output lengths, long-prompt mix
    sim_engine.py      cost model fitted from measurements + drop-in SimEngine
  step1_slot_engine/   cache + ragged forward, torch.cat vs in-place
  step2_static/        static batching, zombie rows
  step3_dynamic/       timeout, length-bucketing oracle
  step4_continuous/    iteration-level scheduling (Orca): prefill-first and mixed
  step5_ragged/        padding vs ragged
  step6_chunked_prefill/  token budget, stall-free decode (Sarathi / vLLM v1)
  step7_compare/       calibrate, validate simulator, all policies head to head, regenerate the brief
  tests/               correctness: every policy gives each request its single-request tokens
  BLOG_BRIEF.md  INTERVIEW_QS.md  run_all.sh
```

## The story in one screen

| Step | Strategy | Problem it solves | What it leaves |
| --- | --- | --- | --- |
| 2 | static | uses the batch at all | zombie rows, head-of-line blocking |
| 3 | dynamic | batch-formation wait | zombie rows |
| 4 | continuous | zombies, HOL: batch re-decided every step | decode stalls during prefill |
| 5 | ragged | padding waste; lets phases mix in one step | step as slow as the biggest prompt |
| 6 | chunked prefill | prefill stalls: bounded step time | KV memory is now the ceiling (Module 3) |

## Design decisions (and why)
* **One engine, many policies.** The engine takes `Work(slot, start, tokens, sample)`. Prefill, chunk and decode are the
  same call with different `n_tokens`. That is why chunked prefill costs no extra engine code.
* **Slots, not a growing tensor.** A request owns a slot; K/V is written in place. Joining and leaving the batch is free.
  Price: each slot reserves `max_len` tokens (internal fragmentation), the setup for Module 3.
* **Virtual clock.** Arrivals are pre-drawn; the clock advances by *measured* step time and jumps over idle gaps.
  Real compute cost, real queueing, no sleeping.
* **Static is a generous baseline.** No prompt padding, preallocated cache. Real static batching is worse (step 5).
* **Output length is assigned, not emitted.** Lets us control length variance; stands in for EOS.
* **Greedy decoding.** The experiments need token streams, not sampling variety. Per-request sampling plugs in at the
  argmax in `ragged_forward` (step4_sampling).
* **Simulator for sweeps, validated against the real engine.** ~100 sweep points x hundreds of requests is not feasible
  for real runs; `validate_sim.py` checks that the *ranking* survives.

## Run

```bash
python batching/tests/test_correctness.py     # first, always
bash batching/run_all.sh                      # whole module, ~40 min on an M1
python batching/step7_compare/make_brief.py   # numbers for the blog brief
```

Each step folder has a `NOTES.md` (theory, what to predict, what to explain) and a `results/` folder.
