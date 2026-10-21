# Step 7: Everything head to head

Three scripts, run in this order:

```bash
python batching/step7_compare/calibrate.py        # 1 min: measure step times on this machine, fit the cost model
python batching/step7_compare/validate_sim.py     # ~8 min: same workload on the real engine vs the simulator
python batching/step7_compare/compare_all.py      # headline on the real engine (~8 min) + simulated sweeps (~1 min)
python batching/step7_compare/make_brief.py       # regenerate ../BLOG_BRIEF.md numbers from results/*.json
```

(`batching/run_all.sh` runs the whole module in order.)

## 7a. calibrate.py: a cost model of YOUR machine
Fits `t_step = t0 + f(N) + c*A + d*n_seq` (N tokens in the step, A attention pairs, n_seq sequences), with `f` a convex
piecewise-linear function: a roofline with a learned knee. It prints the knee next to the ridge point from
`foundations/HARDWARE.md` (predicted knee ~ 2 x ridge for fp32 weights). Disagreement is a finding, not a bug:
on a CPU with few cores the math runs out almost immediately; on the M1 the weights dominate up to a few dozen tokens.

## 7b. validate_sim.py: how far to trust the simulator
Real vs simulated, per metric and per policy, plus rank agreement. Sweeps in 7c rely on the simulator, so the
*ranking* must hold even if absolute p99s are off by tens of percent (they are single worst events).

## 7c. compare_all.py

| Part | Engine | Question |
| --- | --- | --- |
| A headline | real | six policies, one workload at 0.8x load: throughput, goodput, TTFT, ITL |
| B load sweep | sim | where does each policy's latency blow up as offered load rises? |
| C variance sweep | sim | which policies lose throughput as output lengths spread? |
| D slot sweep | sim | how much throughput does concurrency buy, and how much KV does it reserve? |

Goodput uses SLOs relative to the machine (TPOT <= 3 x full-batch decode step, TTFT <= 25 x). They are printed
with every table.

**D is the bridge to Module 3.** Throughput keeps rising with concurrency, but each slot reserves `max_len` tokens
of K/V for a request that averages far fewer: that wasted share is printed. KV memory, not compute, caps concurrency.

## Policies at a glance

| Policy | Batch decided | Joins mid-flight | Prefill vs decode | Fixes | Leaves open |
| --- | --- | --- | --- | --- | --- |
| static | once per batch | no | whole batch prefills together | nothing | zombie rows, HOL blocking, formation wait |
| dynamic | once, with timeout | no | same | formation wait | zombie rows |
| bucketed (oracle) | once, grouped by length | no | same | most zombie rows (needs length prediction) | HOL blocking |
| continuous, prefill-first | every step | yes | prefill-only steps; decodes stall | zombies, HOL | ITL spikes |
| continuous, mixed (Orca) | every step | yes | whole prompts ride with decodes | stalls become shorter | step as slow as the biggest prompt |
| chunked prefill | every step | yes | token budget; decodes first | ITL ceiling | budget tuning; KV reservation (Module 3) |
