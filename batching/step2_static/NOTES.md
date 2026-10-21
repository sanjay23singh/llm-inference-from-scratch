# Step 2: Static batching

**What it is.** Wait for B requests, prefill them together, decode them together, free everything when the
*longest* one finishes. It is `model.generate(list_of_prompts)`, and it is what Part 2's batching experiment assumed.

**Why it breaks on real traffic** (`policy.py` has the model in 40 lines):

| Failure | Cause | Cost |
| --- | --- | --- |
| Zombie rows | a finished request keeps its row until the longest one ends; the engine still computes it | slot efficiency = mean length / max length in the batch |
| Head-of-line blocking | a request arriving 1 step after launch waits for the whole batch to drain | TTFT includes someone else's decode |
| Batch formation wait | the first request of a batch waits for the B-th to arrive | TTFT at light load |

This baseline is *generous*: no prompt padding, preallocated cache. The only loss left is the schedule itself.
Real static batching also pays for padding (step 5).

## Predict, then run

Slot efficiency is computable from the workload alone: `sum(len) / (B x max(len))` per batch.

```bash
python batching/step2_static/sweep_variance.py     # ~4 min on a Mac; --engine sim after step 7a for seconds
```

Outputs: `results/static_variance.json`, `results/static_variance.png`. The black ticks on the bars are the
computed prediction; they should sit on the measured bars. If they do, the loss is **purely the schedule**, not
engine overhead.

## What to see
* sigma = 0 (all outputs equal): 100% efficiency. This is the only case static batching is fine.
* Spread lengths out and watch efficiency fall toward `E[len] / E[max len]`; throughput falls with it, since the
  hardware is busy computing rows nobody wants.
* TTFT p99 is dominated by queueing behind batches, not by compute.

## You should be able to explain
1. Why does efficiency equal E[len] / E[max len in batch]? Why does a bigger B make it worse?
2. What is the one line in `on_finish` that makes this static, and what does continuous batching do instead?
3. Why does a *longer average* output hurt static batching less than a *wider spread*?
