# Step 4: Continuous batching (iteration-level scheduling)

**Paper.** Orca (Yu et al., OSDI 2022). Core idea: schedule per *iteration* (one forward pass), not per *request*.
Two ingredients, both already built in step 1:

1. **Iteration-level scheduling**: before every step, finished requests leave and waiting requests join. No request
   waits for batch-mates, so there are no zombie rows and TTFT no longer includes anyone else's decode.
2. **Selective batching**: requests at different stages (1-token decode vs whole-prompt prefill) can share a step
   because only the weight matmuls are batched across them; attention runs per sequence (see step 1).

Orca needed this because its requests have different lengths *and* different phases. Static batching can't mix
phases; vLLM v0 mixes them only by alternating whole steps.

## The two variants in `policy.py`

| | step contains | cost |
| --- | --- | --- |
| `prefill_first` (vLLM v0 default) | if anyone needs prefill: prefill only, decodes stall; else decodes | decode stalls during every prefill (ITL spikes) |
| `mixed` (Orca selective batching) | decodes + whole new prompts together | the step is as slow as the biggest prompt in it |

## Predict, then run

```bash
python batching/step4_continuous/run_continuous.py     # ~5 min on a Mac
```

Predictions: continuous has slot efficiency 1.00, much higher throughput at the same load, TTFT p50 an order of
magnitude lower. **But ITL p99 is worse than static** (static never does a mid-batch prefill). Whoever sees only the
throughput bar will miss this; the third bar is the reason step 6 exists.

Outputs: `results/continuous.json`, `continuous_bars.png`, `continuous_occupancy.png` (static: rows computed vs rows
that were useful; the gap is the waste), `continuous_ttft_cdf.png`.

## Caveat when reading throughput
At 0.8x load the engine keeps up, so *all* non-saturated policies produce tokens at the arrival rate. Throughput
only separates policies once the system is saturated; at lower load the difference shows up as latency. The load
sweep in step 7 shows both regimes.

## You should be able to explain
1. Why does continuous batching need a slot-based cache and a ragged forward pass? What breaks with `(B, T)` tensors?
2. Why is ITL p99 worse than static, while TPOT p50 is better?
3. What is the admission policy here (FCFS, until slots run out)? What happens when slots are full and a long request is at the front? (Hint: Module 3, preemption.)
4. How does iteration-level scheduling change the shape of the TTFT distribution?
