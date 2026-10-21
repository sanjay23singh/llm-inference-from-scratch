# Blog brief: Part 3, batching (input for the blog-writing agent, not prose)

**Results machine:** arm, 8 cores, device `cpu`, macOS-26.5.1-arm64-arm-64bit-Mach-O, torch 2.14.1, weights `gpt2-pretrained` (timings depend on shapes, not weight values), run 2026-10-19 17:09.
If this is not the M1 in `foundations/HARDWARE.md`, re-run `bash batching/run_all.sh` on the Mac and `python batching/step7_compare/make_brief.py`; every number below regenerates. Ratios and rankings carry across machines, absolute milliseconds do not. Quote hardware peaks from `foundations/HARDWARE.md` only.

Labels: **measured** = real engine wall-clock, **simulated** = cost model (`step7_compare/results/cost_model.json`), **computed** = from the workload or arithmetic.

## 1. The problem and which post left it open

Part 2 ended with three open items: *continuous batching*, *cache allocation (torch.cat copies)*, *long prefills stalling decode*. It showed batching is almost free in decode but assumed a tidy batch: all requests start together and are the same length. Real traffic is neither. Do not re-cover roofline, ridge point, KV sizing or GQA (Part 2).

Thread of the post: static batching wastes the batch -> a timeout (dynamic) fixes the wrong thing -> make the batch per-iteration (continuous) -> that exposes prefill stalls -> chunk the prefill -> the new ceiling is KV memory.

## 2. Attempts: what was tried, the number, why it failed or worked

### 2.1 Static batching (the baseline from Part 2's experiment)
Setup: saturated arrivals, output-length spread sigma swept, 8 slots. Generous baseline: no prompt padding, preallocated cache; only the schedule's waste remains. Source: `batching/step2_static/results/static_variance.json`

| sigma | length CV (computed) | slot efficiency, expected (computed) | slot efficiency (measured) | avg live rows/step (measured) | tok/s (measured) | TTFT p50 s | TTFT p99 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.00 | 1.00 | 1.00 | 8.0 | 167.7 | 4.21 | 7.86 |
| 0.4 | 0.39 | 0.59 | 0.59 | 4.7 | 109.6 | 6.76 | 11.03 |
| 0.8 | 0.75 | 0.41 | 0.41 | 3.3 | 75.5 | 8.85 | 14.07 |
| 1.2 | 1.01 | 0.33 | 0.33 | 2.6 | 61.4 | 8.93 | 13.42 |

Why it fails: a finished request keeps its row until the longest request ends (zombie rows). Efficiency = E[len]/E[max len per batch]; the measured value tracks the computed one, so the loss is purely the schedule.

### 2.2 Dynamic batching (timeout) and the length-bucketing oracle
Source: `batching/step3_dynamic/results/dynamic_wait.json`. `avg batch` = live decode rows per step (measured).

Load 0.25x decode capacity:

| policy | TTFT p50 s | TTFT p99 s | queue p50 s | tok/s | slot eff | avg batch |
| --- | --- | --- | --- | --- | --- | --- |
| static | 1.56 | 5.82 | 1.40 | 33.9 | 0.49 | 3.9 |
| dynamic w=0.1s | 0.68 | 2.09 | 0.62 | 35.1 | 0.65 | 1.6 |
| dynamic w=0.3s | 0.40 | 1.76 | 0.31 | 35.8 | 0.63 | 1.7 |
| dynamic w=0.6s | 0.67 | 2.45 | 0.60 | 35.3 | 0.63 | 1.8 |
| dynamic w=1.2s | 1.05 | 2.07 | 0.97 | 34.7 | 0.62 | 2.1 |
| bucketed w=0.6s | 0.71 | 2.09 | 0.62 | 34.7 | 0.63 | 1.8 |

Load 1.5x decode capacity:

| policy | TTFT p50 s | TTFT p99 s | queue p50 s | tok/s | slot eff | avg batch |
| --- | --- | --- | --- | --- | --- | --- |
| static | 2.92 | 4.41 | 2.78 | 84.6 | 0.49 | 3.9 |
| dynamic w=0.1s | 3.12 | 5.11 | 2.97 | 80.4 | 0.48 | 3.2 |
| dynamic w=0.3s | 3.33 | 6.58 | 3.19 | 70.4 | 0.44 | 3.1 |
| dynamic w=0.6s | 2.78 | 4.14 | 2.64 | 87.5 | 0.49 | 3.9 |
| dynamic w=1.2s | 2.78 | 4.12 | 2.63 | 87.8 | 0.49 | 3.9 |
| bucketed w=0.6s | 1.34 | 4.41 | 1.22 | 91.3 | 0.52 | 4.2 |

Why: the timeout bounds batch-formation delay, so it helps at light load (at the price of near-empty batches) and does nothing once a full batch is always waiting. The bucketed row is an ORACLE (uses true output length) and shows how much of the continuous-batching win is just length awareness.

### 2.3 Replacing torch.cat with a slot cache (Part 2's open allocation problem)
Batch 16; source `batching/step1_slot_engine/results/engine_bench.json`. Weights 498 MB (computed). `cat copied` is computed: 2 x layers x B x (S+1) x hidden x 4 B.

| context | append via cat ms (measured) | append in place ms (measured) | cat copies MB (computed) | decode step, Step 3 model ms | decode step, slot engine ms |
| --- | --- | --- | --- | --- | --- |
| 32 | 26.5 | 0.39 | 39 | 291 | 363 |
| 64 | 8.6 | 0.28 | 77 | 62 | 56 |
| 128 | 8.0 | 0.11 | 152 | 52 | 54 |
| 192 | 14.5 | 0.39 | 228 | 58 | 54 |
| 248 | 11.7 | 0.11 | 294 | 63 | 55 |

Why: cat copies the whole cache every step (grows with batch x context); in-place writes one row. The slot engine pays a python loop over sequences for attention (a real engine fuses it). Report which side wins on this machine, honestly.

### 2.4 Continuous batching (iteration-level scheduling)
Same workload for every policy, 0.8x load. Source: `batching/step7_compare/results/compare_headline.json`. SLO for goodput: {'slo_tpot': 0.12678400240838528, 'slo_ttft': 1.0565333534032106}.

| policy | tok/s | goodput tok/s | TTFT p50 s | TTFT p99 s | TPOT p50 ms | ITL p99 ms | ITL max ms | slot eff | avg batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static | 64.1 | 11.1 | 5.09 | 9.59 | 42 | 52 | 64 | 0.39 | 3.1 |
| dynamic | 58.1 | 6.1 | 7.01 | 14.24 | 42 | 61 | 81 | 0.38 | 2.6 |
| bucketed | 84.2 | 21.9 | 1.89 | 6.18 | 42 | 47 | 48 | 0.64 | 3.8 |
| continuous | 104.4 | 104.4 | 0.08 | 0.41 | 52 | 155 | 186 | 1.00 | 4.4 |
| continuous_mixed | 106.2 | 106.2 | 0.07 | 0.23 | 44 | 123 | 125 | 1.00 | 3.9 |
| chunked | 105.9 | 105.9 | 0.08 | 0.29 | 44 | 69 | 70 | 1.00 | 3.8 |

Why it works: nothing waits for batch-mates, so no zombie rows and TTFT stops including someone else's decode. Why it is not free: prefill-first makes every admitted prompt a prefill-only step during which all decoders stall (ITL tail). At 0.8x load every unsaturated policy produces tokens at the arrival rate; throughput only separates policies at saturation, which is why the load sweep below matters.

### 2.5 Padding vs ragged batches
Batch 8, T_max 160. Source `batching/step5_ragged/results/padding.json`. Predicted speedup is computed: B x T_max / sum(T_i).

| real tokens / padded | prefill padded ms | prefill ragged ms | speedup (measured) | predicted (computed) | decode padded ms | decode ragged ms | K/V bytes ratio (computed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100% | 366 | 362 | 1.01x | 1.00x | 43 | 43 | 1.00x |
| 88% | 366 | 320 | 1.14x | 1.14x | 43 | 42 | 1.14x |
| 75% | 371 | 280 | 1.32x | 1.33x | 44 | 42 | 1.33x |
| 62% | 372 | 256 | 1.46x | 1.60x | 43 | 41 | 1.60x |

Why: prefill is compute-bound, so padding is real work; decode streams weights once either way, so padding only matters through the K/V reads, which are a small share of a GPT-2 small step at these lengths. Padded cost is the pad tokens run through the same kernels (a faithful stand-in for a masked tensor).

### 2.6 Chunked prefill
15% of prompts are long. Source `batching/step6_chunked_prefill/results/stall.json`.

| policy | long-prompt TTFT p50 s | short TTFT p50 s | ITL p50 ms | ITL p99 ms | ITL max ms | max step ms | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefill-first | 0.20 | 0.06 | 41 | 241 | 314 | 209 | 72.5 |
| mixed | 0.19 | 0.07 | 41 | 178 | 188 | 188 | 72.5 |
| chunked 32 | 0.64 | 0.14 | 39 | 51 | 58 | 58 | 72.7 |
| chunked 64 | 0.43 | 0.08 | 41 | 59 | 90 | 90 | 72.3 |
| chunked 128 | 0.28 | 0.07 | 41 | 74 | 96 | 96 | 72.8 |
| chunked 256 | 0.24 | 0.07 | 41 | 108 | 111 | 111 | 73.1 |
| chunked 512 | 0.19 | 0.07 | 40 | 178 | 181 | 181 | 72.9 |

Why: step time is bounded by the budget, so ITL has a ceiling; the long prompt itself pays by needing ceil(P/budget) steps. A small budget also re-streams the weights for few tokens.

### 2.7 Cost model and simulator (used for the sweeps)
Source `batching/step7_compare/results/cost_model.json`. Fit: t0 28.7 ms, c 259.5 ns/attention pair, d 1356 us/sequence, R^2 0.922, worst relative error 79% (all measured/fitted). Fitted knee (a step 10% slower than a 1-token step): **38 tokens**; HARDWARE.md ridge for this device predicts ~40 tokens (M1 CPU fp32: 2 x 19.8) if weights dominate. Disagreement is a finding.

Simulator vs real engine, same workload, six policies (`batching/step7_compare/results/sim_validation.json`):

| metric | Spearman rank agreement | median abs error |
| --- | --- | --- |
| throughput_tps | 0.94 | 2% |
| ttft_p50 | 0.94 | 16% |
| ttft_p99 | 0.94 | 17% |
| tpot_p50 | 0.09 | 4% |
| itl_p99 | 0.77 | 5% |

### 2.8 Load sweep (simulated, median of 3 seeds)
Source `batching/step7_compare/results/compare_load.json`; plot `batching/step7_compare/results/compare_load.png`.

| load | static tok/s | dynamic tok/s | bucketed tok/s | continuous tok/s | continuous_mixed tok/s | chunked tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 35.4 | 35.5 | 35.5 | 35.5 | 35.5 | 35.5 |
| 0.4 | 70.4 | 70.7 | 70.7 | 70.9 | 70.9 | 70.9 |
| 0.6 | 81.3 | 80.2 | 99.9 | 106.2 | 106.2 | 106.1 |
| 0.8 | 82.6 | 83.7 | 118.7 | 140.2 | 141.3 | 141.3 |
| 1 | 83.3 | 81.4 | 139.6 | 153.5 | 167.2 | 167.2 |
| 1.2 | 83.8 | 82.0 | 143.1 | 155.2 | 172.5 | 172.6 |
| 1.5 | 84.0 | 84.0 | 145.7 | 156.9 | 174.7 | 175.0 |

| load | static TTFT p99 s | dynamic TTFT p99 s | bucketed TTFT p99 s | continuous TTFT p99 s | continuous_mixed TTFT p99 s | chunked TTFT p99 s |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 10.67 | 3.18 | 3.18 | 0.14 | 0.14 | 0.20 |
| 0.4 | 6.63 | 5.46 | 5.81 | 0.15 | 0.15 | 0.23 |
| 0.6 | 24.73 | 25.32 | 7.76 | 0.70 | 0.46 | 0.47 |
| 0.8 | 39.86 | 37.73 | 11.93 | 2.08 | 1.22 | 1.24 |
| 1 | 48.94 | 50.85 | 18.09 | 7.21 | 3.45 | 3.45 |
| 1.2 | 55.08 | 56.69 | 24.22 | 13.27 | 8.62 | 8.64 |
| 1.5 | 61.87 | 61.87 | 27.49 | 19.43 | 14.72 | 14.69 |

### 2.9 Output-length variance sweep (simulated, saturated)
| length CV (computed) | static tok/s | dynamic tok/s | bucketed tok/s | continuous tok/s | continuous_mixed tok/s | chunked tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 174.2 | 172.5 | 172.5 | 167.2 | 179.4 | 178.7 |
| 0.25 | 130.9 | 130.1 | 160.0 | 159.5 | 178.1 | 177.6 |
| 0.49 | 101.9 | 101.9 | 150.8 | 159.1 | 176.4 | 176.1 |
| 0.69 | 85.9 | 85.9 | 147.1 | 158.0 | 175.0 | 174.8 |
| 0.84 | 77.2 | 77.2 | 140.7 | 154.9 | 172.4 | 172.2 |
| 0.98 | 69.5 | 69.5 | 137.0 | 150.1 | 168.6 | 168.9 |

### 2.10 Concurrency vs reserved KV memory (simulated, saturated)
Each slot reserves 416 tokens; a request averages 122 (computed) -> **71% of reserved KV is unused** (internal fragmentation). Source `batching/step7_compare/results/compare_slots.json`.

| slots | KV reserved MB (computed) | static tok/s | continuous tok/s | chunked tok/s |
| --- | --- | --- | --- | --- |
| 2 | 61 | 45.0 | 59.4 | 60.9 |
| 4 | 123 | 62.4 | 101.6 | 107.8 |
| 8 | 245 | 84.0 | 156.9 | 175.0 |
| 16 | 491 | 107.9 | 215.1 | 244.6 |
| 32 | 981 | 125.5 | 267.9 | 296.9 |

## 3. Predictions vs outcome (auto-checked)

| prediction | result | evidence |
| --- | --- | --- |
| Static slot efficiency equals E[len]/E[max len] computed from the workload | OK | max gap 0.000 |
| At saturation dynamic ~ static (timeout never fires) | MISS | max throughput gap 17% |
| Continuous TTFT p50 < half of static's | OK | 0.08s vs 5.09s |
| Continuous (prefill-first) ITL p99 is WORSE than static's | OK | 155ms vs 52ms |
| Continuous slot efficiency = 1.00 | OK | 1.00 |
| Ragged prefill speedup within 20% of computed B*Tmax/sum(T) at the widest spread | OK | 1.46x vs 1.60x |
| Chunked ITL max < prefill-first ITL max (best budget) | OK | 58ms vs 314ms |
| Long-prompt TTFT rises as the budget shrinks | OK | budget 32: 0.64s vs budget 512: 0.19s |
| Simulator keeps the policy ranking for throughput (rho >= 0.9) | OK | rho 0.94 |

A MISS is not a bug to hide: say what the machine did instead and why (profile it, or list it under section 6).

## 4. Plots to include

- `batching/step1_slot_engine/results/engine_bench.png`
- `batching/step2_static/results/static_variance.png`
- `batching/step3_dynamic/results/dynamic_wait.png`
- `batching/step4_continuous/results/continuous_occupancy.png`
- `batching/step4_continuous/results/continuous_bars.png`
- `batching/step5_ragged/results/padding.png`
- `batching/step6_chunked_prefill/results/stall_steps.png`
- `batching/step6_chunked_prefill/results/stall_tradeoff.png`
- `batching/step7_compare/results/compare_load.png`
- `batching/step7_compare/results/compare_variance.png`
- `batching/step7_compare/results/compare_slots.png`
- `batching/step7_compare/results/sim_validation.png`

Best single figure for the post: `continuous_occupancy.png` (rows computed vs rows useful: the waste is visible).

## 5. Interview questions

In `batching/INTERVIEW_QS.md`.

## 6. Still unexplained / unprofiled

- Per-sequence attention is a python loop; its overhead vs a fused varlen kernel is not profiled (needs CUDA/Triton, Modules 4 and 7).
- The step-time curve's shape at small N (kernel choice, thread count) is taken as measured, not explained; see the knee vs ridge comparison in 2.7.
- Output lengths are assigned by the workload, not by EOS; real length distributions are heavier-tailed and correlated with the prompt.
- No preemption: when slots are full a long request at the head of the queue blocks the rest (FCFS). Recompute vs swap is Module 3.
- Greedy decoding only; per-request sampling params and stop strings are not batched.
- fp32 weights; with fp16/INT8 the knee moves (fewer bytes per weight -> lower ridge in tokens).
- Prefix sharing between requests is not modelled (Module 3).

## 7. Open problem that leads to the next post

Throughput keeps rising with concurrency, but each slot reserves a full-length K/V region and 71% of it is unused on this workload (computed), while the reservation grows linearly with slots (2.10). KV memory, not compute, caps concurrency. Next: KV memory management: fragmentation, PagedAttention and block tables, prefix caching, preemption (swap vs recompute), and what GQA/MLA do to bytes per token.
