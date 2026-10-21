# Step 6: Chunked prefill

**Papers.** Sarathi / Sarathi-Serve (Agrawal et al., 2023-24): "stall-free" scheduling; the default in vLLM v1 and SGLang.

**Problem left by step 4.** One prefill step is compute-bound and costs time proportional to the prompt length. While
it runs, every decoding request produces nothing. A 480-token prompt landing in a healthy decode stream gives every
user one long gap. TPOT (the mean) barely moves; **ITL p99 explodes**. Mean metrics hide it; tail ITL doesn't.

**Idea.** Give every step a token budget and fill it in a fixed order:

```
step = [ 1 token for every running decode ]  +  [ a CHUNK of prompt tokens, up to the leftover budget ]
```

* Decodes are never skipped: stall-free.
* A prompt that doesn't fit is continued next step. Its K/V so far stays in its slot; the next chunk's attention reads
  it through the `start` offset of the causal mask (step 1).
* Step time is now bounded by the time to push `budget` tokens, so ITL has a ceiling regardless of prompt length.

## The budget is a trade-off

| Budget | Effect |
| --- | --- |
| small | tight ITL, but a long prompt needs many steps (TTFT of long prompts rises), and each chunk re-streams the weights for few tokens (the memory-bound regime) |
| large | approaches `mixed`: efficient prefill, big stalls |
| just above the knee of the step-time curve | decodes ride along with prefill tokens at nearly no extra cost. This is why defaults sit in the hundreds of tokens |

The knee comes from `step7_compare/calibrate.py` (`knee`), to be compared with 2 x the ridge point in `foundations/HARDWARE.md`.

## Predict, then run

```bash
python batching/step6_chunked_prefill/run_stall.py       # ~5 min on a Mac
```

Predictions: prefill-first / mixed show ITL max about one full long-prompt prefill; chunked ITL max tracks the budget;
long-prompt TTFT rises as the budget shrinks; short-request TTFT is unaffected until the budget gets very small.

Outputs: `results/stall.json`, `stall_tradeoff.png` (TTFT of long prompts vs ITL p99, one point per budget),
`stall_steps.png` (duration of every step over time: spikes for whole-prompt prefill, a flat ceiling for chunked).

## Correctness
`tests/test_correctness.py` checks chunked prefill (budgets 7 and 32) gives exactly the tokens of single-request
generation, and that logits of a prefill split into 3 chunks equal the unchunked logits.

## You should be able to explain
1. Why does chunked prefill leave TPOT nearly unchanged but fix ITL p99?
2. What is the cost of chunking for the long prompt itself? Who pays?
3. How would you choose the budget from a roofline / step-time curve?
4. What changes in the causal mask for chunk k > 0? What breaks if you forget it?
5. What does *disaggregated prefill/decode* (Module 9) do about the same interference, and when is it better than chunking?
