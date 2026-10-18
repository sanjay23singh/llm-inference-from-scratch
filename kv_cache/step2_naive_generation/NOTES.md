# Step 2: Naive generation

**Idea:** to get token n+1, feed all n tokens in and take the logits at the last position. Append, repeat.

**Why it is slow:** step k processes (prompt + k) tokens, so total work is 1 + 2 + ... + n = O(n^2).
Almost all of it is redundant: the K and V for earlier tokens never change, but we recompute them every step.

## Run

```bash
python kv_cache/step2_naive_generation/generate.py --max_new_tokens 100
python kv_cache/step2_naive_generation/generate.py --max_new_tokens 400   # watch per-step latency grow
```
(First run downloads the tokenizer; if offline mode blocks it: `unset TRANSFORMERS_OFFLINE` for that run.)

## You should be able to explain
1. Why do we only use `logits[:, -1, :]` and ignore the other positions?
2. Why does per-step latency grow with sequence length?
3. What exactly is being recomputed that did not change? (Hint: K and V.)
