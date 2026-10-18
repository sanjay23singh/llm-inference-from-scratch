# Step 4: Sampling

The model outputs logits. Greedy takes the argmax, which is deterministic and tends to loop (you saw this).
Sampling draws from softmax(logits) instead; three knobs reshape the distribution first:

| Knob | What it does | Too low | Too high |
|------|--------------|---------|----------|
| temperature | divides logits by T | repetitive (T->0 = greedy) | incoherent |
| top-k | keep the k most likely tokens | repetitive | long tail of junk tokens |
| top-p | keep smallest set with mass >= p | repetitive | junk tail |

top-k has a fixed size; top-p adapts: few tokens when the model is confident, many when it is unsure.

## Run

```bash
python kv_cache/step4_sampling/test_sampling.py      # offline, instant
python kv_cache/step4_sampling/run.py                # uses cached GPT-2
python kv_cache/step4_sampling/run.py --seed 1       # same settings, different draws
```

## You should be able to explain
1. Why does greedy loop, and why does sampling escape it?
2. Why does top-p keep a different number of tokens at different positions?
3. Why apply temperature BEFORE top-k/top-p?
4. In `top_p_filter`, why `mass_before >= p` rather than `cumulative > p`?
