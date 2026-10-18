# Step 5c: Batching experiment

**Claim to test:** decode is memory-bound, so feeding B users' tokens through the model in one step costs
much less than B separate steps. The weights are read once and shared; only the activations differ.

## Run

```bash
python kv_cache/step5c_batching/test_batch_isolation.py   # offline: users in a batch do not affect each other
python kv_cache/step5c_batching/batch_decode.py           # ~1 min: ms/step and tokens/s for B = 1..32
```

Outputs: `results/batching.png`, `results/batching.json`.

## How to read the table
- **ms/step**: time for ONE decode step for the whole batch.
- **total tok/s** = B / step time: what the server produces across all users.
- **per-user tok/s** = 1 / step time: what each user experiences (it drops as the step gets slower).
- **gain**: batched total throughput vs serving users one at a time.
- **KV MB**: every user brings their own cache, so memory grows linearly with B.

## Expect, and why it levels off
Step time rises slowly at first (weights are read once), then faster: the chip eventually runs out of spare math.
A Mac CPU has far less spare math than a GPU, so it levels off earlier than a real server would.
That trade-off (throughput vs per-user speed vs KV memory) is what every serving engine tunes.

## You should be able to explain
1. Why does total throughput rise with batch size while per-user speed falls?
2. Why does KV memory, not compute, usually cap the batch size on a GPU?
3. Why must the isolation test pass? What would it mean if it failed?
