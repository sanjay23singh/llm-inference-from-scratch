# Step 3: KV cache

**Idea:** in attention, token t's K and V depend only on tokens <= t, so they never change once computed.
Store them. Each decode step then feeds ONE token and reads the stored K/V of everything before it.

- **Prefill:** whole prompt in one pass (parallel, compute-bound). Builds the cache. This is your TTFT.
- **Decode:** one token per step (tiny compute, but must read all weights + the whole cache: memory-bound).

**KV memory per token** = 2 (K and V) x n_layer x n_embd x bytes
GPT-2 small fp32: 2 x 12 x 768 x 4 = 73,728 bytes = 72 KB/token.

## Run

```bash
python kv_cache/step3_kv_cache/test_cache_correctness.py    # offline, instant
python kv_cache/step3_kv_cache/run.py --max_new_tokens 200
python kv_cache/step3_kv_cache/run.py --max_new_tokens 400
```

## You should be able to explain
1. Why is the causal mask `tril(..., diagonal=S-T)` during decode, and what is it when T=1?
2. Why must position embeddings start at `past` (the cache length), not at 0?
3. Why does decode ms/token stay almost flat while naive keeps growing?
4. Why is decode memory-bound even though it does far less math than prefill?
