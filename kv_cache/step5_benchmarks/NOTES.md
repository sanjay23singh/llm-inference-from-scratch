# Step 5: Benchmarks

Turn Steps 2-4 into evidence. One script, three charts, one table.

| Output | What it shows |
|--------|---------------|
| `results/latency_per_token.png` | naive latency grows with sequence length; cached stays ~flat |
| `results/throughput.png` | the speedup widens the longer you generate |
| `results/prefill_vs_decode.png` | prefill: lots of math per step; decode: little math, mostly waiting on memory |
| KV memory table (printed) | bytes per token = 2 x layers x kv_heads x head_dim x bytes |

## Run

```bash
python kv_cache/step5_benchmarks/benchmark.py --quick     # ~1 min, check it works
python kv_cache/step5_benchmarks/benchmark.py             # full run (a few minutes; naive at 512 tokens is slow)
```

## Why decode is memory-bound (the key idea)
Each decode step multiplies ONE token's vector by every weight matrix. Every weight (~0.5 GB in fp32 for GPT-2 small)
must be read from memory to do just 2 FLOPs per weight. Prefill reuses each weight loaded across all prompt tokens,
so the chip is busy doing math. Look at the GFLOP/s columns: decode achieves a small fraction of prefill.

## You should be able to explain
1. Why is achieved GFLOP/s so much lower in decode than prefill, even though both run the same model?
2. Roughly: decode time per token ~ bytes read / memory bandwidth. Check it against your "decode GB/s" number.
3. Why does the KV cache (not the weights) become the bottleneck as batch size and context grow? (see the KV table)
4. Why does GQA (8 KV heads instead of 32) matter so much for serving? (Step 5b)
