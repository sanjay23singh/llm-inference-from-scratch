# Module 0 wrap-up

## Benchmark (M1, CPU fp32, from this repo)
| Measure | Number |
|---|---|
| Decode, B=1 | 20 ms/token, ~23 GB/s of 45.8 GB/s peak: memory-bound |
| Prefill, 128 tokens | 74 ms (TTFT), ~425 of 907 GFLOP/s peak |
| Batching B=1 to 16 | 57 to 403 tok/s (7x) while step time only 17.6 to 39.7 ms |
| Batching B=32 | step time 53 ms: past the ridge (B~40 predicted) |
| KV per token | GPT-2 72 KB, Llama-3-8B 128 KB, Llama-2-7B (no GQA) 512 KB |
Takeaway: decode is starved for memory bandwidth, so batching is almost free until intensity hits the ridge; then KV memory caps the batch.

## Blog outline
Title: Why LLM decoding is slow, and why batching is (almost) free
1. Problem: a 100-token answer takes 2 s and 96% of it is decode.
2. Naive fix: a faster chip. Fails: the chip is idle, waiting for weights.
3. Constraints: two ceilings (compute, bandwidth), the roofline, ridge at 20 FLOP/byte.
4. Design: reuse each weight read across users. Intensity = 0.5 x B, so predicted knee at B~40; measured ~32.
5. New constraint: every user adds KV cache. Formula, GQA saves 4x (59 vs 14 users).
6. Results: throughput vs per-user speed vs goodput under an SLO.

## Interview Qs
1. Why is decode memory-bound? Each token reads all weights for ~2 FLOPs per weight, so intensity ~1 FLOP/byte (fp16) vs a ridge in the hundreds.
2. How does batching help, and where does it stop? Weights are read once per step and shared, so intensity grows with B (fp16: ~B FLOP/byte). It stops at the ridge (H100 ~300), and earlier in practice because KV reads are per-user and KV memory fills up.
3. KV cache for Llama-3-8B at 8k, and how many users on 80 GB? 128 KB/token, 1.07 GB/user, ~59 users after 16 GB of weights.
4. TTFT vs TPOT: what improves each? TTFT (compute-bound prefill): chunked prefill, prefix caching, more compute. TPOT (memory-bound decode): batching, quantization, GQA, speculative decoding.
5. SLO is TTFT < 500 ms and TPOT < 50 ms. How do you pick the batch size? Sweep batch size, measure goodput, cap the scheduler at the knee where TPOT hits the SLO, add admission control.
6. What breaks at scale? KV memory fragmentation and OOM at long contexts; a long prefill stalls other users' decodes (TPOT spikes), fixed with chunked prefill or disaggregated prefill/decode.
