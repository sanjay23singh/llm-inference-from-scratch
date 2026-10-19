# Module 0: Foundations

| Step | Goal | Where | Status |
|------|------|-------|--------|
| 0a | Measure this machine's compute + bandwidth ceilings | `step0a_peaks/measure_peaks.py` | done |
| 0b | Roofline: place prefill and decode on it | `step0b_roofline/roofline.py` | done |
| 0c | KV cache sizing, real model configs, GQA | `step0c_kv_math/kv_calc.py` | done |
| 0d | Serving metrics: TTFT, TPOT, throughput, goodput | `step0d_metrics/metrics.py` | done |

## Key results (M1, CPU fp32)
- Ridge point 19.8 FLOP/byte. Decode intensity = 0.5 x batch, so decode flips to compute-bound at B ~ 40.
- Prefill intensity = 0.5 x prompt_len: prefill is decode with the prompt as the batch.
- KV bytes = 2 x layers x kv_heads x head_dim x bytes x tokens x batch.
- Llama-3-8B on an 80 GB GPU at 8k context: ~59 users with GQA vs ~14 without.
