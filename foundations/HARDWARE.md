# Hardware facts

Single source for hardware numbers. Blog posts, briefs and notes quote from here only. If a new measurement replaces a number, update it here first.

## Measured: Apple M1 (`step0a_peaks/results/peaks.json`)

Big matmul for FLOP/s, big copy for bandwidth. Not spec-sheet numbers.

| Device | dtype | Peak compute | Peak bandwidth | Ridge point |
| --- | --- | --- | --- | --- |
| CPU | fp32 | 907 GFLOP/s | 45.8 GB/s | 19.8 FLOP/byte |
| MPS | fp32 | 1,711 GFLOP/s | 58.9 GB/s | 29.0 FLOP/byte |
| MPS | fp16 | 2,310 GFLOP/s | 62.4 GB/s | 37.0 FLOP/byte |

Achieved by the engine (not peak): GPT-2 small decode at B = 1 streams weights at about 23 GB/s on CPU fp32, half the measured peak. Cause not profiled.

## Reference GPUs (approximate, spec sheet, not measured)

Dense fp16/bf16, no sparsity.

| GPU | Compute | Bandwidth | Ridge point |
| --- | --- | --- | --- |
| A100 80GB SXM | ~312 TFLOP/s | ~2.0 TB/s | ~156 FLOP/byte |
| H100 SXM | ~989 TFLOP/s | ~3.35 TB/s | ~300 FLOP/byte |

Always label these "approximate" when quoted.
