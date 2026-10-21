# Step 3: Dynamic batching

**What it adds to static.** A timeout: launch when the queue reaches B *or* the oldest request has waited `max_wait`.
Triton and TorchServe call this "dynamic batching". It fixes **when** a batch starts, not **what** it does afterwards.

## Predict, then run

```bash
python batching/step3_dynamic/sweep_wait.py        # ~4 min on a Mac
```

Two loads, same workload:

| Load | Prediction |
| --- | --- |
| light (0.25x capacity) | TTFT drops a lot (no waiting for the B-th arrival), but batches are tiny (~1-2 live rows per step): you traded batching for latency. Small `max_wait` = small batches, large = long waits. |
| saturated (1.5x) | the timeout never fires. Dynamic == static in every column. |

The `bucketed` row groups requests of similar (oracle) output length. It lowers zombie rows without iteration-level
scheduling, showing how much of continuous batching's win is "just" length awareness. A real system would have to
*predict* output length (research: S3, multi-bin batching), and a misprediction brings the zombies back.

Outputs: `results/dynamic_wait.json`, `results/dynamic_wait.png`.

## Reading the table
* `avg batch` = average **live** decode rows per step. At light load dynamic batching has *higher* slot efficiency than
  static only because its batches are almost empty (nothing to waste). Judge it together with `avg batch`.
* `queue p50` = time from arrival to getting a slot. For static at light load this is the batch-formation wait.

## You should be able to explain
1. What problem does `max_wait` solve, and what does it make worse?
2. Why is dynamic batching identical to static at saturation?
3. Why can't any grouping scheme (bucketing, sorting) fully fix zombie rows? What does iteration-level scheduling do that grouping can't?
