# Step 5: Padding vs ragged batches

**Problem.** A `(B, T)` tensor needs equal-length rows. Requests aren't, so classic batching pads short rows to the
longest and masks the pads out. The pads still go through every matmul.

**Ragged batch.** Concatenate the real tokens into `(N,)`, `N = sum(T_i)`, and carry per-token positions and slot
ids (`slot_engine.ragged_forward`). Attention is per sequence, so no mask is needed to hide pads: there are none.
This is the "selective batching" half of Orca, and it is also what FlashAttention-varlen / `cu_seqlens` APIs expose.

## Predict before you measure

| Phase | Bound by | Padding cost | Predicted ragged speedup |
| --- | --- | --- | --- |
| prefill | compute: time ~ tokens | pads are real work | `B x T_max / sum(T_i)` |
| decode | memory: weights streamed once either way | the new token is 1 row regardless; but a padded *cache* makes attention read `T_max` positions of K/V for every row | only the K/V-bytes ratio, so small when weights dominate |

```bash
python batching/step5_ragged/padding_vs_ragged.py    # ~1 min on a Mac
```

Outputs: `results/padding.json`, `results/padding.png`. The dashed lines are the computed predictions.

## How the padded cost is measured
The padded batch is fed to the *same* engine with every row stretched to `T_max` dummy tokens. Pads run through the
same kernels as real tokens, so this is the cost of a masked padded tensor (an implementation that built the mask
would also add a bit more). Honest limitation: decode attention is a small share of a GPT-2 small step at these
lengths, so the decode speedup is expected to be near 1x and noisy. It grows with `batch x context`, i.e. exactly
when KV traffic starts to dominate (long contexts, big batches, Module 3).

## You should be able to explain
1. Why does padding cost nothing extra for the *new* token in decode but a lot in prefill?
2. A batch of 8 prompts has lengths 40..160. What fraction of prefill FLOPs is padding?
3. What would you need to change to run a *padded* batch correctly (mask, position ids)? Why is left-padding common for generation?
4. Why can a ragged batch mix prefill and decode but a padded one can't?
