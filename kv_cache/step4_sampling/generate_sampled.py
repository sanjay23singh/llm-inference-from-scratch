"""Step 4: cached generation with sampling (reuses the Step 3 KV-cache model)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sampling import sample_next  # noqa: E402


@torch.no_grad()
def generate_sampled(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, seed=0):
    # sample on CPU so seeding behaves the same on cpu / mps / cuda (logits are only ~50k floats)
    gen = torch.Generator().manual_seed(seed)

    def pick(logits):
        nxt = sample_next(logits[:, -1, :].float().cpu(), temperature, top_k, top_p, gen)
        return nxt.to(idx.device)

    logits, cache = model(idx)                         # prefill
    nxt = pick(logits)
    out = torch.cat([idx, nxt], dim=1)
    for _ in range(max_new_tokens - 1):                # decode
        logits, cache = model(nxt, cache)
        nxt = pick(logits)
        out = torch.cat([out, nxt], dim=1)
    return out
