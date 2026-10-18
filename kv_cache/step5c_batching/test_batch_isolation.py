"""Offline test: requests in a batch must not affect each other.

Run 3 prompts as one batch, and each prompt alone. The results must match,
because attention never mixes different requests (the batch axis is independent).
"""
import sys
from pathlib import Path

import torch

root = Path(__file__).parent.parent
for sub in ("step1_forward_pass", "step3_kv_cache"):
    sys.path.insert(0, str(root / sub))
from kv_model import CachedGPT2, GPT2Config  # noqa: E402

torch.manual_seed(0)
cfg = GPT2Config(vocab_size=200, n_ctx=64, n_layer=3, n_head=4, n_embd=64)
model = CachedGPT2(cfg).eval()

ids = torch.randint(0, cfg.vocab_size, (3, 10))
with torch.no_grad():
    # batched: prefill 3 prompts together, then decode 5 steps together
    lg, cache = model(ids)
    nxt = lg[:, -1, :].argmax(-1, keepdim=True)
    batched = [nxt]
    for _ in range(5):
        lg, cache = model(nxt, cache)
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        batched.append(nxt)
    batched = torch.cat(batched, dim=1)                       # (3, 6)

    # alone: the same 3 prompts, one at a time
    for i in range(3):
        lg, cache = model(ids[i:i + 1])
        nxt = lg[:, -1, :].argmax(-1, keepdim=True)
        alone = [nxt]
        for _ in range(5):
            lg, cache = model(nxt, cache)
            nxt = lg[:, -1, :].argmax(-1, keepdim=True)
            alone.append(nxt)
        alone = torch.cat(alone, dim=1)
        assert torch.equal(alone[0], batched[i]), f"request {i} changed when batched"

print("PASS: each request gives the same tokens alone or in a batch")
