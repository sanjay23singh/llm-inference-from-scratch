"""Offline test (no downloads): cached forward must equal the Step 1 forward.

1. prefill(prompt[:-1]) then decode(last token) == full forward on the whole prompt
2. several decode steps in a row stay equal to the full forward
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kv_model import CachedGPT2, GPT2, GPT2Config  # noqa: E402

torch.manual_seed(0)
cfg = GPT2Config(vocab_size=200, n_ctx=64, n_layer=3, n_head=4, n_embd=64)
cached = CachedGPT2(cfg).eval()
plain = GPT2(cfg).eval()
plain.load_state_dict(cached.state_dict())   # identical weights

idx = torch.randint(0, cfg.vocab_size, (2, 20))
with torch.no_grad():
    full = plain(idx)                                        # (2, 20, vocab)

    logits_p, cache = cached(idx[:, :10])                    # prefill 10 tokens
    assert torch.allclose(logits_p, full[:, :10], atol=1e-5), "prefill mismatch"

    for t in range(10, 20):                                  # decode 10 tokens, one at a time
        logits_d, cache = cached(idx[:, t:t + 1], cache)
        assert torch.allclose(logits_d[:, 0], full[:, t], atol=1e-5), f"decode mismatch at {t}"

print("PASS: prefill + decode with cache matches full forward")
