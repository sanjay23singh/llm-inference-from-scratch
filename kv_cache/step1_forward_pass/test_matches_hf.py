"""Check that our from-scratch GPT-2 produces the same logits as HuggingFace's.

Uses real GPT-2 weights if they can be downloaded; otherwise falls back to a
small randomly initialised HF model (still a valid correctness test).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT2, GPT2Config  # noqa: E402

from transformers import GPT2Config as HFConfig, GPT2LMHeadModel  # noqa: E402

torch.manual_seed(0)

try:
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    print("Using real GPT-2 weights")
except Exception as e:  # offline / blocked network
    print(f"Could not download weights ({type(e).__name__}); using random small model")
    hf = GPT2LMHeadModel(HFConfig(n_layer=2, n_head=4, n_embd=64, n_positions=128)).eval()

c = hf.config
ours = GPT2(GPT2Config(c.vocab_size, c.n_positions, c.n_layer, c.n_head, c.n_embd)).eval()
ours.load_hf_state_dict(hf.state_dict())

idx = torch.randint(0, c.vocab_size, (2, 16))
with torch.no_grad():
    a = ours(idx)
    b = hf(idx).logits

max_diff = (a - b).abs().max().item()
print(f"max |ours - hf| = {max_diff:.2e}")
assert torch.allclose(a, b, atol=1e-4), "logits do not match"
print("PASS: logits match HuggingFace")
