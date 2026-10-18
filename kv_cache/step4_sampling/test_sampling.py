"""Offline unit tests for the sampling functions (no model, no downloads)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sampling import sample_next, top_k_filter, top_p_filter  # noqa: E402

probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
logits = probs.log()

# top-k keeps exactly k tokens
assert torch.isfinite(top_k_filter(logits, 2)).sum() == 2
assert torch.isfinite(top_k_filter(logits, 0)).sum() == 4          # 0 = disabled

# top-p keeps the smallest set whose mass reaches p
assert torch.isfinite(top_p_filter(logits, 0.75)).sum() == 2       # 0.5 + 0.3
assert torch.isfinite(top_p_filter(logits, 0.90)).sum() == 3       # + 0.15
assert torch.isfinite(top_p_filter(logits, 1.00)).sum() == 4       # disabled
assert torch.isfinite(top_p_filter(logits, 0.01)).sum() == 1       # always keep the top token

# top-p must preserve token order (scatter back to original positions)
shuffled = torch.tensor([[0.05, 0.5, 0.15, 0.3]]).log()
kept = torch.isfinite(top_p_filter(shuffled, 0.75))
assert kept.tolist() == [[False, True, False, True]]

# temperature 0 == greedy; top_k=1 == greedy
assert sample_next(logits, temperature=0).item() == 0
g = torch.Generator().manual_seed(0)
assert sample_next(logits, temperature=1.0, top_k=1, generator=g).item() == 0

# at T=1 samples follow the distribution; low T collapses to the top token
g = torch.Generator().manual_seed(0)
draws = torch.cat([sample_next(logits, 1.0, generator=g) for _ in range(4000)])
freq0 = (draws == 0).float().mean().item()
assert abs(freq0 - 0.5) < 0.03, freq0
g = torch.Generator().manual_seed(0)
draws = torch.cat([sample_next(logits, 0.01, generator=g) for _ in range(500)])
assert (draws == 0).all()

print("PASS: sampling functions behave as expected")
