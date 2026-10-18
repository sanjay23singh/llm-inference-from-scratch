"""Step 4: turning logits into the next token.

Greedy (argmax) is deterministic and tends to loop. Sampling draws from the
model's probability distribution instead. Three knobs reshape that distribution:

  temperature : divide logits by T. T<1 sharpens, T>1 flattens, T->0 is greedy.
  top-k       : keep only the k most likely tokens.
  top-p       : keep the smallest set of tokens whose probabilities sum to >= p.

Order used here (same as HuggingFace): temperature -> top-k -> top-p -> sample.
All functions take logits of shape (B, vocab).
"""
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, NEG_INF)


def top_p_filter(logits, p):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs          # probability mass ahead of each token
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, NEG_INF)
    # put the (possibly masked) values back in original vocab order
    return torch.full_like(logits, NEG_INF).scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """logits: (B, vocab) -> next token ids (B, 1)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
