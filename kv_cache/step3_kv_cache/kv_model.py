"""Step 3: GPT-2 with a KV cache.

Same weights and same parameter names as Step 1, so Step 1's loader still works.
Only the forward pass changes: attention can now take a cache of past K and V.

  prefill: feed the whole prompt once  -> builds the cache, parallel over tokens
  decode : feed ONE new token per step -> reads the cache, appends one K/V row
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "step1_forward_pass"))
from model import GPT2, Block, CausalSelfAttention, GPT2Config  # noqa: E402,F401


class CachedAttention(CausalSelfAttention):
    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:                      # append new K/V to the past ones
            k = torch.cat([cache[0], k], dim=2)    # (B, H, S, d), S = past + T
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        S = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (B, H, T, S)
        # query i sits at absolute position S-T+i, so it may see keys j <= S-T+i
        mask = torch.tril(torch.ones(T, S, device=x.device, dtype=torch.bool), diagonal=S - T)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_cache


class CachedBlock(Block):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.attn = CachedAttention(cfg)           # same name -> same state_dict keys

    def forward(self, x, cache=None):
        a, new_cache = self.attn(self.ln_1(x), cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class CachedGPT2(GPT2):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.h = nn.ModuleList([CachedBlock(cfg) for _ in range(cfg.n_layer)])

    def forward(self, idx, cache=None):
        """idx: (B, T) new tokens only. cache: list of (K, V) per layer, or None.
        Returns (logits, new_cache)."""
        B, T = idx.shape
        past = 0 if cache is None else cache[0][0].shape[2]
        pos = torch.arange(past, past + T, device=idx.device)   # positions continue after the cache
        x = self.wte(idx) + self.wpe(pos)
        new_cache = []
        for i, block in enumerate(self.h):
            x, c = block(x, None if cache is None else cache[i])
            new_cache.append(c)
        return self.lm_head(self.ln_f(x)), new_cache


def cache_bytes(cache):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)
