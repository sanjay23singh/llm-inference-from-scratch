"""Step 1: GPT-2 forward pass in plain PyTorch.

Pipeline: token ids -> embeddings -> N x (attention + MLP) -> final norm -> logits.
No KV cache yet (that is Step 3). Every forward call recomputes everything.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_ctx: int = 1024      # max sequence length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # fused Q, K, V projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scores: how much each position attends to every other position
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))  # causal: no peeking at future tokens
        att = F.softmax(att, dim=-1)

        y = att @ v                                   # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # residual connection around attention
        x = x + self.mlp(self.ln_2(x))    # residual connection around MLP
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.n_embd)       # position embeddings
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight                # weight tying

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))                    # (B, T, vocab) logits

    @classmethod
    def from_pretrained(cls, name: str = "gpt2"):
        """Load HuggingFace GPT-2 weights into this from-scratch model."""
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name)
        c = hf.config
        cfg = GPT2Config(c.vocab_size, c.n_positions, c.n_layer, c.n_head, c.n_embd)
        model = cls(cfg)
        model.load_hf_state_dict(hf.state_dict())
        return model

    def load_hf_state_dict(self, hf_sd):
        """Copy weights from a HF GPT2LMHeadModel state dict.

        HF GPT-2 uses Conv1D layers, which store weights transposed
        relative to nn.Linear, so those four need a .t().
        """
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight",
                      "mlp.c_fc.weight", "mlp.c_proj.weight")
        sd = self.state_dict()
        for key in sd:
            hf_key = "transformer." + key if key != "lm_head.weight" else "lm_head.weight"
            w = hf_sd[hf_key]
            if key.endswith(transposed):
                w = w.t()
            with torch.no_grad():
                sd[key].copy_(w)
