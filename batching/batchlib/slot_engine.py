"""The engine: a preallocated KV cache split into SLOTS, and one forward pass over a RAGGED batch.

Why this replaces the Step 3 model for batching
-----------------------------------------------
Step 3 (CachedGPT2) takes a (B, T) tensor and a cache that grows with torch.cat. That forces three things
that hurt in serving:

  1. Every sequence in the batch must have the same T (and same cache length) -> padding.
  2. Every step re-allocates and copies the whole cache (torch.cat). Cost grows with batch x context.
  3. Sequences cannot join or leave the batch; the cache tensor is one block.

Here:

  SlotKV        one fixed tensor per layer, shape (n_slots, heads, max_len, head_dim). A request owns one slot
                for its lifetime. A new K/V row is WRITTEN IN PLACE; nothing is ever copied or re-allocated.
  ragged batch  all tokens of all sequences in this step are concatenated into ONE (N, C) tensor
                (N = sum of per-sequence token counts). No padding. A sequence may contribute 1 token (decode)
                or many (a prefill chunk) in the same step.

Where batching actually pays off (the key mental model)
-------------------------------------------------------
Per layer a step does:   QKV proj -> attention -> output proj -> MLP.

  * QKV proj, output proj, MLP, lm_head are matmuls against WEIGHTS shared by every sequence.
    One call over all N tokens reads the weights from memory ONCE. This is where batching is "almost free"
    (Part 2): decode is memory-bound, so N tokens cost about the same as 1 until you hit the ridge point.
  * Attention reads each sequence's OWN K/V. Nothing is shared between sequences, so batching does not
    reduce its memory traffic; the cost is sum over sequences of (their context length). We loop over
    sequences here. Real engines fuse that loop into one kernel (FlashAttention varlen / PagedAttention),
    but the bytes moved are the same.

This is Orca's "selective batching": batch the weight-sharing ops across sequences, run attention per sequence.
"""
import time

import torch

from . import paths  # noqa: F401
from model import GPT2  # noqa: E402  (kv_cache/step1_forward_pass: same weights + HF loader as before)


def sync(device: str):
    """Accelerator calls are async; wait for them so timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class SlotKV:
    """Preallocated K/V for n_slots sequences.

    Cost of this design: a slot reserves max_len tokens even if its request uses 30 of them.
    That unused reservation is INTERNAL FRAGMENTATION; paying for it is what PagedAttention (Module 3) removes.
    """

    def __init__(self, cfg, n_slots, max_len, device="cpu", dtype=torch.float32):
        assert max_len <= cfg.n_ctx, "slot length cannot exceed the model's position table"
        self.n_slots, self.max_len, self.device = n_slots, max_len, device
        self.head_dim = cfg.n_embd // cfg.n_head
        shape = (n_slots, cfg.n_head, max_len, self.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.bytes_per_token = 2 * cfg.n_layer * cfg.n_embd * torch.finfo(dtype).bits // 8

    def reserved_bytes(self):
        return self.n_slots * self.max_len * self.bytes_per_token


def _causal_mask(n, start, device):
    """(n, start+n) bool. Query i sits at absolute position start+i, so it may see keys j <= start+i.
    start=0 -> ordinary lower-triangular mask. start>0 -> a prefill CHUNK that also sees earlier chunks."""
    S = start + n
    return torch.arange(S, device=device)[None, :] <= (start + torch.arange(n, device=device))[:, None]


@torch.no_grad()
def ragged_forward(model: GPT2, kv: SlotKV, works, return_logits=False):
    """Run one engine step. works: list[Work]. Returns next-token ids aligned with works (None if no sample).
    return_logits=True also returns the (R, vocab) logits of the sampled rows (used by tests only).

    Greedy argmax only: the scheduling experiments need token streams, not sampling variety.
    (Per-request sampling params plug in here later via step4's sample_next.)
    """
    cfg, dev = model.cfg, kv.device
    C, H = cfg.n_embd, cfg.n_head
    d = C // H
    scale = 1.0 / (d ** 0.5)

    # ---- flatten every sequence's new tokens into one axis ------------------------------------------
    ids, pos, segs, off = [], [], [], 0
    for w in works:
        n = len(w.tokens)
        assert w.start + n <= kv.max_len, f"slot overflow: {w.start}+{n} > {kv.max_len}"
        ids += w.tokens
        pos += range(w.start, w.start + n)      # positions CONTINUE after what is already cached
        segs.append((w.req.slot, w.start, n, off))
        off += n
    N = off
    x = model.wte(torch.tensor(ids, device=dev)) + model.wpe(torch.tensor(pos, device=dev))   # (N, C)

    # where every new K/V row goes: (slot, position) per token, so ALL writes of a layer are ONE indexing op
    slot_idx = torch.tensor([sl for sl, st, n, _ in segs for _ in range(n)], device=dev)
    pos_idx = torch.tensor(pos, device=dev)

    masks = {}                                   # one mask per (n, start) per step, reused by all 12 layers
    for li, blk in enumerate(model.h):
        a = blk.attn
        # ONE matmul for all N tokens of all sequences: the weights are read from memory once.
        q, k, v = a.c_attn(blk.ln_1(x)).split(C, dim=-1)
        q, k, v = (t.view(N, H, d) for t in (q, k, v))
        y = torch.empty_like(q)
        kc, vc = kv.k[li], kv.v[li]
        kc[slot_idx, :, pos_idx] = k             # in-place write of every new K row of every sequence (no torch.cat)
        vc[slot_idx, :, pos_idx] = v

        # attention is per-sequence: each reads only its own slot's K/V
        for slot, start, n, o in segs:
            K = kc[slot, :, :start + n]                                   # (H, S, d) a VIEW, no copy
            V = vc[slot, :, :start + n]
            att = (q[o:o + n].transpose(0, 1) @ K.transpose(-2, -1)) * scale   # (H, n, S)
            if n > 1:                                                     # decode (n=1) sees everything
                key = (n, start)
                if key not in masks:
                    masks[key] = _causal_mask(n, start, dev)
                att = att.masked_fill(~masks[key], float("-inf"))
            y[o:o + n] = (att.softmax(-1) @ V).transpose(0, 1)           # (n, H, d)

        x = x + a.c_proj(y.reshape(N, C))                                 # again one matmul for everyone
        x = x + blk.mlp(blk.ln_2(x))

    # lm_head is (C x 50257) = the single largest matrix. Only rows that need a next token go through it.
    # (Step 3's prefill computed logits for EVERY prompt position and threw all but the last away.)
    want = [i for i, w in enumerate(works) if w.sample]
    out, logits = [None] * len(works), None
    if want:
        last = torch.tensor([segs[i][3] + segs[i][2] - 1 for i in want], device=dev)
        logits = model.lm_head(model.ln_f(x[last]))
        for i, t in zip(want, logits.argmax(-1).tolist()):
            out[i] = t
    return (out, logits) if return_logits else out


class RealEngine:
    """Runs steps for real and reports wall-clock time. The scheduler advances its clock by this time."""
    kind = "real"

    def __init__(self, model: GPT2, n_slots, max_len, device="cpu"):
        self.model, self.device = model, device
        self.n_slots, self.max_len = n_slots, max_len
        self.kv = SlotKV(model.cfg, n_slots, max_len, device)

    def execute(self, works):
        sync(self.device)
        t0 = time.perf_counter()
        toks = ragged_forward(self.model, self.kv, works)
        sync(self.device)
        return toks, time.perf_counter() - t0
