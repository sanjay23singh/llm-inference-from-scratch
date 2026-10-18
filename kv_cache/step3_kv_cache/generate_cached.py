"""Step 3: greedy generation with a KV cache (prefill + decode)."""
import time

import torch


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, device="cpu"):
    from kv_model import cache_bytes

    # ---- prefill: process the whole prompt once, build the cache ----
    sync(device)
    t0 = time.perf_counter()
    logits, cache = model(idx)
    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, nxt], dim=1)
    sync(device)
    ttft = time.perf_counter() - t0          # time to first token

    # ---- decode: ONE token per step, reuse the cache ----
    decode_times = []
    for _ in range(max_new_tokens - 1):
        sync(device)
        t0 = time.perf_counter()
        logits, cache = model(nxt, cache)    # only the newest token goes in
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        sync(device)
        decode_times.append(time.perf_counter() - t0)

    return idx, ttft, decode_times, cache_bytes(cache)
