"""Step 2: naive greedy generation (no KV cache).

Each new token requires a full forward pass over the ENTIRE sequence so far.
Step k processes (prompt + k) tokens, so total work grows quadratically.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "step1_forward_pass"))
from model import GPT2  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def sync(device: str):
    """GPU calls are async; wait for them so timings are honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens, device="cpu"):
    step_times = []          # seconds per generated token
    tokens_processed = 0     # total tokens pushed through the model
    for _ in range(max_new_tokens):
        sync(device)
        t0 = time.perf_counter()

        logits = model(idx)                                   # recompute ALL positions
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # greedy: take the top token
        tokens_processed += idx.shape[1]
        idx = torch.cat([idx, nxt], dim=1)

        sync(device)
        step_times.append(time.perf_counter() - t0)
    return idx, step_times, tokens_processed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The key to understanding LLM inference is")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2.from_pretrained("gpt2").to(args.device).eval()

    idx = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    n_prompt = idx.shape[1]
    assert n_prompt + args.max_new_tokens <= model.cfg.n_ctx, "exceeds GPT-2 context (1024)"

    out, times, processed = generate_naive(model, idx, args.max_new_tokens, args.device)

    print("\n--- output ---")
    print(tok.decode(out[0]))

    n = args.max_new_tokens
    total = sum(times)
    print("\n--- stats (naive, no cache) ---")
    print(f"prompt tokens: {n_prompt} | generated: {n}")
    print(f"total time: {total:.2f}s | throughput: {n / total:.1f} tokens/s")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = min(int(frac * (n - 1)), n - 1)
        print(f"  step {i + 1:>4} (seq len {n_prompt + i:>4}): {times[i] * 1000:7.1f} ms")
    print(f"tokens pushed through model: {processed} "
          f"(vs {n} actually new -> {processed / n:.0f}x redundant work)")


if __name__ == "__main__":
    main()
