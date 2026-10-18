"""Step 3: naive vs cached generation, side by side."""
import argparse
import sys
from pathlib import Path

import torch

root = Path(__file__).parent.parent
for sub in ("step1_forward_pass", "step2_naive_generation", "step3_kv_cache"):
    sys.path.insert(0, str(root / sub))

from generate import generate_naive          # noqa: E402  (step 2)
from generate_cached import generate_cached  # noqa: E402
from kv_model import CachedGPT2, GPT2        # noqa: E402
from transformers import AutoTokenizer       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The key to understanding LLM inference is")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()
    n, dev = args.max_new_tokens, args.device

    tok = AutoTokenizer.from_pretrained("gpt2")
    naive_model = GPT2.from_pretrained("gpt2").to(dev).eval()
    cached_model = CachedGPT2.from_pretrained("gpt2").to(dev).eval()
    idx = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
    n_prompt = idx.shape[1]

    # warm-up so first-call overhead does not distort timings
    generate_naive(naive_model, idx, 3, dev)
    generate_cached(cached_model, idx, 3, dev)

    out_n, times_n, _ = generate_naive(naive_model, idx, n, dev)
    out_c, ttft, dec, kv_bytes = generate_cached(cached_model, idx, n, dev)

    same = torch.equal(out_n, out_c)
    total_n = sum(times_n)
    total_c = ttft + sum(dec)

    print("\n--- output (cached) ---")
    print(tok.decode(out_c[0]))
    print(f"\nsame tokens as naive: {same}")
    print(f"\nprompt tokens: {n_prompt} | generated: {n}")
    print(f"naive : {total_n:6.2f}s | {n / total_n:6.1f} tokens/s")
    print(f"cached: {total_c:6.2f}s | {n / total_c:6.1f} tokens/s  ({total_n / total_c:.1f}x faster)")
    print(f"\ncached breakdown:")
    print(f"  prefill / TTFT      : {ttft * 1000:7.1f} ms  (whole prompt, 1 pass)")
    print(f"  decode, first steps : {sum(dec[:10]) / 10 * 1000:7.1f} ms/token")
    print(f"  decode, last steps  : {sum(dec[-10:]) / 10 * 1000:7.1f} ms/token  (should barely grow)")
    print(f"  KV cache size       : {kv_bytes / 1e6:7.2f} MB for {n_prompt + n - 1} tokens "
          f"({kv_bytes / (n_prompt + n - 1) / 1024:.0f} KB/token)")


if __name__ == "__main__":
    main()
