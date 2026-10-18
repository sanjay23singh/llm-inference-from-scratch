"""Step 4: see what each sampling knob does."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

root = Path(__file__).parent.parent
for sub in ("step1_forward_pass", "step3_kv_cache", "step4_sampling"):
    sys.path.insert(0, str(root / sub))

from generate_sampled import generate_sampled  # noqa: E402
from kv_model import CachedGPT2                 # noqa: E402
from sampling import top_k_filter, top_p_filter  # noqa: E402
from transformers import AutoTokenizer          # noqa: E402


def distinct_ngrams(ids, n=4):
    """Fraction of unique n-grams. Near 0 = stuck in a loop, near 1 = never repeats."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The key to understanding LLM inference is")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = CachedGPT2.from_pretrained("gpt2").to(args.device).eval()
    idx = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)

    # ---- 1. the next-token distribution, reshaped by each knob ----
    with torch.no_grad():
        last = model(idx)[0][:, -1, :].float().cpu()
    print(f'Next-token distribution after: "{args.prompt}"\n')

    def show(name, lg):
        probs = F.softmax(lg, dim=-1)[0]
        top = torch.topk(probs, 5)
        kept = int(torch.isfinite(lg).sum())
        cells = "  ".join(f"{tok.decode(i)!r}:{v:.2f}" for v, i in zip(top.values, top.indices))
        print(f"{name:<18} kept={kept:>5}  {cells}")

    show("raw (T=1)", last)
    show("T=0.5", last / 0.5)
    show("T=1.5", last / 1.5)
    show("top_k=40", top_k_filter(last, 40))
    show("top_p=0.9", top_p_filter(last, 0.9))

    # ---- 2. full generations under different settings ----
    configs = [
        ("greedy (T=0)",        dict(temperature=0)),
        ("T=0.7",               dict(temperature=0.7)),
        ("T=1.0",               dict(temperature=1.0)),
        ("T=1.5",               dict(temperature=1.5)),
        ("top_k=40",            dict(temperature=1.0, top_k=40)),
        ("top_p=0.9",           dict(temperature=1.0, top_p=0.9)),
        ("T=0.8 + top_p=0.95",  dict(temperature=0.8, top_p=0.95)),
    ]
    for name, kw in configs:
        out = generate_sampled(model, idx, args.max_new_tokens, seed=args.seed, **kw)
        new_ids = out[0, idx.shape[1]:].tolist()
        print(f"\n=== {name}  | distinct-4gram: {distinct_ngrams(new_ids):.2f} ===")
        print(tok.decode(new_ids).strip())


if __name__ == "__main__":
    main()
