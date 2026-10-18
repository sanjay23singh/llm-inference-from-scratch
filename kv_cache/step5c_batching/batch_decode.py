"""Step 5c: how does decode scale with batch size?

For batch sizes 1, 2, 4, ... : prefill B prompts together, then time decode steps.
Compare with the 'one at a time' baseline = B x (step time at B=1).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

root = Path(__file__).parent.parent
for sub in ("step1_forward_pass", "step2_naive_generation", "step3_kv_cache", "step5_benchmarks"):
    sys.path.insert(0, str(root / sub))

from generate import sync                                       # noqa: E402
from kv_model import CachedGPT2, cache_bytes                    # noqa: E402
from benchmark import C, set_style, style_axes, legend, end_label  # noqa: E402

OUT = Path(__file__).parent / "results"


@torch.no_grad()
def measure(model, B, prompt_len, steps, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.cfg.vocab_size, (B, prompt_len), generator=g).to(dev)
    logits, cache = model(ids)                               # prefill all B prompts together
    nxt = logits[:, -1, :].argmax(-1, keepdim=True)          # (B, 1): one new token per user
    times = []
    for _ in range(steps + 2):                               # first 2 steps are warm-up
        sync(dev)
        t0 = time.perf_counter()
        logits, cache = model(nxt, cache)                    # ONE step for all B users
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        sync(dev)
        times.append(time.perf_counter() - t0)
    t = statistics.median(times[2:])
    return dict(batch=B, step_ms=t * 1e3, total_tps=B / t, per_user_tps=1 / t,
                kv_mb=cache_bytes(cache) / 1e6)


def make_chart(rows):
    x = [r["batch"] for r in rows]
    t1 = rows[0]["step_ms"]
    seq_ms = [r["batch"] * t1 for r in rows]                 # serve users one after another
    seq_tps = rows[0]["total_tps"]                           # flat: no gain from more users
    kw = dict(linewidth=2, marker="o", markersize=8, markeredgecolor=C["surface"], markeredgewidth=2)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    a.plot(x, [r["step_ms"] for r in rows], color=C["s1"], label="Batched (one step for all users)", **kw)
    a.plot(x, seq_ms, color=C["s2"], label="One at a time (B x single-user step)", **kw)
    end_label(a, x[-1], rows[-1]["step_ms"], f"{rows[-1]['step_ms']:.0f} ms")
    end_label(a, x[-1], seq_ms[-1], f"{seq_ms[-1]:.0f} ms")

    b.plot(x, [r["total_tps"] for r in rows], color=C["s1"], **kw)
    b.plot(x, [seq_tps] * len(x), color=C["s2"], **kw)
    end_label(b, x[-1], rows[-1]["total_tps"], f"Batched {rows[-1]['total_tps']:.0f}")
    end_label(b, x[-1], seq_tps, f"One at a time {seq_tps:.0f}")

    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(x[0] * 0.8, x[-1] * 2.6)
        ax.set_ylim(bottom=0)
    style_axes(a, "Batch size (users decoded together)", "Time per decode step (ms)", "Time per step")
    legend(a, "upper left")
    style_axes(b, "Batch size (users decoded together)", "Total tokens/s across all users",
               "Total throughput")
    fig.tight_layout()
    fig.savefig(OUT / "batching.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--prompt_len", type=int, default=32)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()
    assert args.batch_sizes[0] == 1, "first batch size must be 1 (it is the baseline)"

    OUT.mkdir(exist_ok=True)
    set_style()
    model = CachedGPT2.from_pretrained("gpt2").to(args.device).eval()

    rows = []
    for B in args.batch_sizes:
        print(f"  batch {B}...", flush=True)
        rows.append(measure(model, B, args.prompt_len, args.steps, args.device))

    t1 = rows[0]["step_ms"]
    print(f"\n{'batch':>6}{'ms/step':>10}{'total tok/s':>13}{'per-user tok/s':>16}"
          f"{'one-at-a-time ms':>18}{'gain':>7}{'KV MB':>8}")
    for r in rows:
        seq = r["batch"] * t1
        r["gain_vs_sequential"] = seq / r["step_ms"]
        print(f"{r['batch']:>6}{r['step_ms']:>10.1f}{r['total_tps']:>13.1f}{r['per_user_tps']:>16.1f}"
              f"{seq:>18.1f}{r['gain_vs_sequential']:>6.1f}x{r['kv_mb']:>8.1f}")

    make_chart(rows)
    (OUT / "batching.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved batching.png and batching.json to {OUT}")


if __name__ == "__main__":
    main()
