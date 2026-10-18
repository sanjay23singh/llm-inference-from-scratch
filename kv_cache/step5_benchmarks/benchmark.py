"""Step 5: benchmarks that explain WHY the cache helps and WHY decode is memory-bound.

A. Per-token latency vs sequence length   (naive grows, cached stays flat)
B. Throughput (tokens/s) vs tokens generated
C. Prefill vs decode: time per step and achieved GFLOP/s across prompt lengths
D. KV-cache memory math (printed table)

Outputs: PNG charts + benchmark.json in step5_benchmarks/results/
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

root = Path(__file__).parent.parent
for sub in ("step1_forward_pass", "step2_naive_generation", "step3_kv_cache"):
    sys.path.insert(0, str(root / sub))

from generate import generate_naive, sync       # noqa: E402  (step 2)
from generate_cached import generate_cached     # noqa: E402  (step 3)
from kv_model import CachedGPT2, GPT2, cache_bytes  # noqa: E402
from transformers import AutoTokenizer          # noqa: E402

OUT = Path(__file__).parent / "results"

# chart tokens: categorical slot 1 (blue) and slot 2 (orange), light surface, ink for all text
C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834")


# ------------------------------------------------------------------ plotting helpers
def set_style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "text.color": C["ink"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })


def style_axes(ax, xlabel, ylabel, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["axis"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=C["ink2"], fontsize=10)
    ax.set_ylabel(ylabel, color=C["ink2"], fontsize=10)
    if title:
        ax.set_title(title, loc="left", color=C["ink"], fontsize=12, fontweight="bold", pad=12)


def legend(ax, loc="upper left"):
    ax.legend(frameon=False, labelcolor=C["ink2"], fontsize=10, loc=loc)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points",
                va="center", color=C["ink"], fontsize=9)


def smooth(x, y, w=9):
    y = np.asarray(y, float)
    if len(y) < w:
        return np.asarray(x), y
    ys = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.asarray(x)[w // 2: w // 2 + len(ys)], ys


# ------------------------------------------------------------------ A + B: naive vs cached
def bench_latency(naive_model, cached_model, ids, n, dev):
    generate_naive(naive_model, ids, 3, dev)       # warm-up
    generate_cached(cached_model, ids, 3, dev)
    print(f"  running naive for {n} tokens (this is the slow one)...", flush=True)
    _, t_naive, _ = generate_naive(naive_model, ids, n, dev)
    print(f"  running cached for {n} tokens...", flush=True)
    _, ttft, t_dec, _ = generate_cached(cached_model, ids, n, dev)
    return t_naive, ttft, t_dec


def chart_latency(t_naive, ttft, t_dec, n_prompt, n):
    x = np.arange(n_prompt, n_prompt + n)
    t_cached = [ttft] + list(t_dec)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs, ys = smooth(x, np.array(t_cached) * 1000)
    ax.plot(xs, ys, color=C["s1"], linewidth=2, label="With KV cache")
    end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f} ms")
    xs, ys = smooth(x, np.array(t_naive) * 1000)
    ax.plot(xs, ys, color=C["s2"], linewidth=2, label="No cache (naive)")
    end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f} ms")
    ax.set_xlim(right=x[-1] * 1.12)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Sequence length (tokens)", "Latency per generated token (ms)",
               "Per-token latency vs sequence length")
    legend(ax)
    fig.text(0.01, 0.01, "9-step moving average", color=C["muted"], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "latency_per_token.png", dpi=160)
    plt.close(fig)


def throughput_points(t_naive, ttft, t_dec, n):
    pts = [k for k in (16, 32, 64, 128, 256, 512, 1000) if k <= n]
    if n not in pts:
        pts.append(n)
    rows = []
    for k in pts:
        rows.append(dict(tokens=k,
                         naive_tps=k / sum(t_naive[:k]),
                         cached_tps=k / (ttft + sum(t_dec[:k - 1]))))
    return rows


def chart_throughput(rows):
    x = [r["tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, [r["cached_tps"] for r in rows], color=C["s1"], linewidth=2,
            marker="o", markersize=8, markeredgecolor=C["surface"], markeredgewidth=2,
            label="With KV cache")
    ax.plot(x, [r["naive_tps"] for r in rows], color=C["s2"], linewidth=2,
            marker="o", markersize=8, markeredgecolor=C["surface"], markeredgewidth=2,
            label="No cache (naive)")
    end_label(ax, x[-1], rows[-1]["cached_tps"], f"{rows[-1]['cached_tps']:.0f} tok/s")
    end_label(ax, x[-1], rows[-1]["naive_tps"], f"{rows[-1]['naive_tps']:.0f} tok/s")
    ax.set_xlim(right=x[-1] * 1.18)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Tokens generated", "Throughput (tokens/s, whole run)",
               "Throughput vs generation length")
    legend(ax, "center right")
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ C: prefill vs decode
def matmul_params(cfg):
    d = cfg.n_embd
    return cfg.n_layer * 12 * d * d + cfg.vocab_size * d    # attention + MLP + lm_head


@torch.no_grad()
def bench_prefill_decode(model, dev, lengths, reps=3, decode_steps=10):
    P = matmul_params(model.cfg)
    g = torch.Generator().manual_seed(0)
    rows = []
    for L in lengths:
        print(f"  context length {L}...", flush=True)
        ids = torch.randint(0, model.cfg.vocab_size, (1, L), generator=g).to(dev)
        model(ids)                                          # warm-up
        pre = []
        for _ in range(reps):
            sync(dev)
            t0 = time.perf_counter()
            logits, cache = model(ids)
            sync(dev)
            pre.append(time.perf_counter() - t0)
        t_pre = statistics.median(pre)

        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        dec = []
        for _ in range(decode_steps):
            sync(dev)
            t0 = time.perf_counter()
            logits, cache = model(nxt, cache)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            sync(dev)
            dec.append(time.perf_counter() - t0)
        t_dec = statistics.median(dec)
        kv = cache_bytes(cache)
        rows.append(dict(
            context=L,
            prefill_ms=t_pre * 1e3, prefill_tok_s=L / t_pre,
            prefill_gflops=2 * P * L / t_pre / 1e9,
            decode_ms=t_dec * 1e3, decode_gflops=2 * P / t_dec / 1e9,
            decode_gb_s=(P * 4 + kv) / t_dec / 1e9,         # fp32 weights + KV read per step
            kv_mb=kv / 1e6,
        ))
    return rows


def chart_prefill_decode(rows):
    x = [r["context"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    kw = dict(linewidth=2, marker="o", markersize=8, markeredgecolor=C["surface"], markeredgewidth=2)

    a.plot(x, [r["prefill_ms"] for r in rows], color=C["s1"], label="Prefill (whole prompt, 1 pass)", **kw)
    a.plot(x, [r["decode_ms"] for r in rows], color=C["s2"], label="Decode (1 token per step)", **kw)
    a.set_yscale("log")
    a.set_ylim(10, 800)
    a.set_yticks([20, 50, 100, 200, 500])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(NullFormatter())
    style_axes(a, "Context length (tokens)", "Time per step (ms, log scale)", "Time per step")
    legend(a, "upper left")

    b.plot(x, [r["prefill_gflops"] for r in rows], color=C["s1"], **kw)
    b.plot(x, [r["decode_gflops"] for r in rows], color=C["s2"], **kw)
    end_label(b, x[-1], rows[-1]["prefill_gflops"], f"Prefill {rows[-1]['prefill_gflops']:.0f}")
    end_label(b, x[-1], rows[-1]["decode_gflops"], f"Decode {rows[-1]['decode_gflops']:.0f}")
    b.set_xlim(right=x[-1] * 1.3)
    b.set_ylim(bottom=0)
    style_axes(b, "Context length (tokens)", "Achieved GFLOP/s", "Achieved compute")

    fig.tight_layout()
    fig.savefig(OUT / "prefill_vs_decode.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ D: KV memory math
def kv_bytes_per_token(n_layer, n_kv_heads, head_dim, bytes_per):
    return 2 * n_layer * n_kv_heads * head_dim * bytes_per      # 2 = K and V


REFERENCE_CONFIGS = [   # (name, layers, kv_heads, head_dim, bytes) - approximate public architectures
    ("GPT-2 small, fp32 (this repo)", 12, 12, 64, 4),
    ("GPT-2 small, fp16", 12, 12, 64, 2),
    ("Llama-2-7B-like, fp16, MHA", 32, 32, 128, 2),
    ("Llama-2-70B-like, fp16, GQA (8 KV heads)", 80, 8, 128, 2),
]


def print_kv_table():
    print(f"\n{'config':<44}{'KB/token':>10}{'4k ctx, 1 seq':>16}{'4k ctx x 32 seqs':>20}")
    out = []
    for name, nl, nkv, hd, b in REFERENCE_CONFIGS:
        per = kv_bytes_per_token(nl, nkv, hd, b)
        one = per * 4096
        print(f"{name:<44}{per / 1024:>10.0f}{one / 1e9:>13.2f} GB{one * 32 / 1e9:>17.1f} GB")
        out.append(dict(config=name, kb_per_token=per / 1024, gb_4k_one_seq=one / 1e9))
    return out


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="The key to understanding LLM inference is")
    p.add_argument("--max_len", type=int, default=512, help="tokens to generate for charts A/B")
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1000])
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--quick", action="store_true", help="small run (~1 min) to check everything works")
    p.add_argument("--skip_latency", action="store_true")
    p.add_argument("--skip_prefill_decode", action="store_true")
    p.add_argument("--replot", action="store_true",
                   help="redraw prefill_vs_decode.png from saved benchmark.json (no model run)")
    args = p.parse_args()
    if args.replot:
        set_style()
        data = json.loads((OUT / "benchmark.json").read_text())
        chart_prefill_decode(data["prefill_decode"])
        print("Redrew prefill_vs_decode.png")
        return
    if args.quick:
        args.max_len, args.lengths = 96, [16, 64, 256]

    OUT.mkdir(exist_ok=True)
    set_style()
    dev = args.device
    results = {}

    tok = AutoTokenizer.from_pretrained("gpt2")
    cached_model = CachedGPT2.from_pretrained("gpt2").to(dev).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)
    n_prompt = ids.shape[1]

    if not args.skip_latency:
        print("\n[A/B] naive vs cached")
        naive_model = GPT2.from_pretrained("gpt2").to(dev).eval()
        n = args.max_len
        assert n_prompt + n <= cached_model.cfg.n_ctx, "max_len too large for GPT-2's 1024 context"
        t_naive, ttft, t_dec = bench_latency(naive_model, cached_model, ids, n, dev)
        chart_latency(t_naive, ttft, t_dec, n_prompt, n)
        rows = throughput_points(t_naive, ttft, t_dec, n)
        chart_throughput(rows)
        results["throughput"] = rows
        print(f"\n{'tokens':>8}{'naive tok/s':>14}{'cached tok/s':>14}{'speedup':>10}")
        for r in rows:
            print(f"{r['tokens']:>8}{r['naive_tps']:>14.1f}{r['cached_tps']:>14.1f}"
                  f"{r['cached_tps'] / r['naive_tps']:>9.1f}x")

    if not args.skip_prefill_decode:
        print("\n[C] prefill vs decode")
        lengths = [L for L in args.lengths if L + 10 <= cached_model.cfg.n_ctx]
        rows = bench_prefill_decode(cached_model, dev, lengths)
        chart_prefill_decode(rows)
        results["prefill_decode"] = rows
        print(f"\n{'ctx':>6}{'prefill ms':>12}{'prefill GFLOP/s':>17}{'decode ms':>11}"
              f"{'decode GFLOP/s':>16}{'decode GB/s':>13}{'KV MB':>8}")
        for r in rows:
            print(f"{r['context']:>6}{r['prefill_ms']:>12.1f}{r['prefill_gflops']:>17.1f}"
                  f"{r['decode_ms']:>11.1f}{r['decode_gflops']:>16.1f}{r['decode_gb_s']:>13.1f}"
                  f"{r['kv_mb']:>8.1f}")
        print("\nGB/s = (fp32 weights + KV cache) read per decode step / step time.")
        print("Compare it with your chip's memory bandwidth; GFLOP/s with its compute peak.")

    results["kv_reference"] = print_kv_table()
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved charts + benchmark.json to {OUT}")


if __name__ == "__main__":
    main()
