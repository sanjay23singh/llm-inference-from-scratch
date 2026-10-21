"""Step 6 experiment: long prompts stall everyone's decode. How much, and what does chunking cost?

Traffic: mostly short prompts, but 15% are LONG (480 tokens). Each long prompt is ~5x a normal one; as one
uninterrupted prefill step it takes as long as dozens of decode steps.

Compared:
  continuous (prefill-first)   whole prompt, decodes stall for the full prefill step
  continuous mixed (Orca)      whole prompt shares the step with decodes; the step is as slow as the prompt
  chunked, budget = 32 ... 512 tokens per step; decodes always included

Predictions before you run it:
  * ITL p99 / max: prefill-first and mixed show a spike about the size of a full long-prompt prefill.
    Chunked caps every step at ~ the time for `budget` tokens, so ITL max tracks the budget.
  * The price is paid by the long prompt itself: it needs ceil(480/budget) steps, so ITL falls and TTFT of
    long requests rises as the budget shrinks. And a tiny budget wastes the memory-bound regime: each chunk
    re-streams the weights for few tokens.
  * The knee in the step-time curve (sim_engine.fit) says where extra tokens stop being free. Budgets near it
    keep decode ITL low at almost no TTFT cost: that is how vLLM/SGLang pick their default.

Outputs: results/stall.json, stall_tradeoff.png (TTFT of long prompts vs ITL p99 per budget),
stall_steps.png (duration of every engine step over time).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, POLICY_COLOR, LABEL, end_label, legend, set_style, style_axes  # noqa: E402
from batchlib.registry import make_policy  # noqa: E402
from batchlib.runner import get_engine, meta, rate_for_load, run_one, save  # noqa: E402
from batchlib.workload import WorkloadSpec, describe, make_requests  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "sim"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--load", type=float, default=0.5)
    ap.add_argument("--long_frac", type=float, default=0.15)
    ap.add_argument("--long_len", type=int, default=480)
    ap.add_argument("--out_mean", type=float, default=32)
    ap.add_argument("--out_cap", type=int, default=64)
    ap.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    spec = WorkloadSpec(n=args.n, prompt_min=16, prompt_max=96, long_frac=args.long_frac, long_len=args.long_len,
                        out_mean=args.out_mean, out_sigma=0.6, out_cap=args.out_cap)
    eng, wk = get_engine(args.engine, args.slots, spec.max_total(), args.device, args.weights)
    spec.rate, t_step = rate_for_load(args.load, eng, spec)
    reqs = make_requests(spec)
    is_long = np.array([len(r.prompt) >= args.long_len for r in reqs])
    print(f"decode step {t_step*1e3:.0f} ms | rate {spec.rate:.2f} req/s | {is_long.sum()} of {len(reqs)} prompts are "
          f"{args.long_len} tokens long\n")

    runs = [("continuous", dict(), "prefill-first"), ("continuous_mixed", dict(), "mixed")]
    runs += [("chunked", dict(token_budget=b), f"chunked {b}") for b in args.budgets]

    rows, steps_keep = [], {}
    print(f"{'policy':<16}{'long TTFT p50':>14}{'short TTFT p50':>15}{'ITL p50':>9}{'ITL p99':>9}{'ITL max':>9}"
          f"{'tok/s':>7}{'max step':>10}")
    for name, kw, label in runs:
        res, pr, steps = run_one(make_policy(name, args.slots, **kw), eng, spec, keep_steps=True)
        ttft = np.array(pr["ttft"])
        res.update(label=label, budget=kw.get("token_budget"),
                   long_ttft_p50=float(np.median(ttft[is_long])), short_ttft_p50=float(np.median(ttft[~is_long])),
                   max_step_s=max(s["dt"] for s in steps))
        rows.append(res)
        steps_keep[label] = [(s["t"], s["dt"]) for s in steps]
        print(f"{label:<16}{res['long_ttft_p50']:>13.2f}s{res['short_ttft_p50']:>14.2f}s{res['itl_p50']*1e3:>8.0f}m"
              f"{res['itl_p99']*1e3:>8.0f}m{res['itl_max']*1e3:>8.0f}m{res['throughput_tps']:>7.1f}"
              f"{res['max_step_s']*1e3:>9.0f}m")

    save(OUT / "stall.json", dict(meta=meta(args.device, wk, dict(engine=args.engine)), spec=spec.__dict__,
                                  workload=describe(reqs), rows=rows))

    set_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ch = [r for r in rows if r["budget"]]
    ax.plot([r["long_ttft_p50"] for r in ch], [r["itl_p99"] * 1e3 for r in ch], color=POLICY_COLOR["chunked"], linewidth=2,
            marker="o", markersize=8, markeredgecolor=C["surface"], markeredgewidth=1.5)
    for r in ch:
        ax.annotate(f"budget {r['budget']}", (r["long_ttft_p50"], r["itl_p99"] * 1e3), xytext=(7, 6),
                    textcoords="offset points", fontsize=9, color=C["ink2"])
    for r, key in zip(rows[:2], ("continuous", "continuous_mixed")):
        ax.scatter([r["long_ttft_p50"]], [r["itl_p99"] * 1e3], s=70, color=POLICY_COLOR[key], edgecolor=C["surface"],
                   linewidth=1.5, zorder=3)
        ax.annotate(LABEL[key], (r["long_ttft_p50"], r["itl_p99"] * 1e3), xytext=(7, -12), textcoords="offset points",
                    fontsize=9, color=C["ink2"])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style_axes(ax, "TTFT of the long (480-token) prompts, p50 (s)", "Inter-token latency p99 (ms)",
               "Smaller budget: smoother decode, slower long prompts")
    fig.tight_layout()
    fig.savefig(OUT / "stall_tradeoff.png", dpi=160)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
    pick = [("prefill-first", "continuous"), ("mixed", "continuous_mixed"), (f"chunked {args.budgets[len(args.budgets)//2]}", "chunked")]
    for ax, (lab, key) in zip(axes, pick):
        t, dt = zip(*steps_keep[lab])
        ax.vlines(t, 0, np.array(dt) * 1e3, color=POLICY_COLOR[key], linewidth=1.2)
        style_axes(ax, "", "Step duration (ms)", lab)
    axes[-1].set_xlabel("Time (s)", color=C["ink2"], fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "stall_steps.png", dpi=160)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
