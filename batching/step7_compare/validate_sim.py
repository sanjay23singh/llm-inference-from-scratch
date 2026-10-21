"""Step 7b: how far can we trust the simulator?

Run the same workload under every policy twice: once on the real engine, once on SimEngine (cost model from
calibrate.py). Compare the metrics and, more importantly, whether the policy RANKING survives. The big load and
variance sweeps in compare_all.py use the simulator, so this is the check that makes them honest.

What to expect: throughput and TTFT p50 within ~10-20%. p99 and max ITL are noisier (they are single
worst events, and the real engine has timer jitter and kernel-shape quirks the 4-parameter model smooths over).

Outputs: results/sim_validation.json, results/sim_validation.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, POLICY_COLOR, LABEL, legend, set_style, style_axes  # noqa: E402
from batchlib.registry import ALL, make_policy  # noqa: E402
from batchlib.runner import get_engine, meta, rate_for_load, run_one, save  # noqa: E402
from batchlib.workload import WorkloadSpec  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"
METRICS = ["throughput_tps", "ttft_p50", "ttft_p99", "tpot_p50", "itl_p99"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--load", type=float, default=0.8)
    ap.add_argument("--out_mean", type=float, default=24)
    ap.add_argument("--out_cap", type=int, default=64)
    ap.add_argument("--long_frac", type=float, default=0.1)
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    spec = WorkloadSpec(n=args.n, prompt_min=16, prompt_max=96, long_frac=args.long_frac, long_len=320,
                        out_mean=args.out_mean, out_sigma=0.8, out_cap=args.out_cap)
    real, wk = get_engine("real", args.slots, spec.max_total(), args.device, args.weights)
    sim, _ = get_engine("sim", args.slots, spec.max_total())
    spec.rate, t_step = rate_for_load(args.load, real, spec)

    rows = {}
    print(f"{'policy':<28}{'metric':<16}{'real':>10}{'sim':>10}{'error':>8}")
    for n in ALL:
        r, _, _ = run_one(make_policy(n, args.slots), real, spec)
        s, _, _ = run_one(make_policy(n, args.slots), sim, spec)
        rows[n] = dict(real={k: r[k] for k in METRICS}, sim={k: s[k] for k in METRICS})
        for k in METRICS:
            err = (s[k] - r[k]) / r[k] * 100
            print(f"{LABEL[n]:<28}{k:<16}{r[k]:>10.3f}{s[k]:>10.3f}{err:>7.0f}%")
        print()

    # ranking agreement: does the sim order the policies like the real engine does, per metric?
    print("Rank agreement (Spearman) between real and sim across the 6 policies:")
    summary = {}
    for k in METRICS:
        a = np.array([rows[n]["real"][k] for n in ALL])
        b = np.array([rows[n]["sim"][k] for n in ALL])
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        rho = float(np.corrcoef(ra, rb)[0, 1])
        med = float(np.median(np.abs(b - a) / a) * 100)
        summary[k] = dict(spearman=rho, median_abs_err_pct=med)
        print(f"  {k:<16} rho {rho:5.2f}   median |error| {med:4.0f}%")

    save(OUT / "sim_validation.json", dict(meta=meta(args.device, wk), spec=spec.__dict__, rows=rows, summary=summary))

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, k, ttl, unit in zip(axes, ["throughput_tps", "ttft_p50", "itl_p99"],
                                ["Throughput (tok/s)", "TTFT p50 (s)", "ITL p99 (s)"], [1, 1, 1]):
        xs = [rows[n]["real"][k] for n in ALL]
        ys = [rows[n]["sim"][k] for n in ALL]
        ax.scatter(xs, ys, s=70, color=[POLICY_COLOR[n] for n in ALL], edgecolor=C["surface"], linewidth=1.5, zorder=3)
        m = max(max(xs), max(ys)) * 1.1
        ax.plot([0, m], [0, m], color=C["ink2"], linestyle=(0, (4, 3)), linewidth=1.5)
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        style_axes(ax, "Real engine", "Simulator", ttl)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=POLICY_COLOR[n], markersize=8, label=LABEL[n]) for n in ALL]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, labelcolor=C["ink2"], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "sim_validation.png", dpi=160)


if __name__ == "__main__":
    main()
