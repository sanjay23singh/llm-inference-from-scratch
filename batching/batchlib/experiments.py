"""Shared experiment helpers: run several policies on ONE workload and draw the standard comparison charts."""
import numpy as np

from . import paths  # noqa: F401
from .plotting import C, LABEL, POLICY_COLOR, bars, end_label, legend, line, plt, set_style, style_axes
from .registry import make_policy
from .runner import run_one


def compare_policies(eng, spec, names, slots, max_wait=0.5, token_budget=128, slo=None, log=print):
    """Same requests (same seed), same engine, one policy at a time. Returns {name: (summary, per_request, steps)}."""
    out = {}
    log(f"{'policy':<28}{'tok/s':>7}{'good/s':>8}{'TTFT p50':>9}{'p99':>7}{'TPOT p50':>9}{'ITL p99':>8}{'ITL max':>8}"
        f"{'slot eff':>9}{'batch':>6}")
    for n in names:
        res, pr, steps = run_one(make_policy(n, slots, max_wait, token_budget), eng, spec, slo=slo, keep_steps=True)
        out[n] = (res, pr, steps)
        log(f"{LABEL[n]:<28}{res['throughput_tps']:>7.1f}{res['goodput_tps']:>8.1f}{res['ttft_p50']:>9.2f}{res['ttft_p99']:>7.2f}"
            f"{res['tpot_p50']*1e3:>8.0f}m{res['itl_p99']*1e3:>7.0f}m{res['itl_max']*1e3:>7.0f}m"
            f"{res['slot_eff']:>9.2f}{res['avg_decode_batch']:>6.1f}")
    return out


def plot_bars(results, path, title_suffix=""):
    """Throughput / TTFT p99 / ITL p99, one bar per policy, fixed policy colours."""
    set_style()
    names = list(results)
    cols = [POLICY_COLOR[n] for n in names]
    short = [LABEL[n].replace("Continuous, ", "Cont. ").replace(" (time window)", "").replace("Static + oracle length buckets", "Bucketed")
             .replace("(Orca)", "").replace("prefill-first", "pf") for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    spec = [("throughput_tps", 1, "Output tokens / s", "Throughput (higher is better)", "{:.0f}"),
            ("ttft_p99", 1, "seconds", "TTFT p99 (lower is better)", "{:.1f}"),
            ("itl_p99", 1e3, "milliseconds", "Inter-token latency p99 (lower is better)", "{:.0f}")]
    for ax, (k, mul, yl, ttl, fmt) in zip(axes, spec):
        bars(ax, short, [results[n][0][k] * mul for n in names], cols, fmt)
        ax.tick_params(axis="x", labelrotation=25)
        style_axes(ax, "", yl, ttl + title_suffix)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_occupancy(results, names, path):
    """Rows computed per step over time. For static, the gap between 'rows computed' and 'useful rows' IS the waste."""
    set_style()
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.2), sharey=True, squeeze=False)
    ymax = 0
    for ax, n in zip(axes[0], names):
        st = results[n][2]
        t = np.array([s["t"] for s in st])
        useful = np.array([s["n_dec"] for s in st])
        total = useful + np.array([s["n_zombie"] for s in st])
        ymax = max(ymax, total.max())
        if total.sum() != useful.sum():
            ax.fill_between(t, 0, total, step="post", color=POLICY_COLOR[n], alpha=0.25, linewidth=0, label="rows computed")
        ax.step(t, useful, where="post", color=POLICY_COLOR[n], linewidth=1.8, label="useful rows (live requests)")
        style_axes(ax, "Time (s)", "Decode rows in the step", LABEL[n])
        legend(ax, "upper right")
    for ax in axes[0]:
        ax.set_ylim(0, ymax * 1.4)                       # headroom so the legend never sits on the data
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cdf(results, key, path, xlabel, title, mul=1.0):
    set_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for n, (_, pr, _) in results.items():
        v = np.sort(np.array(pr[key]) * mul)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=POLICY_COLOR[n], linewidth=2, label=LABEL[n])
    style_axes(ax, xlabel, "Fraction of requests", title)
    ax.set_ylim(0, 1.02)
    legend(ax, "lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
