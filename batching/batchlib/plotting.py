"""Chart style shared by all batching plots. Same tokens as kv_cache/step5_benchmarks/benchmark.py
(light surface, ink text, recessive grid), plus a FIXED colour per policy: colour follows the entity,
never its rank, so a policy keeps its colour in every chart."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")

# categorical slots 1..6 in the repo palette's fixed order
POLICY_COLOR = {
    "static": "#2a78d6",              # blue
    "dynamic": "#eb6834",             # orange
    "continuous": "#1baf7a",          # aqua   (prefill-first, vLLM-v0 style)
    "continuous_mixed": "#eda100",    # yellow (Orca: decode + whole prompts in one step)
    "chunked": "#e87ba4",             # magenta (Sarathi / vLLM-v1: token budget, stall-free)
    "bucketed": "#008300",            # green  (static + oracle length grouping)
}
LABEL = {
    "static": "Static", "dynamic": "Dynamic (time window)", "continuous": "Continuous, prefill-first",
    "continuous_mixed": "Continuous, mixed (Orca)", "chunked": "Chunked prefill", "bucketed": "Static + oracle length buckets",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "text.color": C["ink"], "font.family": "sans-serif",
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
    ax.legend(frameon=False, labelcolor=C["ink2"], fontsize=9, loc=loc)


def end_label(ax, x, y, text, color=None):
    ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", va="center",
                color=color or C["ink"], fontsize=9)


def line(ax, x, y, policy, label=None, **kw):
    ax.plot(x, y, color=POLICY_COLOR[policy], linewidth=2, marker="o", markersize=7,
            markeredgecolor=C["surface"], markeredgewidth=1.5, label=label or LABEL[policy], **kw)


def bars(ax, names, values, colors, fmt="{:.1f}"):
    """Thin horizontal-ish bars drawn vertically, 4px-ish rounded look approximated by plain bars + value labels."""
    x = np.arange(len(names))
    ax.bar(x, values, width=0.55, color=colors, edgecolor=C["surface"], linewidth=2)
    for xi, v in zip(x, values):
        ax.annotate(fmt.format(v), (xi, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=C["ink"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color=C["ink2"])
