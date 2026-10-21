"""Step 3 experiment: what does the dynamic-batching timeout buy, and what can it NOT fix?

Two loads, same workload otherwise:
  light  (load 0.25 of decode capacity): a full batch rarely exists. Static waits for the B-th arrival, so the
         first requests of each batch sit in the queue. Dynamic caps that wait at `max_wait`.
  heavy  (load 1.5, saturated): a full batch is always waiting, the timeout never fires.

Prediction to make before you run it:
  light  -> dynamic cuts TTFT a lot, but launches near-empty batches (avg useful decode batch ~1-2): you got
            latency by giving up batching. Small max_wait = small batches; large max_wait = long waits.
  heavy  -> dynamic == static in every column. The timeout fixes WHEN a batch starts, not WHAT it does after:
            zombie rows (slot efficiency) and head-of-line blocking are untouched.
The `bucketed` row groups requests by (oracle) output length: it shrinks zombie rows without any iteration-level
scheduling, showing how much of continuous batching's win is "just" length awareness.

Outputs: results/dynamic_wait.json, results/dynamic_wait.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, POLICY_COLOR, legend, set_style, style_axes  # noqa: E402
from batchlib.registry import make_policy  # noqa: E402
from batchlib.runner import get_engine, meta, rate_for_load, run_one, save  # noqa: E402
from batchlib.workload import WorkloadSpec  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "sim"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.25, 1.5])
    ap.add_argument("--out_mean", type=float, default=24)
    ap.add_argument("--out_cap", type=int, default=64)
    ap.add_argument("--waits", type=float, nargs="+", default=[0.1, 0.3, 0.6, 1.2])
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    spec = WorkloadSpec(n=args.n, prompt_min=16, prompt_max=96, out_mean=args.out_mean, out_sigma=0.8,
                        out_cap=args.out_cap)
    eng, wk = get_engine(args.engine, args.slots, spec.max_total(), args.device, args.weights)

    mid = args.waits[len(args.waits) // 2]
    runs = [("static", dict(), "static")]
    runs += [(f"dynamic w={w:g}s", dict(max_wait=w), "dynamic") for w in args.waits]
    runs.append((f"bucketed w={mid:g}s", dict(max_wait=mid), "bucketed"))

    out = {}
    for load in args.loads:
        spec.rate, t_step = rate_for_load(load, eng, spec)
        print(f"\n== load {load}  (decode step @ {args.slots} seqs = {t_step*1e3:.0f} ms -> {spec.rate:.2f} req/s)")
        print(f"{'policy':<20}{'TTFT p50':>10}{'TTFT p99':>10}{'queue p50':>11}{'tok/s':>8}{'slot eff':>10}{'avg batch':>11}")
        rows = []
        for label, kw, name in runs:
            res, _, _ = run_one(make_policy(name, args.slots, **kw), eng, spec)
            res["label"], res["max_wait"], res["load"] = label, kw.get("max_wait"), load
            rows.append(res)
            print(f"{label:<20}{res['ttft_p50']:>10.2f}{res['ttft_p99']:>10.2f}{res['queue_p50']:>11.2f}"
                  f"{res['throughput_tps']:>8.1f}{res['slot_eff']:>10.2f}{res['avg_decode_batch']:>11.1f}")
        out[str(load)] = rows

    save(OUT / "dynamic_wait.json", dict(meta=meta(args.device, wk, dict(engine=args.engine)), spec=spec.__dict__, loads=out))

    set_style()
    fig, axes = plt.subplots(1, len(args.loads), figsize=(5.6 * len(args.loads), 4.4), squeeze=False)
    for ax, load in zip(axes[0], args.loads):
        rows = out[str(load)]
        dyn = [r for r in rows if r["label"].startswith("dynamic")]
        st = rows[0]
        kw = dict(linewidth=2, marker="o", markersize=7, markeredgecolor=C["surface"], markeredgewidth=1.5)
        ax.plot([r["max_wait"] for r in dyn], [r["ttft_p50"] for r in dyn], color=POLICY_COLOR["dynamic"], label="Dynamic p50", **kw)
        ax.plot([r["max_wait"] for r in dyn], [r["ttft_p99"] for r in dyn], color=POLICY_COLOR["dynamic"],
                linestyle=(0, (4, 3)), label="Dynamic p99", **kw)
        ax.axhline(st["ttft_p50"], color=POLICY_COLOR["static"], linewidth=2, label="Static p50")
        ax.axhline(st["ttft_p99"], color=POLICY_COLOR["static"], linewidth=2, linestyle=(0, (4, 3)), label="Static p99")
        ax.set_ylim(bottom=0)
        style_axes(ax, "max_wait (s)", "Time to first token (s)",
                   f"{'Light' if load < 1 else 'Saturated'} load ({load:g}x decode capacity)")
        legend(ax, "upper left")
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_wait.png", dpi=160)
    print(f"\nsaved {OUT}/dynamic_wait.json and .png")


if __name__ == "__main__":
    main()
