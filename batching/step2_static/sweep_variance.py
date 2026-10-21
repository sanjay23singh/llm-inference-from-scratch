"""Step 2 experiment: how badly does STATIC batching waste the batch as output lengths spread out?

Setup: the same arrivals every run, saturated (arrival rate above capacity, so a full batch is always waiting).
Only `out_sigma` changes: 0 = every request generates the same number of tokens (static's best case),
larger = a long tail of long requests.

Prediction before you run it (write it down!):
    slot efficiency  =  E[output length] / E[max output length in a batch of B]
That ratio is computed below from the workload alone ("expected") and compared with what the engine actually did
("measured": useful decode rows / decode rows computed). They should agree: the loss is purely the schedule.

Outputs: results/static_variance.json, results/static_variance.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, POLICY_COLOR, bars, set_style, style_axes  # noqa: E402
from batchlib.registry import make_policy  # noqa: E402
from batchlib.runner import get_engine, meta, run_one, save  # noqa: E402
from batchlib.workload import WorkloadSpec, describe, make_requests  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def expected_slot_eff(reqs, B):
    """Computed, not measured: batches of B consecutive requests; efficiency = sum(len) / (B * max(len)) per batch."""
    lens = [r.max_new for r in reqs]
    useful = cap = 0
    for i in range(0, len(lens), B):
        b = lens[i:i + B]
        useful += sum(b) - len(b)                     # the first token comes from prefill, not a decode row
        cap += len(b) * (max(b) - 1)
    return useful / cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "sim"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out_mean", type=float, default=32)
    ap.add_argument("--out_cap", type=int, default=96)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.2])
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    base = WorkloadSpec(n=args.n, rate=1e3, prompt_min=16, prompt_max=96, out_mean=args.out_mean,
                        out_cap=args.out_cap, out_min=4)
    eng, wk = get_engine(args.engine, args.slots, base.max_total(), args.device, args.weights)

    rows = []
    print(f"{'sigma':>6}{'CV':>6}{'expected eff':>14}{'measured eff':>14}{'avg decode batch':>18}"
          f"{'tok/s':>8}{'TTFT p50 s':>12}{'TTFT p99 s':>12}")
    for sg in args.sigmas:
        spec = WorkloadSpec(**{**base.__dict__, "out_sigma": sg})
        exp = expected_slot_eff(make_requests(spec), args.slots)
        res, pr, _ = run_one(make_policy("static", args.slots), eng, spec)
        res.update(sigma=sg, cv=describe(make_requests(spec))["out_cv"], expected_eff=exp)
        rows.append(res)
        print(f"{sg:>6.1f}{res['cv']:>6.2f}{exp:>14.2f}{res['slot_eff']:>14.2f}{res['avg_decode_batch']:>18.1f}"
              f"{res['throughput_tps']:>8.1f}{res['ttft_p50']:>12.2f}{res['ttft_p99']:>12.2f}")

    save(OUT / "static_variance.json", dict(meta=meta(args.device, wk, dict(engine=args.engine)), spec=base.__dict__, rows=rows))

    set_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(rows))
    names = [f"σ={r['sigma']:g}\nCV {r['cv']:.2f}" for r in rows]
    bars(a, names, [r["slot_eff"] * 100 for r in rows], [POLICY_COLOR["static"]] * len(rows), "{:.0f}%")
    for xi, r in zip(x, rows):                       # computed prediction as a tick on each bar
        a.plot([xi - 0.32, xi + 0.32], [r["expected_eff"] * 100] * 2, color=C["ink"], linewidth=2)
    a.set_ylim(0, 110)
    style_axes(a, "Output-length spread", "Useful share of decode rows (%)",
               "Static batching: share of the batch doing useful work")
    a.annotate("black tick = computed from lengths alone", (0.02, 0.04), xycoords="axes fraction", fontsize=8, color=C["ink2"])
    bars(b, names, [r["throughput_tps"] for r in rows], [POLICY_COLOR["static"]] * len(rows), "{:.0f}")
    style_axes(b, "Output-length spread", "Output tokens / s", "Throughput of the same hardware")
    b.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "static_variance.png", dpi=160)
    print(f"saved {OUT}/static_variance.json and .png")


if __name__ == "__main__":
    main()
