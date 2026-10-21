"""Step 4 experiment: static vs dynamic vs continuous batching on the SAME traffic.

Workload: Poisson arrivals at 0.8x of the engine's decode-bound capacity (busy, not overloaded), output lengths
lognormal (sigma 0.8, i.e. a few long requests among many short ones).

Predictions before you run it:
  * throughput: continuous > static, because no decode row is wasted (slot efficiency -> 1.0).
  * TTFT p50/p99: continuous << static: a request is admitted at the very next step, not after the batch drains.
  * ITL p99: continuous (prefill-first) is WORSE than static. Every admitted prompt is a prefill-only step during
    which all decoders stall. Static has no mid-batch prefill at all. This is the cost step 6 removes.

Outputs: results/continuous.json, continuous_bars.png (3 headline metrics), continuous_occupancy.png
(rows computed per step: static's gap between computed and useful rows is the waste), continuous_ttft_cdf.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.experiments import compare_policies, plot_bars, plot_cdf, plot_occupancy  # noqa: E402
from batchlib.runner import get_engine, meta, rate_for_load, save  # noqa: E402
from batchlib.workload import WorkloadSpec, describe, make_requests  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "sim"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--load", type=float, default=0.8)
    ap.add_argument("--out_mean", type=float, default=32)
    ap.add_argument("--out_sigma", type=float, default=0.8)
    ap.add_argument("--out_cap", type=int, default=96)
    ap.add_argument("--max_wait", type=float, default=0.5)
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    spec = WorkloadSpec(n=args.n, prompt_min=16, prompt_max=96, out_mean=args.out_mean,
                        out_sigma=args.out_sigma, out_cap=args.out_cap)
    eng, wk = get_engine(args.engine, args.slots, spec.max_total(), args.device, args.weights)
    spec.rate, t_step = rate_for_load(args.load, eng, spec)
    print(f"decode step @ {args.slots} seqs = {t_step*1e3:.0f} ms | load {args.load} -> {spec.rate:.2f} req/s | "
          f"{describe(make_requests(spec))}\n")

    names = ["static", "dynamic", "continuous", "continuous_mixed"]
    res = compare_policies(eng, spec, names, args.slots, args.max_wait)

    OUT.mkdir(exist_ok=True)
    save(OUT / "continuous.json", dict(meta=meta(args.device, wk, dict(engine=args.engine)), spec=spec.__dict__,
                                       workload=describe(make_requests(spec)), decode_step_s=t_step,
                                       results={n: r[0] for n, r in res.items()}))
    plot_bars(res, OUT / "continuous_bars.png")
    plot_occupancy(res, ["static", "continuous"], OUT / "continuous_occupancy.png")
    plot_cdf(res, "ttft", OUT / "continuous_ttft_cdf.png", "Time to first token (s)", "TTFT distribution")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
