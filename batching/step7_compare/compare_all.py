"""Step 7c: every batching strategy, head to head.

A. HEADLINE (real engine by default): all six policies on one workload at 0.8x load.
B. LOAD SWEEP  (simulator): offered load 0.2x ... 1.5x of decode capacity. Where does each policy's latency blow up?
C. VARIANCE SWEEP (simulator): output-length spread sigma 0 ... 1.25 at saturation. Who loses what as lengths diverge?
D. SLOT SWEEP (simulator): max concurrency 4 ... 32 vs the KV memory it reserves. Where does the next topic start?

B-D use the simulator (calibrate.py first; validate_sim.py tells you how far to trust it) because they need ~100
runs x hundreds of requests. Every sweep point is the median of 3 seeds. Pass --sweep_engine real to use the real
engine instead (slow).

SLOs for goodput are RELATIVE to this machine: TPOT <= 3 x (decode step at full batch), TTFT <= 25 x that step.
They are printed, so a blog number always carries its SLO.

Outputs (results/): compare_headline.json, compare_load.json, compare_variance.json, compare_slots.json
                    + compare_bars.png, compare_occupancy.png, compare_ttft_cdf.png, compare_load.png,
                    compare_variance.png, compare_slots.png
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.experiments import compare_policies, plot_bars, plot_cdf, plot_occupancy  # noqa: E402
from batchlib.plotting import C, LABEL, POLICY_COLOR, end_label, legend, line, plt, set_style, style_axes  # noqa: E402
from batchlib.registry import ALL, make_policy  # noqa: E402
from batchlib.runner import decode_step_time, get_engine, meta, rate_for_load, run_one, save  # noqa: E402
from batchlib.sim_engine import SimEngine  # noqa: E402
from batchlib.workload import WorkloadSpec, describe, make_requests  # noqa: E402

OUT = Path(__file__).parent / "results"
SEEDS = (0, 1, 2)


def sweep(eng, slots, base: WorkloadSpec, xs, make_spec, names, metrics, slo, max_wait, budget):
    """For each x and policy: median over SEEDS of each metric."""
    out = {n: {m: [] for m in metrics} for n in names}
    for x in xs:
        for n in names:
            vals = []
            for sd in SEEDS:
                r, _, _ = run_one(make_policy(n, slots, max_wait, budget), eng, make_spec(x, sd), slo=slo)
                vals.append(r)
            for m in metrics:
                out[n][m].append(float(np.median([v[m] for v in vals])))
    return out


def lineplot(ax, xs, series, names, key, mul, xlabel, ylabel, title, logy=False):
    for n in names:
        line(ax, xs, np.array(series[n][key]) * mul, n)
    if logy:
        ax.set_yscale("log")
    style_axes(ax, xlabel, ylabel, title)
    if not logy:
        ax.set_ylim(bottom=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="real", choices=["real", "sim"], help="engine for the HEADLINE run")
    ap.add_argument("--sweep_engine", default="sim", choices=["real", "sim"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--n", type=int, default=64, help="requests in the headline run")
    ap.add_argument("--n_sweep", type=int, default=240, help="requests per sweep point")
    ap.add_argument("--out_mean", type=float, default=32)
    ap.add_argument("--out_cap", type=int, default=96)
    ap.add_argument("--long_frac", type=float, default=0.1)
    ap.add_argument("--long_len", type=int, default=320)
    ap.add_argument("--max_wait", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--only", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    base = WorkloadSpec(n=args.n, prompt_min=16, prompt_max=96, long_frac=args.long_frac, long_len=args.long_len,
                        out_mean=args.out_mean, out_sigma=0.8, out_cap=args.out_cap)
    max_len = base.max_total()
    real, wk = get_engine(args.engine, args.slots, max_len, args.device, args.weights)
    t_full = decode_step_time(real)
    slo = dict(slo_tpot=3 * t_full, slo_ttft=25 * t_full)
    print(f"decode step @ {args.slots} seqs: {t_full*1e3:.0f} ms | SLO: TPOT <= {slo['slo_tpot']*1e3:.0f} ms, "
          f"TTFT <= {slo['slo_ttft']:.1f} s | dynamic max_wait {args.max_wait}s, chunk budget {args.budget}")
    m = meta(args.device, wk, dict(headline_engine=args.engine, sweep_engine=args.sweep_engine, slo=slo,
                                   slots=args.slots, max_wait=args.max_wait, budget=args.budget))

    # ------------------------------------------------------------------ A. headline
    if "A" in args.only:
        spec = replace(base)
        spec.rate, _ = rate_for_load(0.8, real, spec)
        print(f"\n== A. headline, load 0.8 -> {spec.rate:.2f} req/s | {describe(make_requests(spec))}\n")
        res = compare_policies(real, spec, ALL, args.slots, args.max_wait, args.budget, slo=slo)
        save(OUT / "compare_headline.json", dict(meta=m, spec=spec.__dict__, workload=describe(make_requests(spec)),
                                                 results={n: r[0] for n, r in res.items()}))
        plot_bars(res, OUT / "compare_bars.png")
        plot_occupancy(res, ["static", "continuous", "chunked"], OUT / "compare_occupancy.png")
        plot_cdf(res, "ttft", OUT / "compare_ttft_cdf.png", "Time to first token (s)", "TTFT distribution")

    if not set(args.only) & {"B", "C", "D"}:
        return
    sweng, _ = get_engine(args.sweep_engine, args.slots, max_len, args.device, args.weights)
    metrics = ["throughput_tps", "goodput_tps", "ttft_p50", "ttft_p99", "itl_p99", "slot_eff"]
    sw = replace(base, n=args.n_sweep)

    # ------------------------------------------------------------------ B. load sweep
    if "B" in args.only:
        loads = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        print("\n== B. load sweep")

        def mk(load, sd):
            sp = replace(sw, seed=sd)
            sp.rate, _ = rate_for_load(load, sweng, sp)
            return sp
        res = sweep(sweng, args.slots, sw, loads, mk, ALL, metrics, slo, args.max_wait, args.budget)
        save(OUT / "compare_load.json", dict(meta=m, loads=loads, series=res))
        set_style()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
        for ax, (k, mul, yl, ttl, logy) in zip(axes.flat, [
                ("throughput_tps", 1, "Output tokens / s", "Throughput", False),
                ("goodput_tps", 1, "Output tokens / s within SLO", "Goodput", False),
                ("ttft_p99", 1, "seconds (log)", "TTFT p99", True),
                ("itl_p99", 1e3, "milliseconds", "Inter-token latency p99", False)]):
            lineplot(ax, loads, res, ALL, k, mul, "Offered load (x decode capacity)", yl, ttl, logy)
        h, lb = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lb, loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=C["ink2"])
        fig.tight_layout(rect=[0, 0.07, 1, 1])                       # legend below the grid, never on the data
        fig.savefig(OUT / "compare_load.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------ C. variance sweep
    if "C" in args.only:
        sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        print("== C. output-length variance sweep (saturated, load 1.5)")

        def mk(sg, sd):
            sp = replace(sw, seed=sd, out_sigma=sg)
            sp.rate, _ = rate_for_load(1.5, sweng, sp)
            return sp
        res = sweep(sweng, args.slots, sw, sigmas, mk, ALL, metrics, slo, args.max_wait, args.budget)
        cv = [describe(make_requests(mk(sg, 0)))["out_cv"] for sg in sigmas]
        save(OUT / "compare_variance.json", dict(meta=m, sigmas=sigmas, cv=cv, series=res))
        set_style()
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
        lineplot(a, cv, res, ALL, "throughput_tps", 1, "Output-length CV (std / mean)", "Output tokens / s",
                 "Throughput as lengths spread out")
        lineplot(b, cv, res, ALL, "ttft_p50", 1, "Output-length CV (std / mean)", "seconds", "TTFT p50")
        h, lb = a.get_legend_handles_labels()
        fig.legend(h, lb, loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=C["ink2"])
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        fig.savefig(OUT / "compare_variance.png", dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------------ D. slots vs reserved KV memory
    if "D" in args.only:
        slot_list = [2, 4, 8, 16, 32]
        print("== D. concurrency (KV slots) sweep, saturated")
        names = ["static", "continuous", "chunked"]
        res = {n: {k: [] for k in ("throughput_tps", "ttft_p99")} for n in names}
        reserved = []
        bpt = 2 * 12 * 768 * 4                                      # GPT-2 small fp32 K+V bytes per token
        for sl in slot_list:
            eng_s = SimEngine(sweng.cost, sl, max_len) if isinstance(sweng, SimEngine) else get_engine("real", sl, max_len, args.device, args.weights)[0]
            reserved.append(sl * max_len * bpt / 1e6)
            for n in names:
                vals = []
                for sd in SEEDS:
                    sp = replace(sw, seed=sd)
                    sp.rate, _ = rate_for_load(1.5, eng_s, sp)
                    r, _, _ = run_one(make_policy(n, sl, args.max_wait, args.budget), eng_s, sp, slo=slo)
                    vals.append(r)
                for k in res[n]:
                    res[n][k].append(float(np.median([v[k] for v in vals])))
        wl = make_requests(sw)
        used = np.mean([len(r.prompt) + r.max_new for r in wl])
        frag = 1 - used / max_len
        save(OUT / "compare_slots.json", dict(meta=m, slots=slot_list, reserved_mb=reserved, series=res,
                                              mean_tokens_used=float(used), slot_len=max_len, internal_fragmentation=float(frag)))
        print(f"   each slot reserves {max_len} tokens but a request averages {used:.0f}: {frag*100:.0f}% of reserved KV is unused")
        set_style()
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for n in names:
            line(ax, reserved, res[n]["throughput_tps"], n)
            end_label(ax, reserved[-1], res[n]["throughput_tps"][-1], f"{res[n]['throughput_tps'][-1]:.0f}")
        style_axes(ax, "KV cache reserved (MB), = slots x slot length x bytes/token", "Output tokens / s",
                   "More slots = more throughput, paid in reserved memory")
        ax.set_xlim(right=reserved[-1] * 1.12)
        ax.set_ylim(bottom=0)
        legend(ax, "upper left")
        fig.tight_layout()
        fig.savefig(OUT / "compare_slots.png", dpi=160)
        plt.close(fig)

    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
