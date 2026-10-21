"""Step 7a: measure step times on THIS machine and fit the cost model that SimEngine uses.

    t_step = t0 + f(N) + c * A + d * n_seq          f convex piecewise linear (hinges, NNLS)

Run once per machine (about a minute). Prints the fit and the knee (tokens per step at which a step is 10% slower
than a 1-token step), then compares it with the ridge point you measured in foundations/HARDWARE.md:

    a decode-style step does ~2 FLOPs per weight per token and streams each fp32 weight once (4 B),
    so arithmetic intensity ~ N/2 FLOP/byte  ->  predicted knee N0 ~ 2 x ridge point.

If the fitted knee is far from that, something else (python overhead, kernel choice, thread count) dominates:
that is a finding worth a sentence in the post, not a bug in the fit.

Outputs: results/cost_model.json, results/cost_model.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, set_style, style_axes, legend  # noqa: E402
from batchlib.runner import get_engine, meta, save  # noqa: E402
from batchlib.sim_engine import calibrate, fit  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=544)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=19.8, help="FLOP/byte ridge for this device from HARDWARE.md (M1 CPU fp32: 19.8)")
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    eng, wk = get_engine("real", args.slots, args.max_len, args.device, args.weights)
    pts = calibrate(eng, reps=args.reps, ctxs=(16, 64, 160, 320))
    cm = fit(pts)
    cm.save(OUT / "cost_model.json")
    save(OUT / "cost_points.json", dict(meta=meta(args.device, wk), points=pts))

    print(f"\nfit: t0 {cm.t0*1e3:.1f} ms | c {cm.c*1e9:.1f} ns/attention pair | d {cm.d*1e6:.0f} us/sequence")
    print("     f(N) hinges (ms per token above knot): " + ", ".join(f"{k}:{w*1e3:.3f}" for k, w in zip(cm.knots, cm.w) if w > 0))
    print(f"     marginal cost per extra token: N=1 {cm.f(2)*1e3-cm.f(1)*1e3:.3f} ms, N=64 {(cm.f(65)-cm.f(64))*1e3:.3f} ms, "
          f"N=256 {(cm.f(257)-cm.f(256))*1e3:.3f} ms")
    print(f"R^2 {cm.r2:.3f}, worst relative error {cm.max_rel_err*100:.0f}%")
    print(f"knee (step 10% slower than a 1-token step): {cm.knee()} tokens | HARDWARE.md ridge {args.ridge} FLOP/byte "
          f"-> predicted ~{2*args.ridge:.0f} tokens for fp32 weights")

    N, A, ns, t = (np.array(x) for x in zip(*pts))
    pred = cm.predict(N, A, ns)
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(t * 1e3, pred * 1e3, s=36, color="#2a78d6", edgecolor=C["surface"], linewidth=1)
    m = max(t.max(), pred.max()) * 1e3
    ax.plot([0, m], [0, m], color=C["ink2"], linestyle=(0, (4, 3)), linewidth=1.5)
    style_axes(ax, "Measured step time (ms)", "Cost-model prediction (ms)", f"Cost model fit, R² {cm.r2:.3f}")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    fig.tight_layout()
    fig.savefig(OUT / "cost_model.png", dpi=160)


if __name__ == "__main__":
    main()
