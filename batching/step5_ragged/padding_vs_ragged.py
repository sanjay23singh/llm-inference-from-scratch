"""Step 5: what does PADDING cost, and what does a RAGGED batch save?

A (B, T) tensor needs every row to have T tokens. Real requests don't, so classic batching pads short ones up
to the longest (pad tokens are computed, then masked out). The ragged batch in slot_engine.py concatenates the
real tokens only: (N,) with N = sum of lengths.

The cost is easy to predict, so predict it first:

  PREFILL (compute-bound): time ~ number of tokens pushed through the weights.
      padded N = B x T_max        ragged N = sum(T_i)        expected speedup = B x T_max / sum(T_i)
  DECODE  (memory-bound):  weights are streamed once either way, so padding the NEW token costs nothing, but
      padding the CACHE does: attention would read T_max positions of K/V for every row instead of its own S_i.
      The bytes read by attention grow from sum(S_i) to B x S_max.

We measure both by feeding the engine the same real step, once with true lengths and once with every row
padded to the longest. (Padded cost = the pad tokens go through exactly the same kernels, so feeding dummy
tokens is a faithful stand-in for a masked, padded tensor.)

Outputs: results/padding.json, results/padding.png
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, set_style, style_axes, legend  # noqa: E402
from batchlib.request import Work  # noqa: E402
from batchlib.runner import get_engine, meta, save  # noqa: E402
from batchlib.sim_engine import _Dummy  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def t_of(eng, ws, reps=5):
    eng.execute(ws)
    return statistics.median(eng.execute(ws)[1] for _ in range(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tmax", type=int, default=160)
    ap.add_argument("--spreads", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75],
                    help="shortest row = (1-spread) x longest; lengths evenly spaced in between")
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()
    B, T = args.batch, args.tmax

    eng, wk = get_engine("real", B, T + 16, args.device, args.weights)
    rows = []
    print(f"{'spread':>7}{'useful':>8} | {'PREFILL padded':>15}{'ragged':>9}{'speedup':>9}{'predicted':>10} | "
          f"{'DECODE padded':>14}{'ragged':>9}{'speedup':>9}{'KV read ratio':>14}")
    for sp in args.spreads:
        lens = [round(T * (1 - sp * i / (B - 1))) for i in range(B)]       # evenly spaced from T down to (1-sp)T
        useful = sum(lens) / (B * T)

        # PREFILL: whole prompts from position 0
        pad = [Work(_Dummy(i), 0, [1] * T, True) for i in range(B)]
        rag = [Work(_Dummy(i), 0, [1] * n, True) for i, n in enumerate(lens)]
        tp, tr = t_of(eng, pad), t_of(eng, rag)

        # DECODE: one new token per row, each row already holds lens[i]-1 cached tokens
        dpad = [Work(_Dummy(i), T - 1, [1], True) for i in range(B)]
        drag = [Work(_Dummy(i), n - 1, [1], True) for i, n in enumerate(lens)]
        dp, dr = t_of(eng, dpad), t_of(eng, drag)

        rows.append(dict(spread=sp, useful=useful, prefill_padded_ms=tp * 1e3, prefill_ragged_ms=tr * 1e3,
                         prefill_speedup=tp / tr, prefill_predicted=1 / useful,
                         decode_padded_ms=dp * 1e3, decode_ragged_ms=dr * 1e3, decode_speedup=dp / dr,
                         kv_read_ratio=B * T / sum(lens)))
        print(f"{sp:>7.2f}{useful:>8.2f} | {tp*1e3:>13.0f}ms{tr*1e3:>7.0f}ms{tp/tr:>8.2f}x{1/useful:>9.2f}x | "
              f"{dp*1e3:>12.0f}ms{dr*1e3:>7.0f}ms{dp/dr:>8.2f}x{B*T/sum(lens):>13.2f}x")

    save(OUT / "padding.json", dict(meta=meta(args.device, wk), batch=B, tmax=T, rows=rows))

    set_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = [r["useful"] * 100 for r in rows]
    kw = dict(linewidth=2, marker="o", markersize=7, markeredgecolor=C["surface"], markeredgewidth=1.5)
    a.plot(x, [r["prefill_speedup"] for r in rows], color="#2a78d6", label="Measured", **kw)
    a.plot(x, [r["prefill_predicted"] for r in rows], color=C["ink2"], linestyle=(0, (4, 3)), linewidth=2, label="Predicted: B·T_max / ΣT")
    a.invert_xaxis()
    a.set_ylim(bottom=0.8)
    style_axes(a, "Real tokens as % of the padded tensor", "Ragged speedup over padded (x)", "Prefill: compute-bound, pay per token")
    legend(a, "upper left")
    b.plot(x, [r["decode_speedup"] for r in rows], color="#eb6834", label="Decode step, measured", **kw)
    b.plot(x, [r["kv_read_ratio"] for r in rows], color=C["ink2"], linestyle=(0, (4, 3)), linewidth=2, label="K/V bytes read ratio")
    b.invert_xaxis()
    b.set_ylim(bottom=0.8)
    style_axes(b, "Real tokens as % of the padded tensor", "Ragged speedup over padded (x)", "Decode: only the K/V reads shrink")
    legend(b, "upper left")
    fig.tight_layout()
    fig.savefig(OUT / "padding.png", dpi=160)
    print(f"saved {OUT}/padding.json and .png")


if __name__ == "__main__":
    main()
