"""Step 0b: roofline for GPT-2 small. You fill in the two intensity functions (the core idea).

arithmetic intensity I = FLOPs / bytes moved from main memory
attainable GFLOP/s   = min(peak_compute, peak_bandwidth * I)
ridge point          = peak_compute / peak_bandwidth   (left of it: memory-bound, right: compute-bound)

Run step0a first, then:  python foundations/step0b_roofline/roofline.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
PEAKS = HERE.parent / "step0a_peaks" / "results" / "peaks.json"
BENCH = HERE.parent.parent / "kv_cache" / "step5_benchmarks" / "results" / "benchmark.json"

N_PARAMS = 124_439_808   # GPT-2 small, tied lm_head, counted once
BYTES = 4                # fp32


def decode_intensity(batch: int) -> float:
    """FLOP/byte of ONE decode step for `batch` sequences. Ignore the KV cache and activations for now.

    Hints:
      - FLOPs: each weight does one multiply-add per token. How many FLOPs is that per token? times batch.
      - Bytes: every weight is read from memory once per step, no matter the batch. How many bytes?
    """
    return batch * 0.5


def prefill_intensity(prompt_len: int) -> float:
    """FLOP/byte of ONE prefill pass over `prompt_len` tokens (batch 1). Same hints as decode.

    Question to answer before coding: what plays the role of `batch` here, and why?
    """
    return 0.5 * prompt_len


def main():
    peaks = json.loads(PEAKS.read_text())
    cpu = next(c for c in peaks["ceilings"] if c["device"] == "cpu")
    pf, bw = cpu["gflops"], cpu["gbs"]
    ridge = pf / bw

    x = np.logspace(-1, 4, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, np.minimum(pf, bw * x), color="#0b0b0b", lw=2, label=f"roofline ({peaks['chip']}, CPU fp32)")
    ax.axvline(ridge, color="#898781", ls="--", lw=1)
    ax.text(ridge * 1.05, pf * 0.04, f"ridge {ridge:.0f} FLOP/B", color="#52514e", fontsize=8)

    for b in (1, 4, 16, 64):
        i = decode_intensity(b)
        ax.plot(i, min(pf, bw * i), "o", color="#2a78d6")
        ax.annotate(f"decode B={b}", (i, min(pf, bw * i)), textcoords="offset points", xytext=(5, -12), fontsize=8)
    for t in (16, 128, 512):
        i = prefill_intensity(t)
        ax.plot(i, min(pf, bw * i), "s", color="#eb6834")
        ax.annotate(f"prefill T={t}", (i, min(pf, bw * i)), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # your real measurements from kv_cache/step5_benchmarks (prefill_gflops / decode_gflops)
    if BENCH.exists():
        for r in json.loads(BENCH.read_text())["prefill_decode"]:
            ax.plot(prefill_intensity(r["context"]), r["prefill_gflops"], "s", mfc="none", color="#eb6834")
            ax.plot(decode_intensity(1), r["decode_gflops"], "o", mfc="none", color="#2a78d6")

    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Roofline: filled = predicted ceiling, hollow = your measurements")
    ax.grid(True, which="both", color="#e1e0d9", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "roofline.png", dpi=150, bbox_inches="tight")
    print("saved", out / "roofline.png", f"| ridge {ridge:.1f} FLOP/B")


if __name__ == "__main__":
    main()
