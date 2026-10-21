"""Step 1 benchmark: what does the slot engine cost / save compared with Step 3's torch.cat cache?

Part 2 left this open: "cache allocation (torch.cat copies)". Two measurements:

  A. The append alone. Step 3 builds the new cache with torch.cat every step: it allocates a (B,H,S+1,d) tensor
     and copies S+1 rows per sequence per layer for K and V. The slot engine writes ONE row in place.
     Bytes copied per step by cat  = 2 (K,V) x layers x B x (S+1) x hidden x 4 B   -> grows with batch x context.
  B. The whole decode step, end to end, same B and context:
       Step 3 model : batched attention (one bmm for all sequences) but cat-copies the cache every step
       slot engine  : in-place writes but a python loop over sequences for attention
     Which wins is NOT obvious. This script reports it either way. (A real engine gets both advantages:
     in-place cache AND a fused attention kernel that loops inside the GPU kernel. That is Module 3-4.)

Outputs: results/engine_bench.json and results/engine_bench.png
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.plotting import C, bars, end_label, legend, set_style, style_axes  # noqa: E402
from batchlib.runner import load_model, meta, save  # noqa: E402
from batchlib.request import Work  # noqa: E402
from batchlib.slot_engine import RealEngine, sync  # noqa: E402
from batchlib.sim_engine import _Dummy  # noqa: E402
from kv_model import CachedGPT2  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "results"


def timeit(fn, dev, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctxs", type=int, nargs="+", default=[32, 64, 128, 192, 248])
    ap.add_argument("--weights", default="auto")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev, B = args.device, args.batch

    model, kind = load_model(dev, args.weights)
    cfg = model.cfg
    cached = CachedGPT2(cfg).to(dev).eval()
    cached.load_state_dict(model.state_dict())
    eng = RealEngine(model, B, max(args.ctxs) + 8, dev)

    rows = []
    for S in args.ctxs:
        # ---- A. the append alone: cat vs in-place write, 12 layers x (K and V) -----------------------
        k_old = [torch.zeros(B, cfg.n_head, S, cfg.n_embd // cfg.n_head, device=dev) for _ in range(cfg.n_layer * 2)]
        k_new = torch.zeros(B, cfg.n_head, 1, cfg.n_embd // cfg.n_head, device=dev)
        slab = [torch.zeros(B, cfg.n_head, S + 8, cfg.n_embd // cfg.n_head, device=dev) for _ in range(cfg.n_layer * 2)]

        def do_cat():
            return [torch.cat([t, k_new], dim=2) for t in k_old]

        def do_write():
            for t in slab:
                t[:, :, S:S + 1] = k_new

        t_cat, t_wr = timeit(do_cat, dev), timeit(do_write, dev)
        copied = 2 * cfg.n_layer * B * (S + 1) * cfg.n_embd * 4

        # ---- B. whole decode step ---------------------------------------------------------------------
        ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
        with torch.no_grad():
            _, cache = cached(ids)
        nxt = torch.randint(0, cfg.vocab_size, (B, 1), device=dev)
        t_step3 = timeit(lambda: cached(nxt, cache), dev)
        ws = [Work(_Dummy(i), S, [1], True) for i in range(B)]
        t_slot = timeit(lambda: eng.execute(ws), dev)

        rows.append(dict(ctx=S, batch=B, cat_ms=t_cat * 1e3, inplace_ms=t_wr * 1e3, cat_copied_mb=copied / 1e6,
                         step3_step_ms=t_step3 * 1e3, slot_step_ms=t_slot * 1e3))
        print(f"ctx {S:>3}: append cat {t_cat*1e3:6.2f} ms ({copied/1e6:6.0f} MB copied) vs in-place {t_wr*1e3:5.2f} ms"
              f" | decode step: Step 3 {t_step3*1e3:6.1f} ms vs slot engine {t_slot*1e3:6.1f} ms")

    weights_mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    print(f"\n(model weights: {weights_mb:.0f} MB. A step must stream these once; cat adds the copy column above on top.)")
    kv = eng.kv
    print(f"slot cache reserved: {kv.reserved_bytes()/1e6:.0f} MB for {B} slots x {kv.max_len} tokens "
          f"({kv.bytes_per_token/1e3:.1f} KB/token).  A request using 100 tokens wastes {(1-100/kv.max_len)*100:.0f}% of its slot"
          f" -> internal fragmentation, Module 3.")

    save(OUT / "engine_bench.json", dict(meta=meta(dev, kind), rows=rows, weights_mb=weights_mb,
                                        reserved_mb=kv.reserved_bytes() / 1e6))

    set_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = [r["ctx"] for r in rows]
    for ax, ks, title, yl in ((a, ("cat_ms", "inplace_ms"), "Appending one token to the cache", "ms per step (24 tensors)"),
                              (b, ("step3_step_ms", "slot_step_ms"), "Whole decode step", "ms per step")):
        for key, col, lab in zip(ks, ("#2a78d6", "#eb6834"),
                                 ("torch.cat (Step 3)", "in-place write (slot cache)") if ks[0] == "cat_ms"
                                 else ("Step 3 model (batched attention, cat cache)", "Slot engine (in-place, per-sequence attention)")):
            ax.plot(x, [r[key] for r in rows], color=col, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=C["surface"], markeredgewidth=1.5, label=lab)
        style_axes(ax, "Context length (tokens already cached)", yl, title + f"  (batch {B})")
        ax.set_ylim(bottom=0)
        legend(ax, "upper left")
    fig.tight_layout()
    fig.savefig(OUT / "engine_bench.png", dpi=160)
    print(f"saved {OUT}/engine_bench.png")


if __name__ == "__main__":
    main()
