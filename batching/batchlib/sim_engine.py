"""A cost model of the engine, fitted from measurements, plus a drop-in engine that uses it.

Why have a simulator at all
---------------------------
Real runs are slow (every policy x every load x every seed = hours) and noisy (timer jitter moves p99).
The scheduling logic does not care HOW LONG a step takes, only that the clock advances by it. So:

  1. calibrate()  measure real steps of many shapes on THIS machine,
  2. fit()        fit  t_step = t0 + f(N) + c*A + d*n_seq
  3. SimEngine    returns that predicted time instead of computing anything.

Then thousands of requests per policy take milliseconds, and the policy code is exactly the same code path.
step7_compare checks the simulator against real runs, so we know how far to trust it.

The model is a roofline with a learned knee
-------------------------------------------
  N = tokens in the step (all sequences).          A = attention "pairs" = sum over sequences of n*(start + (n+1)/2)
  f(N)  a CONVEX piecewise-linear function of N, written as a sum of hinges  sum_j w_j * max(0, N - k_j), w_j >= 0.
        Convexity is the roofline: max(time to stream weights, time to do the math) is convex. Flat at small N
        (memory-bound: the weights are streamed once, extra tokens are free), then bending upward (compute-bound).
        The bend is the measured analogue of the ridge point in foundations/HARDWARE.md. Non-negative weights are
        solved with NNLS so noise cannot produce a step that gets cheaper with more tokens.
  c     attention cost per (query, key) pair: reading K/V, which batching can NOT share between sequences.
  d     per-sequence overhead (python loop over sequences, small kernels).
"""
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import paths  # noqa: F401
from .request import Work


KNOTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass
class CostModel:
    t0: float = 0.040                                   # s: step floor (weights streamed once + fixed overhead)
    knots: tuple = KNOTS
    w: tuple = (0.0,) * len(KNOTS)                      # s per token above each knot
    c: float = 2e-8                                     # s per attention pair
    d: float = 1e-4                                     # s per sequence
    r2: float = float("nan")
    max_rel_err: float = float("nan")

    def f(self, N):
        N = np.asarray(N, float)
        return sum(wj * np.maximum(0.0, N - kj) for wj, kj in zip(self.w, self.knots))

    def predict(self, N, A, n_seq):
        return self.t0 + self.f(N) + self.c * np.asarray(A, float) + self.d * np.asarray(n_seq, float)

    def knee(self, rise=0.10):
        """Smallest N where the token-dependent part of a step has added `rise` (10%) on top of the N=1 step."""
        base = self.t0 + self.f(1) + self.d
        for n in range(1, 4096):
            if self.t0 + self.f(n) + self.d > base * (1 + rise):
                return n
        return 4096

    def save(self, path):
        json.dump({**asdict(self), "knee_n": self.knee()}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        d.pop("knee_n", None)
        d["knots"], d["w"] = tuple(d["knots"]), tuple(d["w"])
        return cls(**d)


def features(works):
    N = sum(len(w.tokens) for w in works)
    A = sum(len(w.tokens) * (w.start + (len(w.tokens) + 1) / 2) for w in works)
    return N, A, len(works)


class SimEngine:
    """Same interface as RealEngine, but 'executes' by looking up the cost model. Tokens are dummies."""
    kind = "sim"

    def __init__(self, cost: CostModel, n_slots, max_len):
        self.cost, self.n_slots, self.max_len = cost, n_slots, max_len

    def execute(self, works):
        N, A, n = features(works)
        return [0 if w.sample else None for w in works], float(self.cost.predict(N, A, n))


# ----------------------------------------------------------------------------- calibration
class _Dummy:                      # just enough of Req for Work (the engine only reads .slot)
    def __init__(self, slot):
        self.slot = slot


def _shape(kind, n, ctx, chunk):
    """Build a synthetic step. 'decode': n sequences each at context ctx. 'prefill': one chunk of `chunk`
    tokens after ctx. 'mixed': n decodes plus a prefill chunk (what chunked prefill produces)."""
    ws = []
    if kind in ("decode", "mixed"):
        ws += [Work(_Dummy(i), ctx, [1], True) for i in range(n)]
    if kind in ("prefill", "mixed"):
        ws += [Work(_Dummy(len(ws)), ctx if kind == "prefill" else 0, [1] * chunk, True)]
    return ws


def calibrate(engine, reps=5, ctxs=(16, 64, 160), log=print):
    """Measure the real engine on many step shapes. Returns list of (N, A, n_seq, seconds)."""
    max_ctx = engine.max_len - 8
    pts = []
    shapes = []
    for ctx in ctxs:
        ctx = min(ctx, max_ctx)
        for n in (1, 2, 4, 8, 12, 16):
            if n <= engine.n_slots:
                shapes.append(("decode", n, ctx, 0))
    for chunk in (4, 16, 32, 64, 128, 192, 256, 384):
        if chunk <= max_ctx:
            shapes.append(("prefill", 1, 0, chunk))
    for ctx, chunk in ((128, 64), (256, 128), (384, 128)):          # later chunks of a long prompt
        if ctx + chunk <= max_ctx:
            shapes.append(("prefill", 1, ctx, chunk))
    for n, chunk in ((4, 32), (8, 64), (8, 128), (12, 96)):
        if n + 1 <= engine.n_slots and chunk + 32 <= max_ctx:
            shapes.append(("mixed", n, 32, chunk))
    for sh in shapes:
        ws = _shape(*sh)
        engine.execute(ws)                                       # warm-up
        ts = sorted(engine.execute(ws)[1] for _ in range(reps))
        N, A, ns = features(ws)
        pts.append((N, A, ns, ts[len(ts) // 2]))
        log(f"  {sh[0]:8s} n={sh[1]:>2} ctx={sh[2]:>3} chunk={sh[3]:>3}  N={N:>4}  {ts[len(ts) // 2] * 1e3:7.1f} ms")
    return pts


def _nnls(X, y, iters=20000):
    """min ||Xw - y||^2, w >= 0, by cyclic coordinate descent (tiny problem, no scipy dependency)."""
    w = np.zeros(X.shape[1])
    r = y.copy()
    col2 = (X ** 2).sum(0) + 1e-12
    for _ in range(iters):
        moved = 0.0
        for j in range(X.shape[1]):
            new = max(0.0, w[j] + X[:, j] @ r / col2[j])
            if new != w[j]:
                r -= X[:, j] * (new - w[j])
                moved += abs(new - w[j])
                w[j] = new
        if moved < 1e-12:
            break
    return w


def fit(pts):
    """Fit t = t0 + sum_j w_j*max(0,N-k_j) + c*A + d*n_seq with all coefficients >= 0 (convex roofline)."""
    N, A, ns, t = (np.array(x, float) for x in zip(*pts))
    cols = [np.ones_like(N)] + [np.maximum(0, N - k) for k in KNOTS] + [A, ns]
    X = np.stack(cols, 1)
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1.0                                  # a knot beyond the largest measured N has no data
    coef = _nnls(X / scale, t) / scale
    pred = X @ coef
    r2 = 1 - ((pred - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return CostModel(float(coef[0]), KNOTS, tuple(float(v) for v in coef[1:1 + len(KNOTS)]),
                     float(coef[-2]), float(coef[-1]), float(r2), float(np.max(np.abs(pred - t) / t)))
