"""Synthetic traffic: Poisson arrivals, random prompt lengths, random output lengths.

Why a synthetic workload and not 'run 8 prompts': static batching only looks bad when requests differ.
Two things make real traffic hard on a batcher, and each has its own knob here:

  arrivals       Poisson with rate `rate` (requests/s). Gaps are exponential, so bursts and lulls happen.
  output length  lognormal with mean `out_mean` and spread `out_sigma`. sigma=0 -> every request is the same
                 length (static batching's best case). Larger sigma -> a few long requests among many short ones.
                 That variance is the whole reason continuous batching exists.

Output length is assigned up front (stands in for 'the model emits EOS here') so we can control its variance.
Arrivals, lengths and token ids use separate random streams: changing out_sigma does not move the arrivals,
so sweeps compare like with like.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import paths  # noqa: F401
from .request import Req


@dataclass
class WorkloadSpec:
    n: int = 64                  # number of requests
    rate: float = 2.0            # arrivals per second (Poisson)
    prompt_min: int = 16
    prompt_max: int = 128
    long_frac: float = 0.0       # fraction of prompts that are LONG (used by the prefill-stall experiment)
    long_len: int = 512
    out_mean: float = 48.0
    out_sigma: float = 0.8       # lognormal sigma; 0 = all equal
    out_min: int = 4
    out_cap: int = 128
    vocab: int = 50257
    seed: int = 0

    def max_total(self):
        return max(self.prompt_max, self.long_len if self.long_frac > 0 else 0) + self.out_cap


def make_requests(spec: WorkloadSpec):
    rng_arr = np.random.default_rng(spec.seed)
    rng_len = np.random.default_rng(spec.seed + 1)
    rng_tok = np.random.default_rng(spec.seed + 2)

    arrivals = np.cumsum(rng_arr.exponential(1.0 / spec.rate, spec.n))
    arrivals -= arrivals[0]                                  # first request arrives at t=0

    plen = rng_len.integers(spec.prompt_min, spec.prompt_max + 1, spec.n)
    if spec.long_frac > 0:
        plen = np.where(rng_len.random(spec.n) < spec.long_frac, spec.long_len, plen)

    if spec.out_sigma == 0:
        olen = np.full(spec.n, int(spec.out_mean))
    else:
        mu = math.log(spec.out_mean) - spec.out_sigma ** 2 / 2          # so that E[length] = out_mean
        olen = np.clip(np.rint(rng_len.lognormal(mu, spec.out_sigma, spec.n)), spec.out_min, spec.out_cap)

    return [Req(arrival=float(arrivals[i]), rid=i, max_new=int(olen[i]),
                prompt=rng_tok.integers(0, spec.vocab, int(plen[i])).tolist())
            for i in range(spec.n)]


def describe(reqs):
    o = np.array([r.max_new for r in reqs], float)
    p = np.array([len(r.prompt) for r in reqs], float)
    return dict(n=len(reqs), span_s=reqs[-1].arrival, out_mean=o.mean(), out_cv=o.std() / o.mean(),
                out_max=int(o.max()), prompt_mean=p.mean(), prompt_max=int(p.max()))
