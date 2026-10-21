"""Turn finished requests + the step log into the numbers a serving engineer quotes.

Definitions (Part 2 / foundations/step0d_metrics):
  TTFT   first token time - arrival.                     What the user feels as 'it started answering'.
  TPOT   (last token - first token) / (n - 1).           Average streaming speed. HIDES stalls: a request that
                                                         froze for 2 s once still has a decent mean.
  ITL    gap between two consecutive tokens of ONE request. p99 / max ITL is what exposes stalls
         (this is the metric that prefill interference ruins).
  throughput  output tokens / makespan.
  goodput     throughput counting only requests that met the SLO (TTFT and TPOT bounds).
  slot efficiency  useful decode tokens / decode slot-steps. Static batching burns slot-steps on finished rows.
"""
import numpy as np

from . import paths  # noqa: F401
from metrics import goodput  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(reqs, steps, slo_ttft=2.0, slo_tpot=0.2):
    ttft = np.array([r.ttft() for r in reqs])
    tpot = np.array([r.tpot() for r in reqs])
    wait = np.array([r.t_admit - r.arrival for r in reqs])
    itl = np.concatenate([np.diff(r.token_times) for r in reqs])
    t_first = min(r.arrival for r in reqs)
    t_end = max(r.t_done for r in reqs)
    span = t_end - t_first
    n_tok = sum(len(r.generated) for r in reqs)

    dec_steps = [s for s in steps if s["n_dec"] + s["n_zombie"] > 0]
    live = sum(s["n_dec"] for s in dec_steps)
    slots = sum(s["n_dec"] + s["n_zombie"] for s in dec_steps)
    busy = sum(s["dt"] for s in steps)

    return dict(
        n_req=len(reqs), tokens=n_tok, makespan_s=span,
        throughput_tps=n_tok / span,
        goodput_tps=goodput(reqs, span, slo_ttft, slo_tpot),
        ttft_p50=pct(ttft, 50), ttft_p90=pct(ttft, 90), ttft_p99=pct(ttft, 99),
        tpot_p50=pct(tpot, 50), tpot_p99=pct(tpot, 99),
        itl_p50=pct(itl, 50), itl_p99=pct(itl, 99), itl_max=float(itl.max()),
        queue_p50=pct(wait, 50), queue_p99=pct(wait, 99),
        slo_ok_frac=float(np.mean([r.meets(slo_ttft, slo_tpot) for r in reqs])),
        slot_eff=live / slots if slots else float("nan"),
        avg_decode_batch=live / len(dec_steps) if dec_steps else 0.0,
        n_steps=len(steps), busy_frac=busy / span,
        mean_step_ms=1e3 * busy / len(steps),
    )


def per_request(reqs):
    """Raw per-request lists (for CDF plots)."""
    return dict(ttft=[round(r.ttft(), 4) for r in reqs], tpot=[round(r.tpot(), 5) for r in reqs])
