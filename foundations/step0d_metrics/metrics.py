"""Step 0d: serving metrics. TTFT, TPOT, throughput, goodput.

Part 1: per-request metrics from timestamps.
Part 2: goodput by batch size from YOUR batching.json under an SLO.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

BATCHING = Path(__file__).parents[2] / "kv_cache" / "step5c_batching" / "results" / "batching.json"


@dataclass
class Request:
    arrival: float                                   # seconds
    token_times: list = field(default_factory=list)  # time each output token was produced

    def ttft(self):
        return self.token_times[0] - self.arrival

    def tpot(self):                                  # average gap between tokens after the first
        n = len(self.token_times)
        return (self.token_times[-1] - self.token_times[0]) / (n - 1)

    def meets(self, slo_ttft, slo_tpot):
        return self.ttft() <= slo_ttft and self.tpot() <= slo_tpot


def throughput(reqs, duration):                      # all tokens / second
    return sum(len(r.token_times) for r in reqs) / duration


def goodput(reqs, duration, slo_ttft, slo_tpot):     # tokens from requests that met the SLO / second
    return sum(len(r.token_times) for r in reqs if r.meets(slo_ttft, slo_tpot)) / duration


def goodput_by_batch(slo_tpot_ms):
    print(f"SLO: TPOT <= {slo_tpot_ms} ms")
    print(f"{'batch':>5s} {'TPOT ms':>8s} {'throughput':>11s} {'goodput':>8s}")
    for r in json.loads(BATCHING.read_text()):
        ok = r["step_ms"] <= slo_tpot_ms
        print(f"{r['batch']:5d} {r['step_ms']:8.1f} {r['total_tps']:11.0f} {r['total_tps'] if ok else 0:8.0f}")


if __name__ == "__main__":
    # one request: arrives at t=0, prefill 74 ms, then a token every 20 ms, 100 tokens
    r = Request(arrival=0.0, token_times=[0.074 + 0.020 * i for i in range(100)])
    print(f"TTFT {r.ttft() * 1000:.0f} ms | TPOT {r.tpot() * 1000:.0f} ms | total {r.token_times[-1] * 1000:.0f} ms\n")
    goodput_by_batch(slo_tpot_ms=45)
