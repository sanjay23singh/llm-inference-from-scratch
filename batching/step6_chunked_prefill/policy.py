"""Step 6: CHUNKED PREFILL with a per-step token budget (Sarathi-Serve, vLLM v1 / SGLang default).

The problem left by step 4
--------------------------
A prompt of P tokens costs a step proportional to P (compute-bound). While that step runs, every decoding request
produces nothing. A 512-token prompt arriving into a healthy decode stream makes every user see one long gap:
TPOT looks fine (the mean barely moves), ITL p99 explodes.

The idea
--------
Cap every step at `token_budget` tokens and build it in a fixed priority order:

    1. every running decode request gets its 1 token first      (decodes are NEVER skipped = 'stall-free')
    2. the remaining budget goes to prefill, as a CHUNK of one or more waiting prompts
    3. a prompt too big for the leftover budget is simply continued next step (its K/V so far stays in its slot)

Step time is now bounded (~ time for `token_budget` tokens), so ITL has a ceiling no matter how long prompts get.

Why the budget is a trade-off, and where to set it
--------------------------------------------------
  small budget -> short steps, tight ITL, but a long prompt needs many steps -> higher TTFT for that request,
                  and each chunk re-streams the weights (memory-bound regime) -> lower prefill efficiency.
  large budget -> approaches mixed (step 4): efficient prefill, big stalls.
  Sweet spot: just above the knee of the step-time curve, where the step is still weight-streaming bound,
  so riding prefill tokens along with decodes is nearly free. The knee comes from sim_engine.fit()
  (compare with the ridge point in foundations/HARDWARE.md).

Correctness needs nothing new: attention for chunk k attends to the K/V written by chunks 0..k-1
(the `start` offset in the causal mask, see slot_engine._causal_mask). tests/ checks the tokens are bit-for-bit
those of unchunked generation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.request import make_work  # noqa: E402
from batchlib.scheduler import Policy  # noqa: E402


class ChunkedPrefill(Policy):
    name = "chunked"

    def __init__(self, token_budget=128, max_running=None):
        self.budget = token_budget
        self.max_running = max_running

    def schedule(self, s):
        works = [make_work(r, 1) for r in s.running if not r.in_prefill]       # 1. all decodes, always
        budget = max(0, self.budget - len(works))

        # 2. continue prompts already admitted (oldest first), then admit new ones while budget + slots remain
        for r in [r for r in s.running if r.in_prefill]:
            if budget == 0:
                break
            n = min(r.pending, budget)
            works.append(make_work(r, n))
            budget -= n
        cap = self.max_running or len(s.free_slots) + len(s.running)
        while budget > 0 and s.waiting and s.free_slots and len(s.running) < cap:
            r = s.waiting[0]
            s.admit(r)
            n = min(r.pending, budget)
            works.append(make_work(r, n))
            budget -= n
        return works or None
