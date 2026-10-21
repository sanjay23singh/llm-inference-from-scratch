"""Step 4: CONTINUOUS batching (iteration-level scheduling, Orca, OSDI'22).

The fix for static batching, in one sentence: stop treating the batch as a unit.
Decide the batch again before EVERY forward pass:

    finished request  -> leaves immediately, slot freed
    waiting request   -> joins at the very next step, if a slot is free

Nothing waits for batch-mates, so there are no zombie rows, and a short request's latency depends on its own
length, not on whoever it was batched with.

Two things are needed to make that possible, and both are in batchlib/slot_engine.py:
  * a slot-based KV cache (a request can enter or leave without touching anyone else's cache), and
  * a ragged forward pass (one step may contain a 1-token decode of request A and a 90-token prefill of B).

Two variants, differing in what a step may contain:

  prefill_first (vLLM v0 default)
      If anyone needs prefill, this step is PREFILL ONLY; every decoding request stalls for it.
      Simple, and prefill runs at full efficiency. Cost: decode stalls (visible as ITL spikes).

  mixed (Orca's selective batching)
      Decodes and whole new prompts share one step. Nobody is skipped, but the step is as slow as the
      biggest prompt in it, so decodes still feel it. Chunked prefill (step 6) is the fix for both.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.request import make_work  # noqa: E402
from batchlib.scheduler import Policy  # noqa: E402


class ContinuousBatching(Policy):
    def __init__(self, mode="prefill_first", max_running=None):
        assert mode in ("prefill_first", "mixed")
        self.mode = mode
        self.max_running = max_running                      # optional cap below the number of KV slots
        self.name = "continuous" if mode == "prefill_first" else "continuous_mixed"

    def schedule(self, s):
        # 1. admission, FCFS: take waiting requests while a slot is free. This is the iteration-level part.
        cap = self.max_running or len(s.free_slots) + len(s.running)
        while s.waiting and s.free_slots and len(s.running) < cap:
            s.admit(s.waiting[0])

        prefill = [r for r in s.running if r.in_prefill]
        decode = [r for r in s.running if not r.in_prefill]

        if self.mode == "prefill_first" and prefill:
            return [make_work(r, r.pending) for r in prefill]          # decodes stall this step
        works = [make_work(r, 1) for r in decode]
        works += [make_work(r, r.pending) for r in prefill]            # whole prompts ride along (mixed)
        return works or None
