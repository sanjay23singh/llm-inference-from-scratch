"""Step 3: DYNAMIC batching (what Triton/TorchServe call 'dynamic batching').

Static batching has a startup problem: if traffic is light it may wait forever for a full batch.
Dynamic batching adds a TIMEOUT:

    launch when the queue reaches `batch_size`  OR  the oldest request has waited `max_wait` seconds.

That bounds the batch-formation delay (good for TTFT at low load) but changes nothing about what happens
AFTER launch: the batch still runs until its longest member finishes, so zombie rows and head-of-line blocking
remain. Dynamic batching fixes WHEN a batch starts, not WHAT the batch does. Expect: better TTFT at low load,
roughly the same throughput as static at high load.

Bonus variant `length_aware=True` (an ORACLE; real systems must predict output length): when forming a batch,
group requests whose output lengths are close. This shrinks zombie rows without iteration-level scheduling.
It shows how much of continuous batching's win is 'just' length awareness; the rest is iteration-level
scheduling (a request never waits for its batch-mates), which no grouping can give you.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from step2_static.policy import StaticBatching  # noqa: E402


class DynamicBatching(StaticBatching):
    name = "dynamic"

    def __init__(self, batch_size=8, max_wait=0.5, length_aware=False):
        super().__init__(batch_size)
        self.max_wait = max_wait
        self.length_aware = length_aware
        if length_aware:
            self.name = "bucketed"

    def ready(self, s):
        if len(s.waiting) >= self.B or not s.more_arrivals:
            return True
        # waiting[0] is the oldest. The 1e-9 absorbs float rounding after the clock jumps to exactly arrival+max_wait.
        return s.now - s.waiting[0].arrival >= self.max_wait - 1e-9

    def on_not_ready(self, s):
        # ask the loop to wake us when the oldest request's timer expires (else it would sleep until next arrival)
        if s.waiting:
            self.wake_at = s.waiting[0].arrival + self.max_wait

    def pick(self, s):
        if not self.length_aware:
            return s.waiting[: self.B]
        # oldest request always goes (no starvation); fill with the waiting requests closest in output length
        first = s.waiting[0]
        rest = sorted(s.waiting[1:], key=lambda r: abs(r.max_new - first.max_new))
        return [first] + rest[: self.B - 1]
