"""Step 2: STATIC batching.

The simplest scheduler that uses a batch at all:

    1. wait until `batch_size` requests are queued
    2. prefill them together, then decode them together
    3. the batch runs until its LONGEST request finishes
    4. only then are slots freed and the next batch formed

This is what `model.generate(batch_of_prompts)` does, and what the Step 5c experiment (batch_decode.py) assumed:
all requests start together and the batch only ends when everyone is done.

Where the time goes (the problem this step lets you SEE)
--------------------------------------------------------
  * ZOMBIE ROWS. A request that finished at token 20 keeps its row until the longest request in the batch
    finishes at token 120. The engine keeps computing that row (real static batching does: the tensor shape is
    fixed) and throws the result away. Slot efficiency = mean(output length) / max(output length) in the batch.
  * HEAD-OF-LINE BLOCKING. A request that arrives one step after a batch started waits for the whole batch.
  * BATCH FORMATION. Waiting to fill the batch delays the first request in it.

Generosity of this baseline: prompts are NOT padded (ragged prefill) and the cache is preallocated, so the only
waste left is the schedule itself. Padding waste is measured separately in step5_ragged. Real static batching is
worse than the numbers you get here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.request import make_work, make_zombie_work  # noqa: E402
from batchlib.scheduler import Policy  # noqa: E402


class StaticBatching(Policy):
    name = "static"

    def __init__(self, batch_size=8):
        self.B = batch_size
        self.batch = []

    # --- how a batch is chosen: FIFO. Subclasses (step3) change launch timing and selection ---------
    def ready(self, s):
        """Launch when full, or when no more requests will ever arrive (flush the tail)."""
        return len(s.waiting) >= self.B or (len(s.waiting) > 0 and not s.more_arrivals)

    def pick(self, s):
        return s.waiting[: self.B]

    def schedule(self, s):
        if not self.batch:
            if not s.waiting or not self.ready(s):
                self.on_not_ready(s)
                return None
            self.batch = list(self.pick(s))
            for r in self.batch:
                s.admit(r)
            # step 1 of the batch: prefill everyone's whole prompt in one ragged step
            return [make_work(r, len(r.prompt)) for r in self.batch]

        # every following step: one token for every row, including rows that already finished
        return [make_zombie_work(r) if r.done else make_work(r, 1) for r in self.batch]

    def on_not_ready(self, s):
        pass

    def on_finish(self, req, s):
        # DO NOT free this request's slot: the batch owns it until its LAST member finishes.
        # (This one line is the whole difference from continuous batching, whose on_finish frees at once.)
        if all(r.done for r in self.batch):
            for r in self.batch:
                s.release(r)                     # all slots freed together, at the very end
            self.batch = []
