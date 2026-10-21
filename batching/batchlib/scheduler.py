"""The serving loop. One loop, many policies.

A serving engine is:      while there is work:  decide what goes into the next step -> run it -> update state.
The only thing that differs between static, dynamic, continuous and chunked-prefill batching is the
"decide" part. So the loop below is shared and each batching strategy is a small Policy class.

Clock
-----
Requests arrive at pre-drawn times (Poisson process). The loop keeps a virtual clock:
  * running a step advances the clock by that step's duration (really measured with RealEngine,
    predicted with SimEngine),
  * if the policy has nothing to run, the clock jumps to the next arrival (or the policy's own timer).
So we never sleep, yet every latency we report is real compute time plus real queueing.
"""
from . import paths  # noqa: F401
from .request import make_work  # noqa: F401  (re-exported for policies)


class Policy:
    """Subclass and implement schedule()."""
    name = "policy"
    wake_at = None            # a policy that wants to be woken up (a timer) sets this when it returns None

    def schedule(self, s: "Scheduler"):
        """Return a non-empty list[Work] to run next, or None if there is nothing to do right now.
        May call s.admit(req) to move a waiting request into a KV slot."""
        raise NotImplementedError

    def on_finish(self, req, s: "Scheduler"):
        """Called the moment a request emits its last token. Default: free its slot at once.
        (Continuous batching relies on this; static batching deliberately does NOT free it.)"""
        s.release(req)


class Scheduler:
    def __init__(self, policy: Policy, engine, reqs):
        self.policy, self.engine = policy, engine
        self.reqs = sorted(reqs, key=lambda r: r.arrival)
        self.now = 0.0
        self.waiting, self.running = [], []
        self.free_slots = list(range(engine.n_slots))
        self._next = 0
        self.steps = []                    # one dict per engine step, for utilisation / stall analysis

    # ---- state the policy can read / change ---------------------------------------------------------
    @property
    def more_arrivals(self):
        return self._next < len(self.reqs)

    def admit(self, req):
        """waiting -> running, and give it a KV slot."""
        self.waiting.remove(req)
        req.slot = self.free_slots.pop(0)
        req.t_admit = self.now
        self.running.append(req)

    def release(self, req):
        self.running.remove(req)
        self.free_slots.append(req.slot)
        req.slot = -1

    # ---- the loop -----------------------------------------------------------------------------------
    def _arrive(self):
        while self._next < len(self.reqs) and self.reqs[self._next].arrival <= self.now:
            self.waiting.append(self.reqs[self._next])
            self._next += 1

    def _idle(self):
        """Nothing to run: jump the clock to whatever happens next (an arrival or the policy's timer)."""
        t = []
        if self.more_arrivals:
            t.append(self.reqs[self._next].arrival)
        if self.policy.wake_at is not None and self.policy.wake_at > self.now:
            t.append(self.policy.wake_at)
        if not t:
            raise RuntimeError(f"{self.policy.name}: deadlock (waiting={len(self.waiting)}, running={len(self.running)})")
        self.now = max(self.now, min(t))

    def run(self):
        n_done = 0
        while n_done < len(self.reqs):
            self._arrive()
            self.policy.wake_at = None
            works = self.policy.schedule(self)
            if not works:
                self._idle()
                continue

            toks, dt = self.engine.execute(works)
            t0, self.now = self.now, self.now + dt

            n_dec = n_zom = n_pre = 0
            for w, tok in zip(works, toks):
                r = w.req
                r.n_ctx += len(w.tokens)
                if w.zombie:                                  # computed, thrown away
                    n_zom += 1
                    continue
                if w.start >= len(r.prompt):
                    n_dec += 1
                else:
                    n_pre += len(w.tokens)
                if w.sample:                                  # reached the end of known tokens -> a new token
                    r.generated.append(tok)
                    r.token_times.append(self.now)
                    if len(r.generated) >= r.max_new:
                        r.done, r.t_done = True, self.now
                        n_done += 1
                        self.policy.on_finish(r, self)
            self.steps.append(dict(t=t0, dt=dt, n_tok=sum(len(w.tokens) for w in works), n_dec=n_dec,
                                   n_zombie=n_zom, n_prefill_tok=n_pre, n_seq=len(works)))
        return self.reqs, self.steps
