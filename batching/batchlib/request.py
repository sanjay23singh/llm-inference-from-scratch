"""Request state and the unit of work the engine executes.

THE ONE IDEA that keeps every scheduling policy tiny:

    A request owns a sequence of "known" tokens = prompt + generated so far.
    Some prefix of it already has K/V in the cache (n_ctx). The rest is "pending".

    Each engine step feeds SOME of the pending tokens of SOME requests.
        prefill (whole prompt)   -> pending = prompt_len, feed all of it
        chunked prefill          -> pending = prompt_len, feed a slice
        decode                   -> pending = 1 (the last generated token), feed it
    If the step consumed ALL pending tokens, the last position's logits give the next token.

So "prefill vs decode" is not two code paths, just two values of n_tokens. That is exactly why
chunked prefill and mixed batches (steps 5 and 6) cost no extra engine code.
"""
from dataclasses import dataclass, field

from . import paths  # noqa: F401
from metrics import Request  # noqa: E402  (foundations/step0d_metrics: ttft(), tpot(), meets())


@dataclass
class Req(Request):
    rid: int = 0
    prompt: list = field(default_factory=list)
    max_new: int = 2                 # output length is fixed by the workload (stands in for "EOS arrives here")
    slot: int = -1                   # KV slot, -1 while waiting
    n_ctx: int = 0                   # tokens whose K/V already sit in the cache
    generated: list = field(default_factory=list)
    done: bool = False
    t_admit: float = float("nan")    # when it got a KV slot (end of queueing)
    t_done: float = float("nan")

    @property
    def n_known(self):
        return len(self.prompt) + len(self.generated)

    @property
    def pending(self):               # known tokens not yet in the cache
        return self.n_known - self.n_ctx

    @property
    def in_prefill(self):
        return self.n_ctx < len(self.prompt)

    def reset(self):
        """Clear run state so the same request can be replayed under another policy."""
        self.token_times = []
        self.slot, self.n_ctx, self.generated = -1, 0, []
        self.done, self.t_admit, self.t_done = False, float("nan"), float("nan")
        return self


@dataclass
class Work:
    """One sequence's contribution to one engine step."""
    req: Req
    start: int                       # absolute position of the first token fed (= tokens already cached)
    tokens: list                     # token ids fed this step
    sample: bool                     # does this step reach the end of the known tokens? then we need logits
    zombie: bool = False             # static batching only: a finished request still occupying a batch slot


def make_work(req: Req, n: int) -> Work:
    """Feed the next n pending tokens of req."""
    assert 0 < n <= req.pending, (n, req.pending)
    toks = (req.prompt + req.generated)[req.n_ctx: req.n_ctx + n]
    return Work(req, req.n_ctx, toks, sample=(n == req.pending))


def make_zombie_work(req: Req) -> Work:
    """A finished request that static batching keeps computing. The engine does real work for it
    (that is the waste we want to measure); the scheduler discards the output."""
    # sample=True on purpose: a real static batch also runs lm_head and softmax for finished rows
    return Work(req, req.n_ctx, [req.generated[-1]], sample=True, zombie=True)
