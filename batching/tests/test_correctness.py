"""Correctness tests. Batching changes WHEN work happens, never WHAT is computed:
every request must get exactly the tokens it would get running alone.

  A. ragged_forward == the Step 3 model (logits match) for a prompt, a decode step, and a prefill split in chunks
  B. every scheduling policy produces, for every request, the same tokens as single-request generation
  C. scheduler invariants: no slot is ever used by two live requests, every request finishes, lengths are right

Runs offline on a tiny random-weight GPT-2 (seconds). Pass --real to also check against pretrained GPT-2.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batchlib import paths  # noqa: E402,F401
from batchlib.request import Work, make_work  # noqa: E402
from batchlib.scheduler import Scheduler  # noqa: E402
from batchlib.slot_engine import RealEngine, SlotKV, ragged_forward  # noqa: E402
from batchlib.workload import WorkloadSpec, make_requests  # noqa: E402
from kv_model import CachedGPT2, GPT2Config  # noqa: E402
from step2_static.policy import StaticBatching  # noqa: E402
from step3_dynamic.policy import DynamicBatching  # noqa: E402
from step4_continuous.policy import ContinuousBatching  # noqa: E402
from step6_chunked_prefill.policy import ChunkedPrefill  # noqa: E402


class R:                                   # minimal stand-in for Req: ragged_forward only reads .slot
    def __init__(self, slot):
        self.slot = slot


@torch.no_grad()
def reference(model, prompt, n_new):
    """Single-request greedy generation with the Step 3 KV-cache model (the ground truth)."""
    logits, cache = model(torch.tensor([prompt]))
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt.item()]
    for _ in range(n_new - 1):
        logits, cache = model(nxt, cache)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
    return out


def test_a_engine_matches_step3(model, cfg):
    torch.manual_seed(1)
    p = torch.randint(0, cfg.vocab_size, (23,)).tolist()
    ref_logits, cache = model(torch.tensor([p]))

    # whole prefill
    kv = SlotKV(cfg, 3, 64)
    _, lg = ragged_forward(model, kv, [Work(R(1), 0, p, True)], return_logits=True)
    assert torch.allclose(lg[0], ref_logits[0, -1], atol=1e-5), "prefill logits differ"

    # prefill in 3 chunks (7 + 9 + 7): chunk k must see the K/V written by chunks < k
    kv = SlotKV(cfg, 3, 64)
    for a, b in ((0, 7), (7, 16), (16, 23)):
        _, lg = ragged_forward(model, kv, [Work(R(2), a, p[a:b], b == 23)], return_logits=True)
    assert torch.allclose(lg[0], ref_logits[0, -1], atol=1e-5), "chunked prefill logits differ"

    # one decode step after the prompt
    tok = ref_logits[0, -1].argmax().item()
    ref_next, _ = model(torch.tensor([[tok]]), cache)
    _, lg = ragged_forward(model, kv, [Work(R(2), 23, [tok], True)], return_logits=True)
    assert torch.allclose(lg[0], ref_next[0, -1], atol=1e-5), "decode logits differ"

    # ragged: three different sequences (prefill chunk + 2 decodes) in ONE step, different slots
    kv = SlotKV(cfg, 3, 64)
    seqs = [torch.randint(0, cfg.vocab_size, (n,)).tolist() for n in (11, 17, 5)]
    for i, sq in enumerate(seqs[:2]):                                  # prefill first two separately
        ragged_forward(model, kv, [Work(R(i), 0, sq, True)])
    ws = [Work(R(0), 11, [3], True), Work(R(1), 17, [4], True), Work(R(2), 0, seqs[2], True)]
    _, lg = ragged_forward(model, kv, ws, return_logits=True)
    for row, (sq, extra) in enumerate(((seqs[0], [3]), (seqs[1], [4]), (seqs[2], []))):
        want, _ = model(torch.tensor([sq + extra]))
        assert torch.allclose(lg[row], want[0, -1], atol=1e-5), f"ragged row {row} differs"
    print("A  PASS  engine == Step 3 model (whole prefill, chunked prefill, decode, mixed ragged step)")


POLICIES = {
    "static": lambda: StaticBatching(batch_size=4),
    "dynamic": lambda: DynamicBatching(batch_size=4, max_wait=0.05),
    "bucketed": lambda: DynamicBatching(batch_size=4, max_wait=0.05, length_aware=True),
    "continuous": lambda: ContinuousBatching("prefill_first"),
    "continuous_mixed": lambda: ContinuousBatching("mixed"),
    "chunked(budget=7)": lambda: ChunkedPrefill(token_budget=7),
    "chunked(budget=32)": lambda: ChunkedPrefill(token_budget=32),
}


def test_bc_policies(model, cfg):
    spec = WorkloadSpec(n=14, rate=40.0, prompt_min=5, prompt_max=40, out_mean=10, out_sigma=0.7,
                        out_min=2, out_cap=24, vocab=cfg.vocab_size, seed=3)
    base = make_requests(spec)
    truth = {r.rid: reference(model, r.prompt, r.max_new) for r in base}

    for name, make in POLICIES.items():
        reqs = [r.reset() for r in make_requests(spec)]
        eng = RealEngine(model, n_slots=4, max_len=spec.max_total(), device="cpu")
        s = Scheduler(make(), eng, reqs)

        # invariant: wrap execute to check that live requests never share a slot
        orig = eng.execute

        def checked(works, orig=orig, s=s):
            slots = [w.req.slot for w in works]
            assert len(slots) == len(set(slots)), "two sequences in one step share a slot"
            live = [r.slot for r in s.running]
            assert len(live) == len(set(live)), "two running requests share a slot"
            return orig(works)
        eng.execute = checked

        done, steps = s.run()
        for r in done:
            assert r.done and len(r.generated) == r.max_new, f"{name}: rid {r.rid} incomplete"
            assert r.generated == truth[r.rid], f"{name}: rid {r.rid} tokens differ from single-request run"
            assert all(b > a for a, b in zip(r.token_times, r.token_times[1:])), "token times not increasing"
            assert r.t_admit >= r.arrival - 1e-12 and r.token_times[0] > r.t_admit - 1e-12
        assert sorted(s.free_slots) == [0, 1, 2, 3] and not s.running, f"{name}: slots leaked"
        print(f"B/C PASS  {name:20s} {len(steps):4d} steps, all {len(done)} requests match single-request output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also load pretrained GPT-2 (needs HF access)")
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=200, n_ctx=256, n_layer=3, n_head=4, n_embd=64)
    m = CachedGPT2(cfg).eval()
    test_a_engine_matches_step3(m, cfg)
    test_bc_policies(m, cfg)
    if args.real:
        m = CachedGPT2.from_pretrained("gpt2").eval()
        test_a_engine_matches_step3(m, m.cfg)
    print("\nALL PASS")
