"""Glue for the experiment scripts: build a model/engine, run a policy on a workload, save results with metadata."""
import json
import os
import platform
import time
from pathlib import Path

import torch

from . import paths  # noqa: F401
from .scheduler import Scheduler
from .sim_engine import CostModel, SimEngine, _shape
from .slot_engine import RealEngine
from .stats import per_request, summarize
from .workload import WorkloadSpec, describe, make_requests

from model import GPT2, GPT2Config  # noqa: E402

DEFAULT_SLO = dict(slo_ttft=2.0, slo_tpot=0.2)      # seconds. Edit per experiment; always printed with results.


def load_model(device="cpu", weights="auto"):
    """GPT-2 small. Timings do not depend on weight VALUES (same shapes, same bytes), only on shapes.
    'auto' tries pretrained and falls back to random weights when HuggingFace is unreachable."""
    torch.manual_seed(0)
    if weights in ("auto", "gpt2"):
        try:
            m = GPT2.from_pretrained("gpt2")
            kind = "gpt2-pretrained"
        except Exception as e:                       # offline sandbox
            if weights == "gpt2":
                raise
            print(f"[runner] pretrained GPT-2 unavailable ({type(e).__name__}); using random weights (same shapes)")
            m, kind = GPT2(GPT2Config()), "gpt2-random-init"
    else:
        m, kind = GPT2(GPT2Config()), "gpt2-random-init"
    return m.to(device).eval(), kind


COST_MODEL_PATH = Path(__file__).resolve().parents[1] / "step7_compare" / "results" / "cost_model.json"


def get_engine(kind, n_slots, max_len, device="cpu", weights="auto"):
    """kind='real' runs the model; kind='sim' replays the fitted cost model (see sim_engine.py).
    Returns (engine, weights_kind)."""
    if kind == "sim":
        if not COST_MODEL_PATH.exists():
            raise SystemExit(f"{COST_MODEL_PATH} missing: run step7_compare/calibrate.py first")
        return SimEngine(CostModel.load(COST_MODEL_PATH), n_slots, max_len), "cost-model"
    model, wk = load_model(device, weights)
    return RealEngine(model, n_slots, max_len, device), wk


def make_engine(model, n_slots, max_len, device="cpu"):
    return RealEngine(model, n_slots, max_len, device)


def meta(device, weights_kind, extra=None):
    cpu = platform.processor() or platform.machine()
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                cpu = ln.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    m = dict(device=device, weights=weights_kind, platform=platform.platform(), cpu=cpu,
             cpu_count=os.cpu_count(), torch=torch.__version__, torch_threads=torch.get_num_threads(),
             date=time.strftime("%Y-%m-%d %H:%M"))
    m.update(extra or {})
    return m


def run_one(policy, engine, spec: WorkloadSpec, slo=None, keep_steps=False):
    """Run one policy on one workload. Returns (summary dict, per-request lists, step log)."""
    slo = slo or DEFAULT_SLO
    reqs = make_requests(spec)
    done, steps = Scheduler(policy, engine, reqs).run()
    out = summarize(done, steps, **slo)
    out["policy"] = policy.name
    return out, per_request(done), (steps if keep_steps else None)


def decode_step_time(engine, n=None, ctx=96, reps=3):
    """Seconds for one decode step with n live sequences (measured, or predicted for SimEngine).
    Used to scale load sensibly."""
    n = n or engine.n_slots
    ws = _shape("decode", n, min(ctx, engine.max_len - 8), 0)
    engine.execute(ws)
    return sorted(engine.execute(ws)[1] for _ in range(reps))[reps // 2]


def rate_for_load(load, engine, spec: WorkloadSpec, ctx=96):
    """Arrival rate (req/s) that offers `load` x the engine's decode-bound capacity.

    capacity = n_slots sequences in flight, each needing out_mean steps of t_step seconds:
        capacity_req_s = n_slots / (t_step * out_mean)        (ignores prefill; so load>~0.8 is already saturated)
    Phrasing sweeps in 'load' instead of req/s keeps them meaningful on any machine."""
    t = decode_step_time(engine, ctx=ctx)
    return load * engine.n_slots / (t * spec.out_mean), t


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
