"""Import this first. It puts the earlier repo steps on sys.path (same trick the kv_cache steps use),
so batching code can reuse GPT2 (step1), CachedGPT2 (step3), sampling (step4) and Request metrics (step0d)
instead of copying them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
BATCHING = ROOT / "batching"

for p in (BATCHING,
          ROOT / "kv_cache" / "step1_forward_pass",
          ROOT / "kv_cache" / "step3_kv_cache",
          ROOT / "kv_cache" / "step4_sampling",
          ROOT / "foundations" / "step0d_metrics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
