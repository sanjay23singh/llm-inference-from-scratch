#!/usr/bin/env bash
# Run the whole batching module in order. From the repo root:  bash batching/run_all.sh
# Total ~40 min on an M1 (CPU). Add DEVICE=mps to try Metal (per-sequence attention loop will be launch-bound there).
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cpu}"

echo "== correctness (must pass before any number means anything)"
python batching/tests/test_correctness.py

echo "== step 1: slot engine"
python batching/step1_slot_engine/bench_engine.py --device "$DEVICE"
echo "== step 2: static"
python batching/step2_static/sweep_variance.py --device "$DEVICE"
echo "== step 3: dynamic"
python batching/step3_dynamic/sweep_wait.py --device "$DEVICE"
echo "== step 4: continuous"
python batching/step4_continuous/run_continuous.py --device "$DEVICE"
echo "== step 5: ragged vs padded"
python batching/step5_ragged/padding_vs_ragged.py --device "$DEVICE"
echo "== step 6: chunked prefill"
python batching/step6_chunked_prefill/run_stall.py --device "$DEVICE"
echo "== step 7: calibrate, validate, compare"
python batching/step7_compare/calibrate.py --device "$DEVICE"
python batching/step7_compare/validate_sim.py --device "$DEVICE"
python batching/step7_compare/compare_all.py --device "$DEVICE"
python batching/step7_compare/make_brief.py
echo "done. Results in batching/step*/results, brief in batching/BLOG_BRIEF.md"
