"""Step 7d: regenerate batching/BLOG_BRIEF.md from the results/*.json files.

The brief is INPUT for the blog-writing agent, not prose. Every number in it is read from a JSON in this repo,
so it cannot drift from the experiments; re-run after re-measuring (e.g. on a different machine) and the brief updates.
Each number is labelled: measured (real engine), simulated (cost model), computed (from the workload / arithmetic).
It also tests the PREDICTIONS the step notes ask you to make and marks each OK / MISS: a missed prediction is a
section of the post, not a failure.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from batchlib import paths  # noqa: E402,F401

R = lambda step, name: ROOT / step / "results" / name  # noqa: E731


def load(step, name):
    p = R(step, name)
    return json.loads(p.read_text()) if p.exists() else None


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(" --- " for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rel(p):
    return f"`batching/{p}`"


def main():
    eb = load("step1_slot_engine", "engine_bench.json")
    sv = load("step2_static", "static_variance.json")
    dw = load("step3_dynamic", "dynamic_wait.json")
    co = load("step4_continuous", "continuous.json")
    pd = load("step5_ragged", "padding.json")
    st = load("step6_chunked_prefill", "stall.json")
    cm = load("step7_compare", "cost_model.json")
    sim = load("step7_compare", "sim_validation.json")
    hl = load("step7_compare", "compare_headline.json")
    ld = load("step7_compare", "compare_load.json")
    vr = load("step7_compare", "compare_variance.json")
    sl = load("step7_compare", "compare_slots.json")

    m = (hl or co or sv or {}).get("meta", {})
    L = []
    a = L.append
    a("# Blog brief: Part 3, batching (input for the blog-writing agent, not prose)\n")
    a(f"**Results machine:** {m.get('cpu', '?')}, {m.get('cpu_count', '?')} cores, device `{m.get('device', '?')}`, "
      f"{m.get('platform', '?')}, torch {m.get('torch', '?')}, weights `{m.get('weights', '?')}` "
      f"(timings depend on shapes, not weight values), run {m.get('date', '?')}.")
    a("If this is not the M1 in `foundations/HARDWARE.md`, re-run `bash batching/run_all.sh` on the Mac and "
      "`python batching/step7_compare/make_brief.py`; every number below regenerates. Ratios and rankings carry across "
      "machines, absolute milliseconds do not. Quote hardware peaks from `foundations/HARDWARE.md` only.\n")
    a("Labels: **measured** = real engine wall-clock, **simulated** = cost model (`step7_compare/results/cost_model.json`), "
      "**computed** = from the workload or arithmetic.\n")

    # ------------------------------------------------------------------ 1. problem
    a("## 1. The problem and which post left it open\n")
    a("Part 2 ended with three open items: *continuous batching*, *cache allocation (torch.cat copies)*, *long prefills "
      "stalling decode*. It showed batching is almost free in decode but assumed a tidy batch: all requests start "
      "together and are the same length. Real traffic is neither. Do not re-cover roofline, ridge point, KV sizing or GQA (Part 2).\n")
    a("Thread of the post: static batching wastes the batch -> a timeout (dynamic) fixes the wrong thing -> make the batch "
      "per-iteration (continuous) -> that exposes prefill stalls -> chunk the prefill -> the new ceiling is KV memory.\n")

    # ------------------------------------------------------------------ 2. attempts
    a("## 2. Attempts: what was tried, the number, why it failed or worked\n")

    if sv:
        a("### 2.1 Static batching (the baseline from Part 2's experiment)")
        a("Setup: saturated arrivals, output-length spread sigma swept, 8 slots. Generous baseline: no prompt padding, "
          "preallocated cache; only the schedule's waste remains. "
          f"Source: {rel('step2_static/results/static_variance.json')}\n")
        a(table(["sigma", "length CV (computed)", "slot efficiency, expected (computed)", "slot efficiency (measured)",
                 "avg live rows/step (measured)", "tok/s (measured)", "TTFT p50 s", "TTFT p99 s"],
                [[f"{r['sigma']:g}", f"{r['cv']:.2f}", f"{r['expected_eff']:.2f}", f"{r['slot_eff']:.2f}",
                  f"{r['avg_decode_batch']:.1f}", f"{r['throughput_tps']:.1f}", f"{r['ttft_p50']:.2f}", f"{r['ttft_p99']:.2f}"]
                 for r in sv["rows"]]))
        a("\nWhy it fails: a finished request keeps its row until the longest request ends (zombie rows). Efficiency = "
          "E[len]/E[max len per batch]; the measured value tracks the computed one, so the loss is purely the schedule.\n")

    if dw:
        a("### 2.2 Dynamic batching (timeout) and the length-bucketing oracle")
        a(f"Source: {rel('step3_dynamic/results/dynamic_wait.json')}. `avg batch` = live decode rows per step (measured).\n")
        for load_, rows in dw["loads"].items():
            a(f"Load {load_}x decode capacity:\n")
            a(table(["policy", "TTFT p50 s", "TTFT p99 s", "queue p50 s", "tok/s", "slot eff", "avg batch"],
                    [[r["label"], f"{r['ttft_p50']:.2f}", f"{r['ttft_p99']:.2f}", f"{r['queue_p50']:.2f}",
                      f"{r['throughput_tps']:.1f}", f"{r['slot_eff']:.2f}", f"{r['avg_decode_batch']:.1f}"] for r in rows]))
            a("")
        a("Why: the timeout bounds batch-formation delay, so it helps at light load (at the price of near-empty batches) "
          "and does nothing once a full batch is always waiting. The bucketed row is an ORACLE (uses true output length) "
          "and shows how much of the continuous-batching win is just length awareness.\n")

    if eb:
        a("### 2.3 Replacing torch.cat with a slot cache (Part 2's open allocation problem)")
        a(f"Batch {eb['rows'][0]['batch']}; source {rel('step1_slot_engine/results/engine_bench.json')}. "
          f"Weights {eb['weights_mb']:.0f} MB (computed). `cat copied` is computed: 2 x layers x B x (S+1) x hidden x 4 B.\n")
        a(table(["context", "append via cat ms (measured)", "append in place ms (measured)", "cat copies MB (computed)",
                 "decode step, Step 3 model ms", "decode step, slot engine ms"],
                [[r["ctx"], f"{r['cat_ms']:.1f}", f"{r['inplace_ms']:.2f}", f"{r['cat_copied_mb']:.0f}",
                  f"{r['step3_step_ms']:.0f}", f"{r['slot_step_ms']:.0f}"] for r in eb["rows"]]))
        a("\nWhy: cat copies the whole cache every step (grows with batch x context); in-place writes one row. The slot "
          "engine pays a python loop over sequences for attention (a real engine fuses it). Report which side wins on this machine, honestly.\n")

    if co or hl:
        src = hl or co
        res = src["results"]
        a("### 2.4 Continuous batching (iteration-level scheduling)")
        a(f"Same workload for every policy, 0.8x load. Source: {rel('step7_compare/results/compare_headline.json' if hl else 'step4_continuous/results/continuous.json')}. "
          f"SLO for goodput: {m.get('slo', 'see json')}.\n")
        names = list(res)
        a(table(["policy", "tok/s", "goodput tok/s", "TTFT p50 s", "TTFT p99 s", "TPOT p50 ms", "ITL p99 ms", "ITL max ms", "slot eff", "avg batch"],
                [[n, f"{res[n]['throughput_tps']:.1f}", f"{res[n]['goodput_tps']:.1f}", f"{res[n]['ttft_p50']:.2f}",
                  f"{res[n]['ttft_p99']:.2f}", f"{res[n]['tpot_p50']*1e3:.0f}", f"{res[n]['itl_p99']*1e3:.0f}",
                  f"{res[n]['itl_max']*1e3:.0f}", f"{res[n]['slot_eff']:.2f}", f"{res[n]['avg_decode_batch']:.1f}"]
                 for n in names]))
        a("\nWhy it works: nothing waits for batch-mates, so no zombie rows and TTFT stops including someone else's decode. "
          "Why it is not free: prefill-first makes every admitted prompt a prefill-only step during which all decoders stall (ITL tail). "
          "At 0.8x load every unsaturated policy produces tokens at the arrival rate; throughput only separates policies at saturation, "
          "which is why the load sweep below matters.\n")

    if pd:
        a("### 2.5 Padding vs ragged batches")
        a(f"Batch {pd['batch']}, T_max {pd['tmax']}. Source {rel('step5_ragged/results/padding.json')}. Predicted speedup is computed: B x T_max / sum(T_i).\n")
        a(table(["real tokens / padded", "prefill padded ms", "prefill ragged ms", "speedup (measured)", "predicted (computed)",
                 "decode padded ms", "decode ragged ms", "K/V bytes ratio (computed)"],
                [[f"{r['useful']*100:.0f}%", f"{r['prefill_padded_ms']:.0f}", f"{r['prefill_ragged_ms']:.0f}",
                  f"{r['prefill_speedup']:.2f}x", f"{r['prefill_predicted']:.2f}x", f"{r['decode_padded_ms']:.0f}",
                  f"{r['decode_ragged_ms']:.0f}", f"{r['kv_read_ratio']:.2f}x"] for r in pd["rows"]]))
        a("\nWhy: prefill is compute-bound, so padding is real work; decode streams weights once either way, so padding only "
          "matters through the K/V reads, which are a small share of a GPT-2 small step at these lengths. Padded cost is the pad tokens "
          "run through the same kernels (a faithful stand-in for a masked tensor).\n")

    if st:
        a("### 2.6 Chunked prefill")
        a(f"15% of prompts are long. Source {rel('step6_chunked_prefill/results/stall.json')}.\n")
        a(table(["policy", "long-prompt TTFT p50 s", "short TTFT p50 s", "ITL p50 ms", "ITL p99 ms", "ITL max ms", "max step ms", "tok/s"],
                [[r["label"], f"{r['long_ttft_p50']:.2f}", f"{r['short_ttft_p50']:.2f}", f"{r['itl_p50']*1e3:.0f}",
                  f"{r['itl_p99']*1e3:.0f}", f"{r['itl_max']*1e3:.0f}", f"{r['max_step_s']*1e3:.0f}",
                  f"{r['throughput_tps']:.1f}"] for r in st["rows"]]))
        a("\nWhy: step time is bounded by the budget, so ITL has a ceiling; the long prompt itself pays by needing "
          "ceil(P/budget) steps. A small budget also re-streams the weights for few tokens.\n")

    if cm:
        a("### 2.7 Cost model and simulator (used for the sweeps only when sweep_engine is sim; the knee is still a measured fit)")
        a(f"Source {rel('step7_compare/results/cost_model.json')}. Fit: t0 {cm['t0']*1e3:.1f} ms, c {cm['c']*1e9:.1f} ns/attention pair, "
          f"d {cm['d']*1e6:.0f} us/sequence, R^2 {cm['r2']:.3f}, worst relative error {cm['max_rel_err']*100:.0f}% (all measured/fitted). "
          f"Fitted knee (a step 10% slower than a 1-token step): **{cm['knee_n']} tokens**; HARDWARE.md ridge for this device "
          f"predicts ~40 tokens (M1 CPU fp32: 2 x 19.8) if weights dominate. Disagreement is a finding.")
        if sim:
            a("\nSimulator vs real engine, same workload, six policies "
              f"({rel('step7_compare/results/sim_validation.json')}):\n")
            a(table(["metric", "Spearman rank agreement", "median abs error"],
                    [[k, f"{v['spearman']:.2f}", f"{v['median_abs_err_pct']:.0f}%"] for k, v in sim["summary"].items()]))
        a("")

    sweep_label = "measured" if ((ld or vr or sl or {}).get("meta", {}).get("sweep_engine") == "real") else "simulated"
    if ld:
        a(f"### 2.8 Load sweep ({sweep_label}, median of 3 seeds)")
        a(f"Source {rel('step7_compare/results/compare_load.json')}; plot `batching/step7_compare/results/compare_load.png`.\n")
        names = list(ld["series"])
        a(table(["load"] + [f"{n} tok/s" for n in names], [[f"{x:g}"] + [f"{ld['series'][n]['throughput_tps'][i]:.1f}" for n in names]
                                                           for i, x in enumerate(ld["loads"])]))
        a("")
        a(table(["load"] + [f"{n} TTFT p99 s" for n in names], [[f"{x:g}"] + [f"{ld['series'][n]['ttft_p99'][i]:.2f}" for n in names]
                                                                for i, x in enumerate(ld["loads"])]))
        a("")
    if vr:
        a(f"### 2.9 Output-length variance sweep ({sweep_label}, saturated)")
        names = list(vr["series"])
        a(table(["length CV (computed)"] + [f"{n} tok/s" for n in names],
                [[f"{c:.2f}"] + [f"{vr['series'][n]['throughput_tps'][i]:.1f}" for n in names] for i, c in enumerate(vr["cv"])]))
        a("")
    if sl:
        a(f"### 2.10 Concurrency vs reserved KV memory ({sweep_label}, saturated)")
        a(f"Each slot reserves {sl['slot_len']} tokens; a request averages {sl['mean_tokens_used']:.0f} (computed) -> "
          f"**{sl['internal_fragmentation']*100:.0f}% of reserved KV is unused** (internal fragmentation). Source {rel('step7_compare/results/compare_slots.json')}.\n")
        names = list(sl["series"])
        a(table(["slots", "KV reserved MB (computed)"] + [f"{n} tok/s" for n in names],
                [[s, f"{sl['reserved_mb'][i]:.0f}"] + [f"{sl['series'][n]['throughput_tps'][i]:.1f}" for n in names]
                 for i, s in enumerate(sl["slots"])]))
        a("")

    # ------------------------------------------------------------------ 3. predictions
    a("## 3. Predictions vs outcome (auto-checked)\n")
    P = []
    if sv:
        err = max(abs(r["slot_eff"] - r["expected_eff"]) for r in sv["rows"])
        P.append(("Static slot efficiency equals E[len]/E[max len] computed from the workload", err < 0.05, f"max gap {err:.3f}"))
    if dw:
        hi = dw["loads"][max(dw["loads"], key=float)]
        st0 = hi[0]["throughput_tps"]
        gap = max(abs(r["throughput_tps"] - st0) / st0 for r in hi if r["label"].startswith("dynamic"))
        P.append(("At saturation dynamic ~ static (timeout never fires)", gap < 0.15, f"max throughput gap {gap*100:.0f}%"))
    if co or hl:
        res = (hl or co)["results"]
        P.append(("Continuous TTFT p50 < half of static's", res["continuous"]["ttft_p50"] < res["static"]["ttft_p50"] / 2,
                  f"{res['continuous']['ttft_p50']:.2f}s vs {res['static']['ttft_p50']:.2f}s"))
        P.append(("Continuous (prefill-first) ITL p99 is WORSE than static's", res["continuous"]["itl_p99"] > res["static"]["itl_p99"],
                  f"{res['continuous']['itl_p99']*1e3:.0f}ms vs {res['static']['itl_p99']*1e3:.0f}ms"))
        P.append(("Continuous slot efficiency = 1.00", abs(res["continuous"]["slot_eff"] - 1) < 1e-6, f"{res['continuous']['slot_eff']:.2f}"))
    if pd:
        r = pd["rows"][-1]
        P.append(("Ragged prefill speedup within 20% of computed B*Tmax/sum(T) at the widest spread",
                  abs(r["prefill_speedup"] - r["prefill_predicted"]) / r["prefill_predicted"] < 0.2,
                  f"{r['prefill_speedup']:.2f}x vs {r['prefill_predicted']:.2f}x"))
    if st:
        rows = {r["label"]: r for r in st["rows"]}
        ch = sorted((r for r in st["rows"] if r["budget"]), key=lambda r: r["budget"])
        P.append(("Chunked ITL max < prefill-first ITL max (best budget)", min(r["itl_max"] for r in ch) < rows["prefill-first"]["itl_max"],
                  f"{min(r['itl_max'] for r in ch)*1e3:.0f}ms vs {rows['prefill-first']['itl_max']*1e3:.0f}ms"))
        P.append(("Long-prompt TTFT rises as the budget shrinks", ch[0]["long_ttft_p50"] > ch[-1]["long_ttft_p50"],
                  f"budget {ch[0]['budget']}: {ch[0]['long_ttft_p50']:.2f}s vs budget {ch[-1]['budget']}: {ch[-1]['long_ttft_p50']:.2f}s"))
    if sim:
        P.append(("Simulator keeps the policy ranking for throughput (rho >= 0.9)", sim["summary"]["throughput_tps"]["spearman"] >= 0.9,
                  f"rho {sim['summary']['throughput_tps']['spearman']:.2f}"))
    a(table(["prediction", "result", "evidence"], [[p, "OK" if ok else "MISS", e] for p, ok, e in P]))
    a("\nA MISS is not a bug to hide: say what the machine did instead and why (profile it, or list it under section 6).\n")

    # ------------------------------------------------------------------ 4. plots
    a("## 4. Plots to include\n")
    for p in ["step1_slot_engine/results/engine_bench.png", "step2_static/results/static_variance.png",
              "step3_dynamic/results/dynamic_wait.png", "step4_continuous/results/continuous_occupancy.png",
              "step4_continuous/results/continuous_bars.png", "step5_ragged/results/padding.png",
              "step6_chunked_prefill/results/stall_steps.png", "step6_chunked_prefill/results/stall_tradeoff.png",
              "step7_compare/results/compare_load.png", "step7_compare/results/compare_variance.png",
              "step7_compare/results/compare_slots.png", "step7_compare/results/sim_validation.png"]:
        a(f"- {rel(p)}{'' if R(p.split('/')[0], p.split('/')[-1]).exists() else '  (missing: run the step)'}")
    a("\nBest single figure for the post: `continuous_occupancy.png` (rows computed vs rows useful: the waste is visible).\n")

    # ------------------------------------------------------------------ 5. interview
    a("## 5. Interview questions\n\nIn `batching/INTERVIEW_QS.md`.\n")

    # ------------------------------------------------------------------ 6. gaps
    a("## 6. Still unexplained / unprofiled\n")
    a("- Per-sequence attention is a python loop; its overhead vs a fused varlen kernel is not profiled (needs CUDA/Triton, Modules 4 and 7).")
    a("- The step-time curve's shape at small N (kernel choice, thread count) is taken as measured, not explained; see the knee vs ridge comparison in 2.7.")
    a("- Output lengths are assigned by the workload, not by EOS; real length distributions are heavier-tailed and correlated with the prompt.")
    a("- No preemption: when slots are full a long request at the head of the queue blocks the rest (FCFS). Recompute vs swap is Module 3.")
    a("- Greedy decoding only; per-request sampling params and stop strings are not batched.")
    a("- fp32 weights; with fp16/INT8 the knee moves (fewer bytes per weight -> lower ridge in tokens).")
    a("- Prefix sharing between requests is not modelled (Module 3).\n")

    # ------------------------------------------------------------------ 7. open problem
    a("## 7. Open problem that leads to the next post\n")
    if sl:
        a(f"Throughput keeps rising with concurrency, but each slot reserves a full-length K/V region and {sl['internal_fragmentation']*100:.0f}% of it is unused on this workload "
          f"(computed), while the reservation grows linearly with slots (2.10). KV memory, not compute, caps concurrency. "
          "Next: KV memory management: fragmentation, PagedAttention and block tables, prefix caching, preemption (swap vs recompute), and what GQA/MLA do to bytes per token.")
    else:
        a("KV memory, not compute, caps concurrency: every slot reserves a full-length K/V region. Next: PagedAttention, block tables, prefix caching, preemption.")

    (ROOT / "BLOG_BRIEF.md").write_text("\n".join(L) + "\n")
    print(f"wrote {ROOT / 'BLOG_BRIEF.md'}  ({len(L)} lines, {sum(1 for p in P if p[1])}/{len(P)} predictions OK)")


if __name__ == "__main__":
    main()
