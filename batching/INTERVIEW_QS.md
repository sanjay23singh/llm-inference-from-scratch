# Interview questions: batching

Crisp answers. Numbers are deliberately absent: quote your own from `step*/results/`.

## Concept

**1. Static vs dynamic vs continuous batching: what's the difference?**
Static: the batch is a unit, formed once, runs until its longest request ends. Dynamic: same, but launches on a
size-or-timeout trigger; it only changes *when* a batch starts. Continuous (iteration-level scheduling, Orca): the
batch is re-decided before every forward pass; finished requests leave and waiting ones join immediately, so nothing
waits for batch-mates. Needs a slot-based KV cache and a ragged forward pass.

**2. Why is batching nearly free in decode but not in prefill?**
Decode is memory-bound: one step streams all weights for 1 token per sequence, so more sequences reuse the same bytes
until the ridge point (arithmetic intensity ~ N/2 FLOP/byte for fp32 weights). Prefill already has hundreds of tokens
per weight read, so it sits right of the ridge: compute-bound, cost ~ tokens. Attention reads per-sequence K/V, which is
*not* shared, so it doesn't get this benefit.

**3. What is a zombie row and how do you quantify its cost?**
A finished request still occupying its row in a static batch; the tensor shape is fixed so the engine keeps computing
it. Slot efficiency = sum(len) / (B x max(len)) per batch = E[len]/E[max len]. Grows worse with larger B and wider
length spread.

**4. What does selective batching mean (Orca)?**
Batch the ops that share weights (QKV/out projections, MLP, lm_head) across all tokens of all sequences as one flattened
`(N, C)` tensor; run attention per sequence because K/V isn't shared. That lets requests in different phases and with
different lengths share a step without padding.

**5. TTFT, TPOT, ITL: which one does prefill interference hurt, and why does the mean hide it?**
ITL tail. A long prefill step freezes every decoder for its duration: one huge gap per user. TPOT is the average gap
over the whole response, so one large gap barely moves it. Always look at ITL p99/max.

**6. How does chunked prefill work and what's the budget trade-off?**
Cap each step at B_tok tokens: all decodes first (1 token each), then fill the rest with prompt chunks; a prompt that
doesn't fit continues next step. Step time is bounded, so ITL has a ceiling. Small budget: tight ITL but long prompts
take more steps (higher TTFT) and each chunk re-streams the weights for few tokens. Large: approaches whole-prompt
prefill. Pick it just above the knee of the step-time curve.

## System design

**7. Design a scheduler for an LLM server with a TTFT SLO of 1 s and a TPOT SLO of 50 ms.**
Continuous batching + chunked prefill with a token budget sized so a full step stays under the TPOT SLO (from the
measured step-time curve). Admission: FCFS with a KV-memory check (not just a slot count). Backpressure: bounded queue,
reject or shed when predicted TTFT exceeds the SLO. Preemption (recompute or swap) when KV runs out. Track goodput, not
throughput: tokens from requests that met the SLO. Autoscale on queue depth / KV utilisation.

**8. Your static-batching service has p99 TTFT of 8 s at moderate load. What do you check first?**
Batch-formation wait and head-of-line blocking: queue time vs compute time per request. If the queue dominates and
output lengths vary, move to continuous batching. A cheap stopgap: dynamic batching with a timeout, and
length-bucketing if output length is predictable.

## What breaks at scale

**9. What limits concurrency in a continuous-batching server?**
KV memory, not compute. Each slot reserves `max_len` tokens; most requests use far fewer (internal fragmentation), and
variable sizes cause external fragmentation. Hence paged allocation (PagedAttention), prefix caching, and GQA/MLA to
shrink bytes per token. Concurrency x average context x bytes per token must fit in the memory left after weights.

**10. Continuous batching raised throughput but customers complain about choppy streaming. Why?**
Prefill-first scheduling: each admitted prompt is a prefill-only step during which all decoders stall. Fix: chunked
prefill, or disaggregate prefill and decode onto different workers so they don't share steps.

**11. Why might a smaller budget (or smaller max batch) *raise* TTFT for long prompts but not for short ones?**
A long prompt needs ceil(P / budget) steps; each step also carries all current decodes. Short prompts fit in one chunk.

**12. How would you test that your batching implementation is correct?**
Per-request tokens must equal single-request generation (bit-for-bit under greedy) for every policy; logits of a
chunked prefill must match the unchunked one; slots must never be shared by two live requests; every request finishes
with exactly its length. See `tests/test_correctness.py`.
