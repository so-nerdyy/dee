# Legal ahead-of-router work: scoping verdict

Status: ANALYSIS COMPLETE on `research/phase2-legal-prefetch` @ base
56dad3c (phase2-integration-lru-fix head). Analysis only — no engine code
changed. Quantified by sealed-journal replay
(`tools/phase2_legal_prefetch_eval.py`; results in `results/`), the same
two-level tier model validated in the serialization verdict
(`research/phase2-concurrent-fill` @ 5adda90).

This memo answers T2's open condition: SERIALIZATION_VERDICT.md §7(b)
kept a conditional GO for the concurrent-fill repair "if a legal
ahead-of-router candidate source exists." Scope here: enumerate
everything that can legally begin before the authoritative router commits
for a token, quantify the bytes it can move, and decide whether any of it
reopens the repair.

## TL;DR

- **The serialization NO-GO stands.** Legal ahead-of-router sources exist
  but are capacity-bound, not knowledge-bound: the binding resource is
  idle-bank time, and there is only ~0.5–1.1 expert-records' worth of it
  per decode row.
- **Even an oracle caps at ~29% of the 86.3 s response fill wall**
  (best cell: −24.8 s decode fill). Every *legal* source delivers ≲1.6 s.
- The only *exact* early source (hash layers 0–2) yields 14–42 preads =
  **0.54–1.60 s/response** — real but ~1–4% of decode fill. DEFER.
- Speculative top-k hints are legal but useless here: prev-token recall
  on the *miss stream* is **0.54%** — misses are precisely the records
  that don't repeat. NO-GO.
- Shared expert, dense tensors, router weights, MTP buckets: **zero
  expert-bank bytes** to move early. Nothing to prefetch.
- The lever remains what Phase 1/T2 said: fewer cold bytes
  (residency/reuse, regime-C contract) and bank placement
  (>~0.6–0.7 GiB/s storage), not earlier submission on this bank.

## Legality framework (exactness contract)

Per AGENTS.md + `research/route-pipeline/OFFICIAL_LOOKAHEAD.md`: a route
is official only if computed from the model's own inputs. Prediction may
drive prefetch *hints* only; the native router stays authoritative. A
wrong prefetch may waste bandwidth — it may never change which expert
executes. Legal ahead-of-router work therefore comprises exactly:

1. **Data movement of exact records** — prefetch a real 12.75 MiB bank
   record into host/VRAM residency ahead of demand. Always legal for any
   candidate set (worst case: wasted bandwidth + one LRU eviction).
2. **Work needing only already-committed state** — anything whose inputs
   exist now (e.g. shared-expert compute needs `h_L`, known pre-router).
3. **Exact early route knowledge** — hash layers 0–2 only
   (`ids = tid2eid[input_ids]`, proven source-grounded; weights still
   chain on `x @ W^T`).

Not legal: executing a predicted expert, consuming a route before it is
official, any lookahead into uncommitted tokens/layers.

## The binding constraint: idle-gap capacity

On a single-stream-saturated bank, prefetch during demand fills is
wall-neutral at best: makespan = bytes/BW regardless of order, and the
measured 3-lane pool was strictly *worse* (W3/W1 ≈ 1.21–1.24 per batch;
SERIALIZATION_VERDICT.md). Prefetch only wins inside genuinely idle
windows — time when the bank has no demand reads. Measured on the sealed
live profile (2×T4, `research/route-pipeline/evidence-live/per-layer.csv`
+ LIVE_PROFILE_RESULTS.md):

- Decode row wall 102.7 ms; fill 65.1 ms (63.4%).
- Idle-bank per row: **23.09 ms** inside native calls (conservative) to
  **37.6 ms** counting all non-fill time incl. orchestration/handoff.
- At the bank ceiling 0.29–0.37 GiB/s that admits **0.54–1.12 records/row**
  → 346–720 record-slots per response ≈ **4.3–8.9 GiB** of early-movable
  bytes vs the 15.24 GiB decode demand (1224 preads).

That is the entire prize pool for every ahead-of-router source combined.
An oracle using every idle byte saves 321–709 preads = **11.3–24.8 s**
(26–58% of decode fill wall, 13–29% of response fill wall). Sources with
worse-than-perfect knowledge get a fraction of that.

## Severity-ranked opportunities

Ordered by recoverable decode fill wall per response (16 forwards,
645 decode rows). Sim cells: gap ∈ {23.0, 37.6} ms × BW ∈ {0.29, 0.33,
0.37} GiB/s; full matrix in `results/gap_prefetch_sweep.csv`.

### 1. `oracle_gap` — BOUND ONLY, not legal — the ceiling itself

Perfect knowledge of every future baseline host-miss, earliest-deadline
first, filling every idle gap. **−11.3 to −24.8 s** decode fill
(42.84 → 18.0–31.6 s). This is the maximum any ahead-of-router mechanism
can recover on this bank; it requires future route knowledge, which does
not exist legally. Verdict: **N/A (bound)** — its role is to cap every
item below.

### 2. Hash layers 0–2 exact staging — LEGAL, EXACT — 0.54–1.60 s

`tid2eid[input_ids]` ids are exactly known at token start
(OFFICIAL_LOOKAHEAD.md, HASH_EARLY_STAGING.md — 3 layers × top-6 = ≤18
records = ≤229.5 MiB/token submittable early). Sim emits them from
same-token gaps (after rows (t,0)/(t,1)) and the token-boundary gap
(after (t,42), where sample(t) makes t+1's ids available).

Result: **14–42 preads saved = 0.54–1.60 s/response** (~1.3–3.7% of
decode fill; ~36–107 ms/token). Coverage is ~7–21% of the 196 decode
hash misses — the ids are exact but the *window* is tiny: L0 gets ≤1 gap
before its own demand, L1 ≤2, L2 ≤3; each record needs 34–44 ms vs a
23–37.6 ms gap. 199–250 issued prefetches abandon partial (wasted idle
bytes only — never a demand delay, never wrong data).

Legality: cleanest item in the memo — exact ids, immutable records,
demand-side unchanged; weights still chain (consume stays gated).
Verdict: **DEFER** — real and exact but ~1–4% of fill wall; it pays only
as a passenger on an idle-gap prefetch engine, which itself is DEFERRED
(§10). Revisit together if the bank moves (§Sensitivity).

### 3. Prev-token speculative hints — LEGAL (hint-only) — 0.25–0.74 s

Candidate set = previous forward's same-layer demand (decode: token t−1),
ranked by causal frequency; `prevtok_freq_backfill` pads with the
layer's hot records. Legal by the exactness contract (hints only;
executing a hinted expert would be illegal — the sim never does).

Result: **8–13 preads (0.25–0.42 s)**; backfill **16–23 (0.51–0.74 s)**.
The kill number: prev-token same-layer recall is 0.360 on *demand* but
**0.0054 on the miss stream** — the records that miss are exactly the
ones the previous token did not use, so a recency predictor aims its
~1-record gap budget at the wrong targets. Pollution is priced, not
hidden: backfill adds +44–73 host evictions on cuda0 and 4.8k–8.6k
abandoned partial fills (idle bytes burned, no correctness cost).

Consistent with the sealed generic-predictor rejection (recall@12 0.503;
Edge0 shows a *trained per-layer* predictor could be much stronger —
that is a Phase-5/serving artifact, not this bank's fix). Verdict:
**NO-GO** on this workload+bank. A trained predictor would have to beat
the miss-stream's intrinsic novelty, not just recall demand.

### 4. Shared expert — LEGAL — but zero bank bytes

`shared(h_L)` needs only the layer input — known pre-router
(SHARED_EXPERT_OVERLAP.md; currently serialized after routed combine by
code order, not dependence). Can it be "resident/prefetched
unconditionally"? It already is, in effect: shared weights are dense
F8_E4M3 checkpoint tensors (`layers.<L>.ffn.shared_experts.w1|w2|w3`,
~1.08 GiB across 43 layers ≈ 25 MiB/layer) served from weight mmap/page
cache — **not** expert-bank records. There is no 12.75 MiB pread to move
early. The legal lever is *compute overlap* (run shared during routed
staging): worth ≤0.31 s/response on the measured device-serial shared
time, unprofiled host-side. It neither consumes nor frees bank
bandwidth. Verdict: **DEFER** (GPU-side micro-overlap; orthogonal to
fill wall; belongs to the compute-overlap track, not storage).

### 5. Dense layers / non-expert tensors — N/A as a prefetch target

Attention/norm/router weights live in the weight mmap + persistent
device buffers (`router_weights_`, `d_router_weight_half_`) — zero
expert-bank bytes. Their compute is what *produces* the idle gaps
already priced above. Nothing to enumerate. Verdict: **NO-OP**.

### 6. MTP/DSpark buckets (mtp.{0,1,2} = 768 records) — no demand

The main forward never routes into them (PHASE3_FULL_EXPERT_STORE.md
§1.1; the sealed 688-record journal has exactly 43 layers × 16 forwards,
zero mtp rows). They matter only under speculative-decode serving, where
the draft head runs *after* main-chain logits — so mtp routes chain on
the sampled token + final hidden state, i.e. the same R5 dependency, the
same lookahead-0 wall. Verdict: **NO-GO** (nothing to move in this
workload; no earlier legality in any mode).

### 7. Cross-layer score-layer pipelining — ILLEGAL — permanent

route(L+1) needs `combine(L)` — the residual stream is the dependency
(OFFICIAL_LOOKAHEAD.md: official lookahead = **0 layers** for all score
layers). Layer N's routing becomes known only after layer N−1's combine;
layer N+1's expert need cannot exist before layer N's output. No
reordering, threading, or dual-GPU staging crosses it (DUAL_GPU_PIPELINE:
decode has no official A↔B overlap either). Verdict: **NO-GO** —
structural, not a missing mechanism.

### 8. Next-token anything — ILLEGAL — permanent

`input_ids[t+1] = sample(logits(t))` needs the full 43-layer chain
(LEGAL_OVERLAP.md §E). Even t+1's *hash* ids cannot be known until t's
last layer + lm-head + sample; the sim exploits only the post-sample
tail of t's final-layer gap (item 2). Verdict: **NO-GO**.

### 9. At-router batch submission (T9 pre-acquire) — legal but not "ahead"

Submitting all ≤6 reads at `route_known` instead of the serial
prepare→stage loop is the T2/T9 design. It is not ahead-of-router work
and was verdicted separately: +23% worse on this bank (saturated single
stream). This memo changes nothing there — an ahead-of-router source
feeds *idle gaps*, which the demand-time pre-acquire pool does not
address. Verdict: **NO-GO carried** (T2's verdict stands verbatim).

### 10. Idle-gap prefetch engine — the mechanism all sources need

None of the legal bytes move without a worker that admits prefetches
into the per-row idle windows (host-tier LRU insert on completion,
generation/lease discipline as in the existing tiers). Its ceiling is
item 1; its feedstock is items 2–3; together they yield ≲1.6 s/response
on this bank. Verdict: **DEFER** — do not build the engine for a
sub-2% payoff. Reopen iff (a) bank ceiling >~0.6–0.7 GiB/s (gap capacity
scales linearly: at 2.9 GiB/s a row's gap holds ~9 records — hash alone
could hide most of its 13 misses/token and the oracle bound approaches
the whole miss stream), or (b) a *trained* per-layer predictor with real
miss-stream precision arrives as a Phase-5 artifact.

### 11. Regime-C / cross-request prewarm — contract-deferred

The only source of *large* early knowledge (the next request's likely
working set) is the prewarmed regime, resolved as a deployment contract
and deferred to Phase 5 (research/phase2-regime-c @ 643d355). Not
in-request work; out of this memo's scope by contract. Verdict: **DEFER**
(standing).

## Does this reopen the serialization NO-GO?

**No.** Condition §7(b) asked whether a legal ahead-of-router candidate
source exists. One does (hash ids) — but the honest quantification shows
the source side was never the constraint: idle-gap capacity admits
≤~1.1 records/row, so even an omniscient source converts ≤29% of decode
fill, and the only legal sources realize ~1–4% of it (≤1.6 s of 86.3 s
response fill wall). The pre-acquire pool remains a demand-time
concurrency mechanism with a measured +23% regression; nothing here
makes it win. **SERIALIZATION_VERDICT stands; T9 stays CANCELED as a
wall optimization on this bank.** What this task adds: the
ahead-of-router channel is now *bounded from above* (oracle 24.8 s) and
*from the legal side* (hash 1.6 s), so the earlier-work lever is closed
with numbers rather than left open.

## Sensitivity

- **Bank speed**: gap capacity = gap_ms × BW / 12.75 MiB scales linearly.
  On a >0.6–0.7 GiB/s bank both the T9 repair (DESIGN.md §7a) and this
  channel reopen together; at the input-mount 2.9 GiB/s, a 37.6 ms gap
  holds ~9.3 records — hash staging alone covers ~most of its 196-miss
  stream, oracle bound ≈ the entire miss stream.
- **Longer prompts / prefill**: prefill hash demand (116 records ≈
  1.45 GiB) is exact-known at prompt arrival, but prefill is
  fill-saturated (62.2 s fill of 93.9 s wall) — early bytes only help in
  the small pre-prefill idle window; no in-request win.
- **Pollution**: priced everywhere (each completed prefetch is a real
  host-LRU insert). Wrong guesses cost idle bytes + ≤1 tail eviction;
  exact guesses can still displace live records — the sim charges both.

## Evidence base / reproduction

- Journal: `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/
  v50-evidence-20260829T195940Z/routed_experts.jsonl`, sha256
  `665aac3e…ae1`, 688 records (16 forwards × 43 layers, topk 6).
- Sim: `tools/phase2_legal_prefetch_eval.py` → `results/
  legal_prefetch_summary.json`, `gap_prefetch_sweep.csv`,
  `miss_stream.csv`. Validation vs sealed counters: single-level host
  misses 1391/1091 vs sealed 1390/1091; device misses 2284/2161 vs
  2285/2159; two-level total 2453 == p2c sim 2453 (±1 boundary noise is
  pre-existing convention).
- Live anchors: LIVE_PROFILE_RESULTS.md (decode 66.233 s, fill 41.99 s,
  645 rows; sim decode fill 42.84 s within ~2%), STORAGE_VERDICT.md
  (0.29–0.37 GiB/s ceiling, 96% in-batch busy), W1 serial batch table.
- Legality proofs: OFFICIAL_LOOKAHEAD.md, LEGAL_OVERLAP.md,
  HASH_EARLY_STAGING.md, DUAL_GPU_PIPELINE.md, PHASE3_FULL_EXPERT_STORE
  §1.1; engine chain engine.cpp:2794 early-return → :3153 stage →
  host_expert_tier.cpp:247 unlocked materialize → expert_store.cpp:631
  pread (all verified at 56dad3c).

## Limitations

- Bank is the sealed 2,364-record trace bank (11,776-record universe is
  Phase 3's). On the full universe, per-token misses rise (colder bank)
  but gap capacity and lookahead-0 legality are structural — the
  *fractions* move, the ordering of verdicts does not.
- The token-boundary gap charge is slightly generous for L0 (ids land
  partway through the post-L42 window); L1/L2 targets are unaffected in
  kind. Worst case this trims a few records off item 2's best cell.
- Prefetch targets host residency (682-slot LRU/GPU); a device-targeted
  variant changes placement, not the bank-read bottleneck.
- CPU-side "earlier work" (pointer tables, combine prep — LEGAL_OVERLAP
  C.3) is ms-scale and excluded; it is not bank work.
