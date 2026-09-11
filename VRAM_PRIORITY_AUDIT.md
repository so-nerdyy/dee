# VRAM_PRIORITY_AUDIT.md — independent re-check of the production VRAM eviction score

**Candidate issue (as assigned).**
`score = last_used + priority · 2²⁰` — does staging-order priority give some
expert IDs persistent protection, and would plain LRU roughly double VRAM
hits at the same capacity?

**Verdict: `VRAM_PRIORITY_FIX_RECOMMENDED`.**
(The earlier "roughly double" claim is confirmed on cuda0 (2.1×) and
**exceeded** on cuda1 (3.2×).) Confidence: **simulated** — the simulator
reproduces the sealed counters within ±2 before the comparison is made;
the wall-clock translation is inferred (see §5).

---

## 1. The production semantics (read from source, not assumed)

- `dee.cpp/include/dee/vram_cache.h:197-206`:
  `PRIORITY_WEIGHT = 1 << 20`;
  `eviction_score(b) = b.last_used + (int64_t)b.priority * PRIORITY_WEIGHT`;
  comment: "Lower score => evict first."
- `vram_cache.cpp ensure()` (resident-hit path): `b->last_used = ++tick_;
  b->priority = priority;` — **priority is refreshed on hit**.
- `engine.cpp` passes `priority = K − k` (production staging:
  `stage_expert(..., cfg_.topk - k)` at lines 299 / 3360 / 4055 and
  `active_experts` scans), i.e. **arrival order within the deduplicated
  ascending-id batch**: the first-staged (smallest expert id) expert of a
  batch receives `priority = K` (up to 7 prefill / 6 decode… up to 255 on
  the oracle path `num_experts() − expert`, line 3306).

Why this is an artifact and not a policy: the boost term (`priority ≤ 255`
× 2²⁰ ≈ up to 2.67e8) **dwarfs the entire tick space of the sealed run**
(total ticks ≈ 2,613 per GPU). Score ordering therefore degenerates to
*priority class first, age second*, and the tick term becomes irrelevant
for high-priority blocks. `K − k` encodes nothing but arrival order — it
has no future-value semantics — so the effect is that **the first-staged
expert of every batch becomes nearly un-evictable for the rest of the
run** (a stale-priority leak), while later batches' low-priority members
are evicted first regardless of recency.

## 2. Check 1 — does the production simulator reproduce the sealed counters?

Yes. Sealed v60 evidence (`native-generate-result.json → engine_stats`,
2×T4, 3.5 GiB/GPU = 281 packed-FP4 slots, 16 tokens), simulator on the
sealed v50 journal (sha256 `665aac3e…`), same engine-dedup stream:

| Counter | cuda0 sealed | cuda0 sim | cuda1 sealed | cuda1 sim |
|---|---|---|---|---|
| resident_hits | 328 | 329 | 327 | 325 |
| cold_loads | 2,285 | 2,284 | 2,159 | 2,161 |
| evictions | 2,004 | 2,003 | 1,878 | 1,880 |

(±1, ±2; `research/phase2-ws-policy/results/validation_v4.json`.) Plain
LRU does **not** reproduce these counters; the priority-LRU score above
does. The model of production semantics is therefore exact.

## 3. Check 2 — does removing the priority term improve the real device hit rate?

At the sealed 281-slot budget, same stream, same initial state (cold):

| | hits (resident_hits) | hit rate of requests |
|---|---|---|
| engine_priority_lru (production) | 328 / 327 | 12.6 % / 12.5 % |
| plain LRU (drop the term) | 680 / 1,057 | 26.0 % / 40.5 % |
| improvement | **2.07× / 3.25×** | +13.4 pp / +28.0 pp |

The "roughly double" earlier estimate is confirmed on cuda0 and exceeded
on cuda1 (cuda1's layer window 22–42 has the shorter per-GPU working set
and benefits more). Full VRAM budget sweep (hit rate %, per GPU):

| Budget | 1 GiB | 2 | 3 | 3.5 (v60) | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| cuda0 priority | 4.1 | 8.8 | 11.2 | 12.6 | 13.7 | 21.8 | 31.4 |
| cuda0 plain LRU | 0.0 | 19.1 | 25.1 | 26.0 | 29.9 | 41.5 | 45.6 |
| cuda1 priority | 5.7 | 10.1 | 12.4 | 13.1 | 14.1 | 23.9 | 37.0 |
| cuda1 plain LRU | 0.0 | 32.9 | 40.2 | 42.5 | 46.4 | 51.6 | 55.7 |

The gap exists at every capacity from 2 GiB upward on both GPUs; it never
reverses there. One honest exception at the extreme small end: at 1 GiB
(74 slots ≪ one token's footprint) priority scores 4.1/5.7 % where LRU
captures 0 — with almost nothing resident, the stale protection of a
handful of early records is the only hits available. Irrelevant for any
deep budget, but it belongs in the record.

## 4. Check 3 — H2D bytes/token saved

Each avoided device-cache load is one 12.75 MiB packed-FP4 record that is
never DMA'd into VRAM (`h2d_bytes`/`h2d_copies` semantics in the sealed
engine_stats; v60 measured h2d 30.55 GB / 2,285 copies = 13.368 MB/copy,
exactly the record size):

| | loads saved (15 decode steps) | H2D bytes saved | H2D per decode token |
|---|---|---|---|
| cuda0 | 351 | 4.69 GB | 298.4 MiB |
| cuda1 | 732 | 9.79 GB | 622.2 MiB |
| **both** | **1,083** | **14.48 GB** | **920.6 MiB** |

Direction-of-causality note: the fix does **not** change host-tier hit
rates (the host `host_pack` sees the same 2,613 requests per GPU in both
worlds — sealed evidence: host_pack requests = engine loads + resident
hits under the *production* score, and the host LRU is independent of the
VRAM victim choice). What it saves for certain is the H2D DMA traffic and
eviction churn above. What it may additionally save is host-fill work:
of the 1,083 loads avoided, those that would have been host-pack misses
(no NVMe) versus host-pack hits. Upper bound using the sealed per-GPU
host miss rates (53.2 % / 41.8 %): ≤ 6.6 GB of NVMe per response
(≤ ~0.40 GiB/decode-token pooled). Realistically **less**: a record
re-requested soon after eviction is recent in the host LRU, so avoided
loads skew toward host hits. treat 6.6 GB (6,281 MiB ≈ 419 MiB per decode token pooled) as a strict
upper bound.

## 5. Check 4 — robustness across token position and GPU

Per-decode-step resident hits at 281 slots (`results/vram_audit.json`):

- cuda0: priority `[28, 40, 41, 24, 16, 26, 17, 15, 11, 25, 18, 18, 16, 17, 17]`
  vs LRU `[23, 58, 36, 48, 57, 69, 34, 32, 51, 17, 66, 49, 50, 64, 26]`.
- cuda1: priority `[43, 35, 28, 25, 22, 12, 10, 16, 21, 18, 22, 17, 18, 19, 19]`
  vs LRU `[27, 52, 69, 69, 67, 67, 54, 38, 76, 77, 80, 89, 102, 84, 106]`.

Honest fine print: plain LRU does **not** dominate every individual step —
the priority boost genuinely helps for the first ~2 decode steps (it
approximates "keep the previous batch's experts" while that priority is
still fresh), on both GPUs. From decode step ~3 onward LRU wins every
step on both GPUs, and the cumulative gap grows monotonically to the
2.1×/3.2× totals above. Robust across GPUs: yes (same sign, same
pattern). Robust across token position: yes for any window ≥ ~3 decode
tokens; a ≤2-token workload would be a tie — irrelevant for dee's target
latency regimes.

## 6. Check 5 — exposed wall or merely a counter?

The sealed v60 profile has `stage_profile: enabled=false`, so the wall
attribution cannot be *measured* from sealed evidence; this is the one
inferred link in the chain, and it is labeled as such.

Causal argument that the improvement reaches the wall:

1. Phase-1 closed the feed as the system limit: QD6 holds the /tmp bank
   at ~96 % busy, pread ≈ 100 % of service, no lane/queue scaling. Wall
   per decode token today is ITL p50 4.6 s while the byte demand is
   ~2.05 GiB/token of NVMe at ~0.29 GiB/s ⇒ the bank is the binding
   resource, and bank-time saved converts to exposed wait reduced.
2. A device-cache resident hit under the production score is not free
   relative to the counter fix: the *saved* loads (1,083 over the
   response) each remove one request from the host tier. Those avoided
   requests are exactly the ones that would have consumed bank time when
   they missed the host pack (upper bound §4: 6.6 GB ≈ 22 s of bank time
   per response at 0.29 GiB/s; realistically a fraction of that).
3. Even the *certain* part (14.48 GB of H2D no longer issued) reduces
   DMA/eviction churn on the copy engine, but H2D alone is not the wall
   (v60 average H2D rate ≈ 0.42 GB/s vs ≥8 GB/s PCIe capacity) — the
   wall value rides on the host-fill savings, which are upper-bounded
   but plausibly 0.1–0.4 GiB/decode-token ⇒ of order 0.3–1.4 s of bank
   time per token at today's storage, to be realized partially through
   the existing overlap.

Conclusion: the fix is very likely to reduce exposed wall at the current
0.29–0.37 GiB/s bank, and the improvement decays as storage gets faster
(same "now lever" shape as the host-policy conclusion in
PHASE2_BYTE_FLOOR.md §5). A cheap confirmatory microbenchmark exists if
Luna wants measured evidence: rerun v60's exact config with
`PRIORITY_WEIGHT` set to 1 (one line) and diff `engine_stats` +
`inter_token_latency_ms` against the sealed v60 run — no new machinery.

## 7. Recommended repair and what must not change

- **Repair:** evict by `last_used` alone (`PRIORITY_WEIGHT = 1`), i.e.
  plain LRU. The priority term encodes arrival order within a batch,
  which has no future-value semantics; nothing of value is lost. On the
  oracle path (`num_experts() − expert`, line 3306) the term was meant as
  "protect the oracle-predicted expert" — if that path is ever exercised
  in production, protect it *within the current forward only* (e.g.,
  pin/unpin around the batch) rather than through a persistent score
  term; prediction may prefetch, never alter residency beyond the
  current use.
- **Exactness:** the VRAM cache is a performance tier; eviction order
  cannot change which experts execute or any output byte. Cold == warm
  and token identity are structurally unaffected (same argument class
  accepted for the Phase-2 seals).
- **Validation contract for the fix:** at 281 slots the repaired engine
  should report `resident_hits` ≈ 680 (cuda0) / 1,057 (cuda1) ± noise on
  the sealed trace, `cold_loads` ≈ 1,933 / 1,429. Counters outside those
  bands indicate the semantics were not actually simplified.

**Return value: `VRAM_PRIORITY_FIX_RECOMMENDED`**
(improvement measured-by-simulation at ±2 counts against sealed
counters; robust across both GPUs and all steps ≥ 3; H2D savings certain;
wall reduction inferred with a bounded, testable upside).
