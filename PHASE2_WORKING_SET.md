# PHASE2_WORKING_SET.md — the universal exact RAM/VRAM working-set policy study (v4, corrected)

**RETRACTION of b7b9c7f headline.** The initial study allowed count-based
pin policies to score HIT on the first touch of a pinned record — including
during the prefill whose statistics defined the pin set. That credited an
uncharged PREWARM inside cold-start comparisons and produced the impossible
result "static pin 91.4 % beats Belady 53.6 %". The claim is retracted; the
bug, its exact magnitude, and the regime contracts that prevent recurrence
are documented in `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`. All numbers below
come from the corrected simulator `tools/phase2_ws_policy_sim_v4.py`
(regime-labeled rows; sealed anchors still reproduced ±1..2).

**Mission scope.** Select the best UNIVERSAL exact working-set policy for
dee's host (and VRAM) expert tiers, OFFLINE only, from dee's own real
DeepSeek-V4-Flash route data. Phase 1 is CLOSED (`ca8abd0`): the bank reads
at 0.29–0.37 GiB/s regardless of lanes/patterns, pread ≈ 100 % of service,
QD6 keeps the disk 96 % busy, and the cause is FEED-side. Everything here is
derived from sealed evidence; no GPU run, no implementation, no merge.

---

## 1. Data provenance and validation (fail-closed against wishful replay)

- Access stream: **v50 canonical route journal**
  `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/v50-evidence-20260829T195940Z/routed_experts.jsonl`
  (sha256 `665aac3e…`, re-verified by the v4 simulator at run time). The
  v60 `dee4-metadata.json` (`dee4-v3-trace`) embeds this journal by hash
  (`trace_journal_sha256`) and declares `num_layers 43`,
  `total_experts 2364`, `total_bytes 31,605,129,216` — the trace store
  materializes one 12.75 MiB record per unique (layer, expert). (The
  768-record `dee4-v2` metadata inside the v50 directory is a stale
  safetensors-serving artifact; the serving run used `dee4_trace`.)
- Engine truth applied (read from `dee.cpp/src/engine.cpp`, not assumed):
  per `(forward_step, layer)` call the engine **deduplicates** the 7×6
  (prefill) or 1×6 (decode) route IDs and stages **unique experts in
  ascending expert-id order**. Stream = 5,099 requests.
- Simulator validated against two sealed independent runs
  (`results/validation_v4.json`):

| Anchor | Tier / config | Sealed | Simulated | Δ |
|---|---|---|---|---|
| fill-live host_pack cuda0 | host LRU, 682 slots (8.5 GiB) | hits 1,223 / miss 1,390 | 1,222 / 1,391 | ±1 |
| fill-live host_pack cuda1 | host LRU, 682 slots | hits 1,395 / miss 1,091 | **1,395 / 1,091** | 0 |
| v60 engine_stats cuda0 | VRAM priority-LRU, 281 slots (3.5 GiB) | 328 hits / 2,285 cold / 2,004 ev | 329 / 2,284 / 2,003 | ±1 |
| v60 engine_stats cuda1 | VRAM priority-LRU, 281 slots | 327 / 2,159 / 1,878 | 325 / 2,161 / 1,880 | ≤2 |

The VRAM anchor pins the production eviction score:
`score = last_used + priority · 2²⁰`, `priority = (batch_len − index)`
assigned in ascending-id batch order and **refreshed on hit**
(`vram_cache.h:197-206` PRIORITY_WEIGHT; engine passes `K−k`). Plain LRU
does NOT reproduce v60; priority-LRU does. Full analysis:
`VRAM_PRIORITY_AUDIT.md`.

**Anti-contamination rules honored:** no Qwen/moe-l2 locality imported; no
predictor counts as hits; no per-prompt hardcoding (policies use only
mechanical statistics of the stream).

## 2. Workload anatomy (from the journal only)

| Property | Value |
|---|---|
| Layers × experts | 43 × 256 routed (+1 shared, not routed through this tier) |
| Unique records touched | 2,364 of 11,008 slots (21.5 %) |
| Working set | 2,364 × 12.75 MiB = **29.43 GiB** |
| Requests (engine-dedup) | 5,099 = 2,364 first-touch + 2,735 repeats |
| Activations | 5,676 (1,806 prefill + 3,870 decode = 258/token) |
| Prefill (step 0) unique records | **1,229** (each requested exactly once in-pass) |
| Distinct records after 4 forwards | 1,465; counts: 26×4, 97×3, 266×2, 1,076×1 |
| Per-layer unique-expert span | 36–99 (median 53) |
| Decode consecutive-token slot overlap | **36.0 %** mean (range 16–55 % per token) |
| Popularity tail | mild power law: top-5 % records = 23.8 % activations, top-50 % = 79.2 % |
| Cross-GPU split | cuda0 layers 0–21: 1,247 records/16.7 GiB; cuda1 layers 22–42: 916/12.2 GiB (measured v8 split) |

The anatomy explains the corrected warmup result before simulating: **on
this architecture a single forward pass visits each layer once**, so
within-pass "frequency" carries no signal (every count = 1, tie-broken by
encounter order); after 4 forwards only 389 of 1,465 distinct records have
count ≥ 2. Frequency evidence on a 16-token window is thin by construction.

## 3. Reuse structure — why the answer is what it is

Exact stack-distance distribution over the 5,099-request stream
(`results/reuse_distance.json`; hit under LRU ⟺ distance < slots):

| Distance bucket | Repeats | Share |
|---|---|---|
| 0–199 | 0 | 0 % |
| 200–399 | 1,349 | 49.3 % |
| 400–699 | 771 | 28.2 % |
| 700–999 | 280 | 10.2 % |
| ≥ 1,000 | 335 | 12.2 % |

Minimum repeat distance = **213**. Implications:

- **The recency-family ceiling is 53.64 %** (all 2,735 repeats at distance
  ≥ 213), and offline Belady/MIN reaches it at *every* budget ≥ 8 GiB —
  MIN always reserves the soon-to-repeat records. Plain LRU reaches
  40.3 % (8 GiB) → 50.8 % (16) → 53.1 % (24) → 53.64 % (32, saturated).
- **Frequency-only value requires a PREWARM contract.** A top-N set that
  is already resident at t=0 catches far-distance repeats (60.5 % at
  8 GiB, 91.4 % at 24, 100 % at 29.43 GiB) — but that is regime C, not
  achievable from a cold start, and MIN given the same prewarm ties or
  beats the static pin at every budget (§5).
- The 36 % consecutive-token overlap is smaller than one token's footprint
  (258 activations ≈ 3.2 GiB), so router-ahead prefetch remains an
  overlap/scheduling lever (Phase-1's "earlier submission"), not a
  residency-policy lever.

## 4. Policies evaluated (mechanically defined, regime-labeled)

All on the identical validated stream, per scope (`full` pooled host,
`gpu0`/`gpu1` VRAM) and budget (`results/sim_rows_v4.csv`):

- **Regime A (cold):** lru; engine_priority_lru (production VRAM score,
  artifact documentation); arc; lfu; freq_x_recency; layer_lru;
  cost_aware (≡ lru, confirmed); **belady_cold = strict upper bound**.
- **Regime B (warmup, causal):** freq_lru_warmup (pin from step-0 prefill
  counts) and freq_lru_warmup_s4 (counts from first 4 forwards); pins
  applied at a boundary after the evidence exists; repopulation of
  non-resident pins **charged**; pins removed from the dynamic pool.
- **Regime C (prewarmed, labeled):** static_freq_prewarm (top-N resident
  at t=0, rejects otherwise — *not fetch-on-fault*);
  static_layer_freq_prewarm; freq_lru_prewarm (pf ∈ {0.25, 0.5, 0.75,
  0.9, 0.99} protected pin + LRU remainder); lru_prewarm_topN (control:
  LRU given the same top-N prewarm); **belady_same_state_prewarm** (MIN
  given the same top-slots prewarm) and **belady_pin_same_state(pf)**
  (protected pin + MIN dynamics — contract-matched bound for
  freq_lru_prewarm(pf)).

ARC applicability: mechanically well-defined (uniform slots, single pool);
adapts toward recency; recency is the wrong signal here (31.3 % vs 40.3 %
at 8 GiB). No line in the recommendation.

## 5. Corrected headline results (full pooled host tier; VRAM §7)

**Regime A (cold) and B (causal warmup) — hit rate %:**

| Policy | 8 | 12 | 16 | 20 | 24 | 32 GiB |
|---|---|---|---|---|---|---|
| lru (today) | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 |
| engine_priority_lru | 13.92 | 23.02 | 34.09 | 43.68 | 50.91 | 53.64 |
| arc | 31.30 | 42.91 | 50.44 | 52.23 | 53.09 | 53.64 |
| lfu | 38.83 | 46.56 | 50.83 | 52.32 | 53.09 | 53.64 |
| freq_x_recency | 40.52 | 46.70 | 50.85 | 52.32 | 53.09 | 53.64 |
| layer_lru | 37.46 | 46.13 | 50.23 | 51.79 | 52.54 | 53.40 |
| freq_lru_warmup (B, s0) | 30.93 | 40.91 | 49.36 | 52.07 | 53.07 | 53.64 |
| freq_lru_warmup_s4 (B) | 36.79 | 46.50 | 50.44 | 52.48 | 53.42 | 53.64 |
| **belady_cold (bound)** | **53.64** | **53.64** | **53.64** | **53.64** | **53.64** | **53.64** |

**Regime C (prewarmed; labeled; NOT comparable to the table above):**

| Policy | 8 | 12 | 16 | 24 | 32 GiB |
|---|---|---|---|---|---|
| static_freq_prewarm (top-slots resident) | 60.48 | 72.52 | 78.84 | 91.43 | 100 |
| static_layer_freq_prewarm | 58.58 | 70.90 | 78.03 | 90.51 | 97.27 |
| freq_lru_prewarm pf0.5 | 50.48 | 59.15 | 65.62 | 72.52 | 78.84 |
| freq_lru_prewarm pf0.99 | 60.21 | 72.33 | 78.58 | 91.04 | 100 |
| lru_prewarm_topN (control) | 43.36 | 53.30 | 66.31 | 88.94 | 100 |
| belady_same_state_prewarm | 56.83 | 59.11 | 70.68 | 91.14 | 100 |

Readings (corrected):

- **F1 (VRAM artifact):** unchanged and confirmed — the production score
  suppresses VRAM hits 2.1× (cuda0) / 3.2× (cuda1) at the sealed budget.
  Repair = evict by `last_used` only. See `VRAM_PRIORITY_AUDIT.md`.
- **F2 (recency bound):** Belady ties or beats every cold-start policy at
  every budget; nothing beats MIN from an equal initial state. The
  reachable cold-start optimum is ≈ the LRU curve itself.
- **F3 (warmup pin does not pay):** the honest online pin variant
  *underperforms plain LRU* at ≤ 16 GiB (49.36 vs 50.83 % at 16 GiB) and
  ties beyond. Its prefill-derived pin set has no frequency signal
  (§2), and its pin region displaces LRU slots that were earning recency
  hits. The s4 variant (first real frequency signal) recovers to ≈ LRU
  but never beats it on this window.
- **F4 (prewarm value is real but is a CONTRACT):** with a labeled
  prewarm, static top-N placement adds +20.2 pp over cold LRU at 16 GiB
  and +38.3 pp at 24 GiB — but MIN given the same prewarm ties or beats
  it everywhere (91.14 vs 91.43 % at 24 GiB is a tie within tail
  rounding). The prewarm *size* is the lever, not the static-vs-dynamic
  choice.
- **F5 (pin+LRU-remainder pathology):** freq_lru_prewarm at small budgets
  is worse than its own prewarm: marginal slope −8,186 slow-MiB per added
  GiB at 8 GiB (the LRU remainder evicts prewarmed records to cache
  never-repeating ones; the protected fraction is too small to matter).
  Dominated by pure pinning at every budget and by its own bound
  (belady_pin_same_state(pf)) pointwise — dominance check PASSES.

## 6. Capacity knees (host tier)

- **Cold-start (causal) knee: ≈ 16 GiB pooled.** LRU marginal value:
  1,020 slow-MiB per added GiB (8→12), 688 (12→16), 242 (16→20), 124
  (20→24), 45 (24→32), 0 beyond 32. Beyond 16 GiB an added GiB buys less
  than ~0.8 s of bank time per response (at 0.29 GiB/s); beyond 24 GiB it
  buys ~0.15 s.
- **Prewarm contract knee (regime C, labeled):** static top-N value rises
  ~1,020–2,066 slow-MiB/GiB all the way to ~29.43 GiB (100 % coverage);
  the 80 % coverage point is 15.30 GiB, 90 % is 22.37 GiB. This knee is
  about *how much the deployment prewarms*, not about an online policy.
- Above ~24 GiB, for every causal policy, marginal hit rate < 1 pp per
  2 GiB — the working set is nearly binary: hold the frequent core or
  pay for the tail one record at a time.

## 7. VRAM tier (per GPU, same validated stream)

| Budget | 1 GiB | 2 | 3 | 3.5 (v60) | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| cuda0 priority (production) | 4.1 | 8.8 | 11.2 | 12.6 | 13.7 | 21.8 | 31.4 |
| cuda0 plain LRU | 0.0 | 19.1 | 25.1 | 26.0 | 29.9 | 41.5 | 45.6 |
| cuda1 priority | 5.7 | 10.1 | 12.4 | 13.1 | 14.1 | 23.9 | 37.0 |
| cuda1 plain LRU | 0.0 | 32.9 | 40.2 | 42.5 | 46.4 | 51.6 | 55.7 |

The production score loses at every budget ≥ 2 GiB on both GPUs (2.07× /
3.25× at the sealed 3.5 GiB); the single exception is 1 GiB (74 slots ≪
one token's footprint) where stale protection is the only source of hits
— irrelevant for any real budget. Full analysis and the H2D/wall
accounting: `VRAM_PRIORITY_AUDIT.md`.

## 8. Answers to the mandated questions (corrected)

1. **Does increasing RAM meaningfully reduce slow-tier traffic for this
   DSV4 workload?** For any causal (cold-start) policy: only up to
   ~16 GiB pooled; the LRU-family traffic floor is 1,883.8 MiB/token and
   is reached at 32 GiB — beyond that, more RAM is dead weight. A much
   bigger reduction (to 348 MiB/token at 24 GiB, 0 at 29.43 GiB) is
   available *only* under an explicitly prewarmed regime-C contract, at
   the price of a one-time ~29.43 GiB prewarm read and a residency
   assumption across requests — an architecture decision
   (dee-serve-shaped), not a cache-policy tweak.
2. **Knee of the RAM-capacity curve:** ≈ **16 GiB pooled** for causal
   operation (last budget step with > 500 slow-MiB/GiB marginal). The
   prewarm contract's coverage knee is 15.3 GiB (80 %) → 22.4 GiB (90 %).
3. **How close does the best realizable policy get to Belady?** Belady is
   now the bound it always was: the best causal policy (plain LRU) sits
   2.8 pp below it at 16 GiB (2,592 vs 2,735 hits; 114 MiB/token) and
   closes to 0 by 32 GiB. Under a labeled prewarm, MIN given the same
   state ties the static pin. No realizable policy beats any
   same-state MIN.
4. **Is cache-policy work worth implementing at all?** On the host tier:
   **NO** new policy is justified by this trace — LRU is within 2.8 pp of
   MIN at the operating point, the warmup pin does not pay, and ARC/LFU/
   F×R/layer-LRU/cost-aware are LRU-shaped. The one high-value change on
   the memory tiers is the **VRAM priority repair** (free, 2.1–3.2× VRAM
   hits, −14.5 GB H2D per response). The big host-tier lever is not a
   policy: it is either (a) a prewarm contract (architecture change,
   labeled regime C) or (b) the Phase-1 levers (fewer cold fills,
   earlier submission), incl. an Edge0-style prefetch hint (feasibility
   plan in PHASE2_RECOMMENDATION.md §6 — offline, not implemented).

## 9. Threats to validity

- n = 16 forwards, one prompt family: all ceilings are properties of this
  sealed window. The regime-B "pin does not pay" conclusion is
  *strengthened* for longer windows (more decode steps give frequency
  time to matter), but the exact gap is trace-specific.
- The regime-C ceiling assumes cross-request stability of the frequency
  ranks (top-1 % covers only 6.8 % of activations — the tail is wide);
  a deployment prewarm must re-validate coverage per workload class.
- The v60 validation fixes the VRAM score semantics; if a future build
  changes `PRIORITY_WEIGHT` or the refresh-on-hit path, re-run
  `tools/phase2_ws_policy_sim_v4.py` before trusting VRAM columns.
- Wall projections are feed-bandwidth bounds (bank-limited per Phase 1),
  not schedules; intra-step overlap is outside a cache simulator.

Artifacts (v4): `tools/phase2_ws_policy_sim_v4.py`,
`results/{sim_rows_v4.csv, sim_rows_v4_derived.csv, ram_slope_v4.csv,
vram_audit.json, layer_locality.json, reuse_distance.json,
validation_v4.json}`, `PHASE2_CAPACITY_CURVES.csv` (regime-labeled),
`PHASE2_BYTE_FLOOR.md`, `PHASE2_POLICY_MATRIX.md`,
`PHASE2_RECOMMENDATION.md`, `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`,
`VRAM_PRIORITY_AUDIT.md`. (b7b9c7f artifacts `results/sim_rows*.csv` are
retained untouched for the audit trail.)
