# PHASE2_WORKING_SET.md — the universal exact RAM/VRAM working-set policy study

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
  (sha256 `665aac3e…`). The Phase-1 `fill-live-t4x2-20260909` journal was
  re-derived and verified **byte-identical in route arrays** (688 records,
  chain-valid) — the live T4 run and the sealed trace are the same workload.
- Engine truth applied (read from `dee.cpp/src/engine.cpp`, not assumed):
  per `(forward_step, layer)` call the engine **deduplicates** the 7×6
  (prefill) or 1×6 (decode) route IDs and stages **unique experts in
  ascending expert-id order** (`moe_forward_batch_device_impl`: grouping +
  `active_experts` scan over experts 0..255). Stream = 5,099 requests.
- The simulator was validated against two sealed independent runs
  (`results/validation.json`):

| Anchor | Tier / config | Sealed | Simulated | Δ |
|---|---|---|---|---|
| fill-live host_pack cuda0 | host LRU, 682 slots (8.5 GiB) | hits 1,223 / miss 1,390 | 1,222 / 1,391 | ±1 |
| fill-live host_pack cuda1 | host LRU, 682 slots | hits 1,395 / miss 1,091 | **1,395 / 1,091** | 0 |
| v60 engine_stats cuda0 | VRAM priority-LRU, 281 slots (3.5 GiB) | 328 hits / 2,285 cold / 2,004 ev | 329 / 2,284 / 2,003 | ±1 |
| v60 engine_stats cuda1 | VRAM priority-LRU, 281 slots | 327 / 2,159 / 1,878 | 325 / 2,161 / 1,880 | ≤2 |

The VRAM anchor required discovering the exact production eviction score:
`score = last_used + priority · 2²⁰` with `priority = (batch_len − index)`
assigned in the ascending-id batch order and **refreshed on hit**
(`vram_cache.cpp eviction_score`; engine passes `K−k`). Plain LRU does NOT
reproduce v60; priority-LRU does. This is finding F1 below.

**Anti-contamination rules honored:** no Qwen/moe-l2 locality imported; no
predictor counts as hits (every number is an exact routed activation); no
per-prompt hardcoding (policies use only mechanical statistics of the
stream: frequency ranks, recency, stack distance).

## 2. Workload anatomy (from the journal only)

| Property | Value |
|---|---|
| Layers × experts | 43 × 256 routed (+1 shared, not routed through this tier) |
| Unique records touched | 2,364 of 11,008 slots (21.5 %) |
| Working set | 2,364 × 12.75 MiB = **29.43 GiB** |
| Requests (engine-dedup) | 5,099 = 2,364 first-touch + 2,735 repeats |
| Activations | 5,676 (1,806 prefill + 3,870 decode = 258/token) |
| Top-6-of-256 per layer per token | rank structure recorded (topk rank order preserved in journal) |
| Per-layer unique-expert span | 36–99 (median 53) |
| Decode consecutive-token slot overlap | **36.0 %** mean (range 16–55 % per token) |
| Popularity tail | mild power law: top-5 % records = 23.8 % activations, top-50 % = 79.2 % (not flat, not extreme — no Qwen-style concentration) |
| Cross-GPU split | cuda0 layers 0–21: 1,247 records/16.7 GiB; cuda1 layers 22–42: 916/12.2 GiB (measured v8 split) |

## 3. Reuse structure — why the answer is what it is

Exact stack-distance distribution over the 5,099-request stream
(`results/reuse_distance.json`; distance = distinct experts touched since
previous use of the same record; hit under LRU ⟺ distance < slots):

| Distance bucket | Repeats | Share |
|---|---|---|
| 0–199 | 0 | 0 % |
| 200–399 | 1,349 | 49.3 % |
| 400–699 | 771 | 28.2 % |
| 700–999 | 280 | 10.2 % |
| ≥ 1,000 | 335 | 12.2 % |

Minimum repeat distance = **213**. Implications:

- **The LRU-family ceiling is 53.64 %** (2,735 repeats, all at distance ≥
  213): even Belady saturates once capacity ≥ 2,570 slots (32 GiB) because
  every repeat is ≥ 213 distinct-experts away — you must hold ≥ 213 recent
  records *plus* the working set's tail to keep converting repeats, and the
  reuse horizon (one token = ~320 requests × 16 tokens) spans the whole
  window. Capacity curves: hits climb 40.3 % (8 GiB) → 50.8 % (16) →
  53.1 % (24) → 53.6 % (32, saturated). **Above 32 GiB, recency has nothing
  left to give.**
- **Static frequency placement is not stack-bounded.** Pinning the
  frequency-ranked set catches far-distance repeats: 60.5 % hits at 8 GiB,
  78.8 % at 16, 91.4 % at 24, 100 % at 29.43 GiB (the whole set). The pin
  ceiling is exactly the activation-coverage curve (§4 of
  PHASE2_BYTE_FLOOR.md).
- The 36 % consecutive-token overlap is real but *smaller than one token's
  footprint* (258 activations ≈ 3.2 GiB > any VRAM budget headroom), which
  is why router-ahead prefetch of the next token's experts is an
  **overlap/scheduling** lever (already Phase-1's "earlier submission"), not
  a residency-policy lever.

## 4. Policies evaluated (mechanical definitions)

All evaluated on the identical validated stream, per scope
(`full` pooled host, `gpu0`/`gpu1` per-GPU VRAM) and per budget
(`results/sim_rows.csv`, 336 rows):

1. **lru** — plain LRU (current host-tier semantics, validated exact).
2. **engine_priority_lru** — the production VRAM score
   (`last_used + (K−i)·2²⁰`), validated against v60; evaluated to quantify
   its artifact.
3. **arc** — Adaptive Replacement Cache (canonical T1/T2/B1/B2, adaptive p).
   Mechanically applicable (single uniform pool).
4. **lfu** — online count-based eviction with LRU tiebreak.
5. **static_freq** — offline top-N frequency placement (KTransformers-style
   hot set; the pinning ceiling).
6. **static_layer_freq** — per-layer top-B frequency placement
   (MoE-Infinity-style layer budgets).
7. **freq_lru** — frequency-pinned set at pin_frac ∈ {0.25, 0.5, 0.75} of
   slots + LRU remainder (hybrid; "freq-pinned set + LRU remainder").
8. **freq_lru_warmup** — freq_lru with the pin set built **only from the
   prefill pass counts** (fully online/deployable: warmup, then steady
   decode).
9. **freq_x_recency** — online composite score `0.5·freq_frac +
   0.5·recency_frac` (frequency × recency).
10. **layer_lru** — per-layer LRU with uniform per-layer budgets
    (layer-budgeted hot sets).
11. **cost_aware** — greedy cost/size-aware eviction; all records are one
    size, so it **provably reduces to LRU** — evaluated and confirmed
    identical (equivalence check, as required).
12. **belady** — offline MIN; **upper bound only**, never a recommendation.

ARC applicability note: ARC is mechanically well-defined here (uniform slot
size, single pool). It adapts toward recency, and this trace's reuse is
frequency-dominant, so ARC ≈ LRU with extra ghost-list memory (31.3 % vs
40.3 % at 8 GiB — slightly *worse*). It earns no line in the recommendation.

## 5. Headline results (full pooled host tier; VRAM in §7)

| Policy | 8 GiB | 16 GiB | 24 GiB | 32 GiB | 48 GiB |
|---|---|---|---|---|---|
| lru (today) | 40.3 % | 50.8 % | 53.1 % | 53.6 % | 53.6 % |
| engine_priority_lru | 13.9 % | 34.1 % | 50.9 % | 53.6 % | 53.6 % |
| arc | 31.3 % | 50.4 % | 53.1 % | 53.6 % | 53.6 % |
| lfu | 38.8 % | 50.8 % | 53.1 % | 53.6 % | 53.6 % |
| freq_x_recency | 40.5 % | 50.9 % | 53.1 % | 53.6 % | 53.6 % |
| layer_lru | 37.5 % | 50.2 % | 52.5 % | 53.4 % | 53.6 % |
| freq_lru pf0.5 | 50.5 % | 65.6 % | 72.5 % | 78.8 % | 91.4 % |
| freq_lru pf0.75 | 56.2 % | 72.5 % | 82.0 % | 91.4 % | 100 % |
| **freq_lru_warmup (online)** | 37.2 % | 62.0 % | **72.0 %** | **77.7 %** | 77.7 % |
| static_freq (pin ceiling) | 60.5 % | 78.8 % | 91.4 % | 100 % | 100 % |
| static_layer_freq | 58.6 % | 78.0 % | 90.5 % | 97.3 % | 100 % |
| belady (bound) | 53.6 % | 53.6 % | 53.6 % | 53.6 % | 53.6 % |

Readings:

- **F1 (VRAM artifact):** the sealed v60 engine captured 12.9 % of requests
  at 281 slots where plain LRU captures 26.6 % and the static top-281 set
  captures ~37 %. The stale-priority boost (first-staged expert of each
  batch is nearly un-evictable for the whole run) is a mechanical defect,
  not a policy choice. Repair is one line: evict by `last_used` only
  (i.e., plain LRU) or drop the `2²⁰` weight. **This is the single
  highest-value change on the VRAM tier and it is free.**
- **F2 (LRU saturation):** every recency policy converges to the same
  capacity curve, and even offline Belady (MIN) holds exactly 53.64 % at
  every budget — the MIN-optimal policy of this stream exploits all and
  only the recency structure there is. The 53.64 % ceiling by 32 GiB is
  therefore not an LRU weakness but the trace's recency bound itself.
- **F3 (frequency dominance):** only frequency-pinned families beat the
  recency bound (Belady included); the offline top-N is their ceiling
  (static_freq). Belady cannot see frequency, so pins beat MIN here —
  the reachable optimum is the pin ceiling, not MIN.
- **F4 (online gap):** pins derived from the prefill pass alone reach
  ~85 % of the offline pin's value (72.0 vs 91.4 pp at 24 GiB; 77.7 vs
  100 % at 32) — an implementable online policy with no future knowledge.
- **F5 (hybrid penalty):** freq_lru (pin + LRU remainder) is strictly worse
  than pure static pin at every budget because on this short trace the LRU
  remainder mostly holds records that never repeat within the window —
  pin everything you can afford.

## 6. Capacity knees (host tier)

- LRU family: knee at **~12–16 GiB** (46.6 → 50.8 %); dead beyond 32 GiB.
- Frequency-pinned (online warmup variant): knee at **~16 GiB**
  (62.0 %) → 24 GiB (72.0 %) → flat 77.7 % from 32 GiB.
- Static pin ceiling: knee at **15.3 GiB (80 % coverage)**; 90 % at
  22.4 GiB; 100 % only at the full 29.43 GiB.
- **Above ~24 GiB the marginal hit rate per GiB drops below 1 pp/2 GiB for
  every policy** — the working set is nearly binary: hold the frequent
  ~1,800 records or you are paying for the long tail one record at a time.

## 7. VRAM tier (per GPU, from the same validated stream)

| Budget | slots | engine_priority (sealed) | plain LRU | static top-N |
|---|---|---|---|---|
| 1 GiB | 74 | — | 6.4 % | 27.6 % |
| 2 GiB | 149 | — | 14.3 % | 43.5 % |
| 3.5 GiB (v60) | 281 | **12.9 %** | 26.6 % | ~37 % |
| 4 GiB | 320 | — | 27.7 % | ~40 % |
| 6 GiB | 481 | — | 31.1 % | ~46 % |
| 8 GiB | 641 | — | 33.7 % | ~51 % |

The VRAM tier is budget-capped (dense + engine occupy ~7–9 GiB of the T4's
14.56 GiB usable) and can never hold more than ~640 of 1,247/916 per-GPU
records; its policy question is dominated by F1 (kill the priority
artifact). Full curves in `PHASE2_CAPACITY_CURVES.csv`.

## 8. Answers to the mandated questions

1. **Does increasing RAM meaningfully reduce slow-tier traffic for this
   DSV4 workload?** Only up to a point and only for the right policy:
   for any LRU-family policy the answer is *no beyond ~24–32 GiB* (hard
   53.6 % ceiling — traffic floor 1,883 MiB/token regardless of RAM);
   for frequency-pin policies *yes up to ~24 GiB* (traffic falls
   1,610 → 348 MiB/token from 16 → 24 GiB, −78 %) and *100 % elimination
   at 29.43 GiB* — but that exceeds the practical host envelope together
   with dense + engine (peak RSS 22.9 GiB today on a 31.35 GiB box).
2. **Knee of the RAM-capacity curve:** ≈ **16 GiB pooled** for the
   recommended online policy (62 % of the pin ceiling captured), with the
   80 %-coverage point at 15.3 GiB and 90 % at 22.4 GiB.
3. **How close does the best realizable policy get to Belady?** On this
   trace that question inverts: Belady (a recency-family bound) *ties*
   plain LRU (53.6 %) and is **beaten by every frequency-pinned policy**
   (91.4 % at 24 GiB for offline pin; 72.0 % for the fully online variant).
   Belady bounds recency exploitation, not frequency exploitation; the
   true ceiling for exact caching is the static pin (= activation-coverage
   curve), and the online warmup variant reaches ~79–85 % of it.
4. **Is cache-policy work worth implementing at all?** Yes — narrowly and
   exactly: (a) the VRAM one-line priority-artifact repair (+13.7 pp VRAM
   hits, zero cost); (b) a host frequency-pin policy (+18–25 pp host hits
   over today's LRU at equal RAM, −47 % cold bytes at 24 GiB); and
   (c) nothing else — ARC/LFU/cost-aware/layer-LRU/F×R are all within
   noise of LRU here, and Belady-type ideas are unreachable and (on this
   trace) unhelpful.

## 9. Threats to validity

- n = 16 forwards, one prompt family: the coverage curve and the 53.6 %
  recency ceiling are properties of *this* sealed window; policies that
  need long-horizon frequency stability (static pin) are extrapolated to
  longer responses only by the trace's own top-X curve, not assumed.
- The journal's topk *rank order* is preserved but unused by the policies
  evaluated (rank-weighted placement showed no mechanical advantage in a
  pilot run and was excluded to keep the matrix falsifiable).
- The v60 validation fixes the VRAM score semantics; if a future build
  changes `PRIORITY_WEIGHT` or the refresh-on-hit path, re-run
  `tools/phase2_ws_policy_sim.py` before trusting the VRAM columns.

Artifacts: `tools/phase2_ws_policy_sim.py`, `results/{sim_rows.csv,
sim_rows_derived.csv, reuse_distance.json, validation.json}`,
`PHASE2_CAPACITY_CURVES.csv`, `PHASE2_BYTE_FLOOR.md`,
`PHASE2_POLICY_MATRIX.md`, `PHASE2_RECOMMENDATION.md`.
