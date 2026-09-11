# PHASE2_RECOMMENDATION.md — corrected final report (v4)

**Deliverable.** Exactly one host-tier decision, the VRAM verdict, the RAM
knee, the byte projection, the Belady gap, and the confidence label — in
the mandated format — followed by the mechanical reasons. Supersedes the
b7b9c7f recommendation (warmup-pin default), which rested on the
causality/initial-state bug documented in
`CAUSALITY_AND_INITIAL_STATE_AUDIT.md`. Everything is offline-derived from
the sealed v50 journal (sha256 `665aac3e…`) and validated against the
sealed `fill-live` (host) and `v60` (VRAM) counters (±1..2 counts,
`results/validation_v4.json`). No GPU run, no implementation, no merge.
Phase 1 remains CLOSED at `ca8abd0`.

---

## FINAL REPORT (mandated format)

```
HOST POLICY:            NO_CHANGE (keep plain LRU on the host expert tier)
VRAM POLICY:            fix  (VRAM_PRIORITY_FIX_RECOMMENDED — evict by last_used only)
RAM KNEE:               ~16 GiB pooled (causal/online operation)
PROJECTED SSD BYTES/TOKEN:
                        1,974.7 (today: LRU @ 17 GiB live envelope, sim;
                        prefill 979.4 + decode 995.3; 51.40 % hits)
                        -> unchanged by HOST POLICY: NO_CHANGE
                        -> minus up to ~420 MiB/decode-token host-fill upside
                           from the VRAM fix (strict upper bound; see §3)
                        -> 859.8 @16 GiB prewarm / 348.2 @24 GiB prewarm
                           ONLY under an explicitly labeled regime-C
                           prewarm contract (architecture decision, not
                           selected here)
BELADY GAP:             2.8 pp at 16 GiB (2,592 vs 2,735 hits; 114 MiB/token)
                        0 pp at >= 32 GiB (LRU saturates at the MIN value)
CONFIDENCE:             simulated (sealed-anchor-validated simulator, +/-1..2
                        counts; wall translation inferred, bounded in
                        VRAM_PRIORITY_AUDIT.md §5-6)
```

## 1. Why HOST POLICY is NO_CHANGE (mechanical reasons)

1. **The corrected study shows the cold-start optimum is the LRU curve
   itself.** Plain LRU sits 2.8 pp below offline Belady/MIN at the
   deployed operating point and ties it by 32 GiB. Every alternative
   causal policy measured *worse or equal*: arc −9 pp at 8 GiB, lfu −1.5,
   freq_x_recency ±0, layer_lru −2.9, cost_aware ≡ LRU (proven), and the
   warmup pins −1.5 pp (s0) / −0.4 pp (s4) at 16 GiB with repopulation
   exposure at small budgets. There is no causal policy left to select.
2. **The b7b9c7f "warmup-pin default" is retracted.** Its +10.9 pp gain at
   16 GiB was phantom prewarm credit (642 first-touch hits = exactly the
   pin size). Corrected, the online pin never beats LRU on this window:
   the prefill pass has no frequency signal (every in-pass count = 1),
   the window is too short for counts to accumulate, and the pin region
   displaces LRU slots that were earning recency hits.
3. **The big host-tier numbers are real but belong to a different
   contract.** A prewarmed top-N set (regime C, explicitly labeled) is
   worth +20.2 pp at 16 GiB and +38.3 pp at 24 GiB over cold LRU — and
   MIN given the same prewarm ties or beats the static pin, so even there
   the win is the prewarm *size*, not the placement policy. That is an
   architecture decision (cross-request residency — the dee-serve shape:
   persistent host tier across requests, sized to the useful working
   set), with a one-time ~prewarm-size read and a rank-stability
   assumption per workload class. It must never be scored against
   cold-start rows; this study labels it and hands the numbers to the
   architecture owner (Luna) without selecting it.
4. **RAM beyond ~16 GiB pooled is dead weight for causal operation**:
   marginal value 1,020 slow-MiB/GiB (8→12), 688 (12→16), 242 (16→20),
   124 (20→24), 45 (24→32), 0 beyond. If RAM is added, it should follow
   a prewarm-contract decision, not an eviction-policy change.

## 2. VRAM POLICY: fix — VRAM_PRIORITY_FIX_RECOMMENDED

- Production score `last_used + priority·2²⁰` (priority = staging order,
  refreshed on hit) leaks a stale protection: the first-staged expert of
  every batch becomes nearly un-evictable for the run.
- Sealed-counter-validated simulation (±2) at the sealed 281-slot budget:
  plain LRU captures **2.07× (cuda0) / 3.25× (cuda1)** the resident hits,
  saving **1,083 device loads and 14.48 GB of H2D per 16-token response**
  (298/622 MiB per decode token per GPU), with host-fill savings bounded
  at ≤ 6.6 GB NVMe per response (realistically less; avoided loads skew
  toward recent host hits).
- Robust across both GPUs and all decode steps ≥ 3 (steps 1–2 are a tie
  or slight priority win — honest fine print).
- Repair is one line (`PRIORITY_WEIGHT = 1`); eviction order cannot
  affect exactness. Validation contract: repaired engine should report
  ≈ 680/1,057 resident_hits at 281 slots on this trace. Confirmatory
  microbenchmark available (rerun v60 config, diff engine_stats + ITL).
- The oracle-path priority (`num_experts() − expert`) must never persist
  residency beyond the current forward if that path is exercised —
  prediction may prefetch, never protect.

## 3. PROJECTED SSD BYTES/TOKEN — accounting

- Baseline (today): 17 GiB host LRU + 3.5 GiB/GPU priority-LRU VRAM:
  **1,974.7 MiB/token** host-tier misses (sim at the exact live envelope:
  2,621 hits / 2,478 misses, 51.40 %; prefill 979.4 + decode 995.3
  MiB/token; live fill-live measured ~29 GiB/response ≈ consistent), of
  which the prefill 15.28 GiB is compulsory for any cold policy.
- HOST POLICY NO_CHANGE leaves this unchanged. The VRAM fix's host-fill
  upside is **bounded at ≤ 6.6 GB/response ≈ 419 MiB/decode-token
  pooled** (upper bound; realistically a fraction — see
  VRAM_PRIORITY_AUDIT.md §4).
- Regime-C prewarm contracts (labeled, not selected): 859.8 MiB/token at
  16 GiB, 348.2 at 24 GiB, 0 at 29.43 GiB — each requires cross-request
  residency plus a one-time prewarm read amortized outside the window.

## 4. BELADY GAP

- Cold start (regime A): best causal policy = LRU. Gap to
  `belady_cold`: **2.8 pp at 16 GiB** (114 MiB/token), 1.4 pp at 20,
  0.5 pp at 24, **0 at ≥ 32 GiB** (both saturate at 53.64 %).
- Regime C: no policy beats MIN given the same initial state (checked
  contract-matched at every budget and pin fraction).

## 5. What Phase 2 implementation must NOT do (guardrails, updated)

- Do not implement any new host placement policy on the basis of this
  study — including the previously recommended warmup pin.
- Do not mix prewarmed (regime C) rows into cold-start comparisons; the
  regime label + `initial_state_gib` columns are now part of every
  artifact and must stay there.
- Do not let the VRAM repair grow beyond the one-line weight change; if
  oracle priority is ever needed, pin within the current forward only.
- Do not import locality parameters from other models; every number here
  is DSV4's own, single-window, and re-validated by
  `tools/phase2_ws_policy_sim_v4.py` on every run.
- Do not exceed the host envelope on the strength of regime-C numbers:
  24 GiB pooled projects ≈ 29–30 GiB peak RSS on the 31.35 GiB Kaggle
  host — a memory-gate matter, not a policy matter.

## 6. Future research — Edge0-style prerouter (OFFLINE feasibility plan only, DO NOT IMPLEMENT)

Per the campaign mandate, and because after this correction the feed-side
ceiling stands (cold floor 1,883.8 MiB/token; VRAM fix bounded), the
remaining feed-side lever of the *prediction* class is an Edge0-style
trained prerouter used strictly as a PREFETCH HINT (never as residency
authority, never altering executed experts — exactness contract):

- Plan (offline, on existing sealed traces only): train a per-layer
  top-k predictor on route sequences from prior tokens; evaluate
  recall@6 / recall@12 / precision, useful vs wasted prefetched bytes,
  ready-before-demand fraction, cache pollution (extra evictions),
  queue pressure, and expected exposed-wait reduction via the same
  bank-bandwidth model used here; run a shadow-mode replay against the
  validated simulator before any live proposal.
- Readiness gate: only if (a) the corrected cache study is complete
  (it now is) and (b) feed-side latency remains the dominant projected
  limit after the VRAM fix (expected: yes, at 0.29–0.37 GiB/s).
- Note the mechanism difference: Edge0's trained router hides latency by
  prefetching *before* the engine asks; dee's earlier generic predictor
  failed as a *residency* heuristic. The hint must feed the existing
  prefetch queue, not the eviction score.

## 7. Reproducibility

- Corrected simulator: `tools/phase2_ws_policy_sim_v4.py` (single file,
  stdlib only; re-verifies the journal sha256 and the sealed anchors on
  every run; prints the in-regime dominance checks' inputs).
- Regime-labeled matrix: `results/sim_rows_v4.csv` +
  `sim_rows_v4_derived.csv` (SSD MiB/token + per-token/per-response
  walls at 0.29/0.33/0.37/3/5/7/12 GiB/s); slopes
  `results/ram_slope_v4.csv`; VRAM per-step audit
  `results/vram_audit.json`; per-layer locality
  `results/layer_locality.json`; anchors `results/validation_v4.json`.
- Deliverable view: `PHASE2_CAPACITY_CURVES.csv` (regime column, 576
  rows), `PHASE2_WORKING_SET.md`, `PHASE2_BYTE_FLOOR.md`,
  `PHASE2_POLICY_MATRIX.md`, `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`,
  `VRAM_PRIORITY_AUDIT.md`. b7b9c7f artifacts retained untouched for the
  audit trail.
