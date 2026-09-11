# PHASE2_POLICY_MATRIX.md — corrected, regime-labeled policy × budget matrix (v4)

All numbers from `tools/phase2_ws_policy_sim_v4.py` on the validated stream
(5,099 engine requests; anchors `results/validation_v4.json`). Hit rates
are host-tier pooled (scope `full`) unless labeled VRAM. What changed vs
b7b9c7f and why: `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`. Regimes:
**A** = cold start (strict bound: `belady_cold`), **B** = causal warmup
(repopulation charged), **C** = explicitly prewarmed (labeled initial
state; bound = MIN given the same state). Full machine-readable matrix:
`research/phase2-ws-policy/results/sim_rows_v4.csv` (+ `_derived` with
SSD bytes/token and per-token/per-response wall columns at
0.29/0.33/0.37/3/5/7/12 GiB/s); deliverable view:
`PHASE2_CAPACITY_CURVES.csv` (regime column; 576 rows).

## 1. Master matrix — full pooled host tier

### 1.1 Regime A (cold start) and B (causal warmup) — hit rate (%)

| Policy | 8 | 12 | 16 | 20 | 24 | 32 | 48+ GiB |
|---|---|---|---|---|---|---|---|
| lru (today) | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 | 53.64 |
| engine_priority_lru | 13.92 | 23.02 | 34.09 | 43.68 | 50.91 | 53.64 | 53.64 |
| arc | 31.30 | 42.91 | 50.44 | 52.23 | 53.09 | 53.64 | 53.64 |
| lfu | 38.83 | 46.56 | 50.83 | 52.32 | 53.10 | 53.64 | 53.64 |
| freq_x_recency | 40.52 | 46.70 | 50.85 | 52.32 | 53.09 | 53.64 | 53.64 |
| layer_lru | 37.46 | 46.13 | 50.23 | 51.79 | 52.54 | 53.40 | 53.64 |
| cost_aware (≡ lru) | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 | 53.64 |
| freq_lru_warmup (B) | 30.93 | 40.91 | 49.36 | 52.07 | 53.07 | 53.64 | 53.64 |
| freq_lru_warmup_s4 (B) | 36.79 | 46.50 | 50.44 | 52.48 | 53.42 | 53.64 | 53.64 |
| **belady_cold (A, bound)** | **53.64** | 53.64 | 53.64 | 53.64 | 53.64 | **53.64** | 53.64 |

### 1.2 Regime C (prewarmed; labeled; never comparable to A/B)

| Policy | 8 | 12 | 16 | 20 | 24 | 32 | 48 | 64+ GiB |
|---|---|---|---|---|---|---|---|---|
| static_freq_prewarm | 60.48 | 72.52 | 78.84 | 85.13 | 91.43 | 100 | 100 | 100 |
| static_layer_freq_prewarm | 58.58 | 70.90 | 78.03 | 84.82 | 90.51 | 97.27 | 99.45 | 100 |
| freq_lru_prewarm pf0.25 | 44.52 | 52.46 | 57.33 | 60.45 | 62.58 | 66.23 | 72.52 | 78.84→100 |
| freq_lru_prewarm pf0.5 | 50.48 | 59.15 | 65.62 | 69.39 | 72.52 | 78.84 | 91.43 | 100 |
| freq_lru_prewarm pf0.75 | 56.17 | 66.13 | 72.52 | 77.25 | 81.98 | 91.43 | 100 | 100 |
| freq_lru_prewarm pf0.9 | 58.07 | 70.21 | 76.31 | 81.98 | 87.64 | 99.00 | 100 | 100 |
| freq_lru_prewarm pf0.99 | 60.21 | 72.33 | 78.58 | 84.80 | 91.04 | 100 | 100 | 100 |
| lru_prewarm_topN (control) | 43.36 | 53.30 | 66.31 | 78.02 | 88.94 | 100 | 100 | 100 |
| belady_same_state_prewarm (bound) | 64.64 | 72.52 | 78.84 | 85.13 | 91.43 | 100 | 100 | 100 |
| belady_pin_same_state pf0.5 (bound) | 59.93 | 63.07 | 66.23 | 69.39 | 72.52 | 78.84 | 91.43 | 100 |
| belady_pin_same_state pf0.75 (bound) | 62.03 | 67.80 | 72.52 | 77.25 | 81.98 | 91.43 | 100 | 100 |

*(Figure corrected 2026-09-10: prior row used never-used-distance init;
exact contract init per `research/phase2-regime-c` @ 643d3559.)*

Every `freq_lru_prewarm(pf)` row is dominated pointwise by its
contract-matched bound `belady_pin_same_state(pf)` — dominance check PASS.

### 1.3 Cold SSD traffic and the wall (corrected; per-token wall = MiB/tok ÷ 1024 ÷ BW)

| Policy | Regime | Budget | MiB/tok | wall @0.29 | @3 | @5 | @7 | @12 |
|---|---|---|---|---|---|---|---|---|
| no cache | A | 0 | 4,063.3 | 13.68 s | 1.32 | 0.79 | 0.57 | 0.33 |
| lru | A | 16 GiB | 1,997.8 | 6.73 s | 0.65 | 0.39 | 0.28 | 0.16 |
| lru | A | 32 GiB | 1,883.8 | 6.34 s | 0.61 | 0.37 | 0.26 | 0.15 |
| freq_lru_warmup_s4 | B | 16 GiB | 2,157.1 | 7.26 s | 0.70 | 0.42 | 0.30 | 0.18 |
| freq_lru_warmup | B | 24 GiB | 1,906.9 | 6.42 s | 0.62 | 0.37 | 0.26 | 0.15 |
| static_freq_prewarm | C | 16 GiB | 859.8 | 2.90 s | 0.28 | 0.17 | 0.12 | 0.07 |
| static_freq_prewarm | C | 24 GiB | 348.2 | 1.17 s | 0.11 | 0.07 | 0.05 | 0.03 |
| static_freq_prewarm | C | 32 GiB | 0 | 0 | 0 | 0 | 0 | 0 |
| belady_cold | A | 24 GiB | 1,883.8 | 6.34 s | 0.61 | 0.37 | 0.26 | 0.15 |

### 1.4 Evictions (whole 16-token response, pooled, regime A)

| Policy | 8 | 16 | 24 | 32 GiB |
|---|---|---|---|---|
| lru | 2,401 | 1,760 | 1,222 | 465 |
| engine_priority_lru | 3,747 | 2,962 | 2,076 | 576 |
| freq_lru_warmup (B, s0) | 3,317 | 2,273 | 1,318 | 466 |
| belady_cold | 1,722 | 1,401 | 1,079 | 437 |

Static (prewarmed) placement does zero eviction work by construction —
relevant because Phase 1 measured reservation/LRU-scan overhead inside
the 13.0 s whole-run reservation bucket.

## 2. Per-policy verdicts (mechanical reasons, corrected)

1. **lru — keep as the host default.** Validated exactly (±1 vs sealed).
   Sits 2.8 pp below cold MIN at 16 GiB and ties MIN by 32 GiB; the cold
   ceiling on this trace is the LRU curve itself.
2. **engine_priority_lru — a defect, not a policy.** Reproduces v60 (±2).
   Suppresses VRAM hits 2.07×/3.25× at the sealed budget. Repair = evict
   by `last_used` only (`VRAM_PRIORITY_AUDIT.md`).
3. **arc.** Recency-adaptive; recency is the wrong signal. Worse below
   16 GiB, ties above. **Rejected.**
4. **lfu (online).** Ties LRU at ≥ 16 GiB; worse at 8. Online counts lag
   the short window. **Rejected.**
5. **static_freq_prewarm.** Regime C only. The pin value is real *as a
   prewarm contract* (+20.2 pp over cold LRU at 16 GiB; +38.3 pp at 24),
   and it beats the admission variant (lru_prewarm_topN) at every budget
   — but MIN given the same prewarm ties or beats it, and its coverage
   assumption is workload-class-dependent (top-1 % = 6.8 % of
   activations). **Not a causal policy; a deployment contract.**
6. **static_layer_freq_prewarm.** Within 0.9–2.7 pp of global prewarm;
   relevant only if the host tier is sharded per-engine with hard caps.
   Same regime-C contract caveat.
7. **freq_lru_prewarm (pin + LRU remainder).** Pathological at small
   budgets (marginal slope −8,186 slow-MiB per added GiB at 8 GiB: the
   remainder evicts prewarmed records to cache never-repeating ones);
   dominated pointwise by its own contract-matched bound
   `belady_pin_same_state(pf)`. **Rejected as a family.**
8. **freq_lru_warmup / _s4 (regime B).** The honest online pin: does not
   beat plain LRU at any budget on this window (−1.5 pp at 16 GiB for
   s0; s4 recovers to ≈ LRU), and carries repopulation exposure when the
   pin set exceeds what the warmup pass left resident (charged misses —
   s0: 321 @8 GiB, 266 @12, 0 at ≥ 16; s4: 0 @8, 92 @12, 180 @16, 0 at
   ≥ 20 GiB). **Not selected.** See PHASE2_RECOMMENDATION.md §6 for why
   this kills the warmup-pin recommendation.
9. **freq_x_recency.** Indistinguishable from LRU. **Rejected.**
10. **layer_lru.** ≤ 0.6 pp below pooled LRU; adds bookkeeping. **Rejected.**
11. **cost_aware.** Uniform record size ⇒ provably LRU; confirmed
    identical at every row. **Rejected as a distinct policy; retained as
    the required equivalence check.**
12. **belady_cold / belady_same_state_prewarm / belady_pin_same_state.**
    Offline MIN bounds for regimes A / C(top-slots) / C(pin+remainder).
    Never a recommendation; the in-regime dominance checks PASS at every
    budget (see §4).

## 3. VRAM tier matrix (per GPU; same stream, cold start)

| Policy | 1 | 2 | 3 | 3.5 (v60) | 4 | 6 | 8 GiB |
|---|---|---|---|---|---|---|---|
| cuda0 engine_priority (production) | 4.1 | 8.8 | 11.2 | 12.6 | 13.7 | 21.8 | 31.4 |
| cuda0 plain LRU | 0.0 | 19.1 | 25.1 | 26.0 | 29.9 | 41.5 | 45.6 |
| cuda1 engine_priority | 5.7 | 10.1 | 12.4 | 13.1 | 14.1 | 23.9 | 37.0 |
| cuda1 plain LRU | 0.0 | 32.9 | 40.2 | 42.5 | 46.4 | 51.6 | 55.7 |

Full curves: `PHASE2_CAPACITY_CURVES.csv` rows `vram,gpu0/gpu1`. The
repair verdict is `VRAM_PRIORITY_FIX_RECOMMENDED`
(`VRAM_PRIORITY_AUDIT.md`).

## 4. The decision landscape (corrected)

- Cold start: **the reachable optimum is the LRU curve itself** (within
  2.8 pp of MIN at the operating point). No count-based, composite, or
  budgeted policy adds a single pp.
- The only shape that beats the recency ceiling is a **prewarm
  contract** (regime C, labeled) — an architecture decision about
  cross-request residency (the dee-serve shape), not a cache-policy
  choice, and MIN-given-the-same-state shows even there the win comes
  from the prewarm size, not from static placement.
- The VRAM priority repair is orthogonal, free, and the single
  highest-value memory-tier change: **VRAM_PRIORITY_FIX_RECOMMENDED**.

## 5. Mechanical checks embedded in the v4 simulator

1. Every row carries `regime`, `initial_records`, `initial_state_gib`,
   `repopulation_charged`.
2. In-regime dominance checks (all PASS, 8..128 GiB host / 1..8 GiB VRAM):
   no A/B policy exceeds `belady_cold`; `freq_lru_prewarm(pf)` never
   exceeds `belady_pin_same_state(pf)`; `static_freq_prewarm` never loses
   to `lru_prewarm_topN`.
3. Warmup repopulation is charged and reported per budget.
4. Sealed anchors re-verified every run (`validation_v4.json`).
