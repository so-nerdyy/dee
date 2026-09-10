# PHASE2_POLICY_MATRIX.md — full policy × budget matrix with per-policy verdicts

All numbers from `tools/phase2_ws_policy_sim.py` on the validated stream
(5,099 engine requests; see PHASE2_WORKING_SET.md §1 for the validation
anchors). Hit rates are host-tier, pooled both GPUs, unless labeled VRAM.
Derived wall columns use the Phase-1 measured bank ceiling (production
qd6/l3 = 0.29 GiB/s) and future floors 3/5/7/12 GiB/s; the full table with
every wall column is `PHASE2_CAPACITY_CURVES.csv` +
`research/phase2-ws-policy/results/sim_rows_derived.csv`.

## 1. Master matrix — full pooled host tier

### 1.1 Hit rate (%) by policy × budget

| Policy | 8 GiB | 12 GiB | 16 GiB | 20 GiB | 24 GiB | 32 GiB | 48 GiB | 64–128 GiB |
|---|---|---|---|---|---|---|---|---|
| lru (today) | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 | 53.64 | 53.64 |
| engine_priority_lru | 13.92 | 23.02 | 34.09 | 43.68 | 50.91 | 53.64 | 53.64 | 53.64 |
| arc | 31.30 | 42.91 | 50.44 | 52.23 | 53.09 | 53.64 | 53.64 | 53.64 |
| lfu | 38.83 | 46.56 | 50.83 | 52.32 | 53.09 | 53.64 | 53.64 | 53.64 |
| freq_x_recency | 40.52 | 46.70 | 50.85 | 52.32 | 53.09 | 53.64 | 53.64 | 53.64 |
| layer_lru | 37.46 | 46.13 | 50.23 | 51.79 | 52.54 | 53.40 | 53.64 | 53.64 |
| cost_aware (= lru) | 40.32 | 46.60 | 50.83 | 52.32 | 53.09 | 53.64 | 53.64 | 53.64 |
| freq_lru pf0.25 | 44.52 | 52.46 | 57.33 | 60.44 | 62.58 | 66.23 | 72.52 | 78.84–100 |
| freq_lru pf0.5 | 50.48 | 59.15 | 65.62 | 69.39 | 72.52 | 78.84 | 91.43 | 100 |
| freq_lru pf0.75 | 56.17 | 66.13 | 72.52 | 77.25 | 81.98 | 91.43 | 100 | 100 |
| freq_lru_warmup (online) | 37.22 | 50.34 | 61.95 | 67.82 | 71.96 | 77.74 | 77.74 | 77.74 |
| static_freq (pin ceiling) | 60.48 | 72.52 | 78.84 | 85.13 | 91.43 | 100 | 100 | 100 |
| static_layer_freq | 58.58 | 70.90 | 78.03 | 84.82 | 90.51 | 97.27 | 99.45 | 100 |
| belady (bound) | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 | 53.64 |

### 1.2 Cold SSD traffic and the wall (selected rows)

`misses/token` (MiB/token) and predicted cold-storage wall seconds per
16-token response at the measured ceiling; future floors in brackets.

| Policy | Budget | MiB/tok | wall @0.29 | @3 | @5 | @7 | @12 |
|---|---|---|---|---|---|---|---|
| no cache | 0 | 4,063.3 | 13.68 s | 1.32 | 0.79 | 0.57 | 0.33 |
| lru | 16 GiB | 1,997.8 | 6.73 s | 0.65 | 0.39 | 0.28 | 0.16 |
| lru | 32 GiB (saturated) | 1,883.8 | 6.34 s | 0.61 | 0.37 | 0.26 | 0.15 |
| freq_lru_warmup | 16 GiB | 1,609.6 | 5.42 s | 0.52 | 0.31 | 0.22 | 0.13 |
| freq_lru_warmup | 24 GiB | 1,137.2 | 3.83 s | 0.37 | 0.22 | 0.16 | 0.09 |
| freq_lru pf0.75 | 24 GiB | 732.3 | 2.47 s | 0.24 | 0.14 | 0.10 | 0.06 |
| static_freq | 16 GiB | 859.8 | 2.90 s | 0.28 | 0.17 | 0.12 | 0.07 |
| **static_freq** | **24 GiB** | **348.2** | **1.17 s** | 0.11 | 0.07 | 0.05 | 0.03 |
| static_freq | 32 GiB | 0 | 0 | 0 | 0 | 0 | 0 |
| belady | 24 GiB | 1,906.1 | 6.42 s | 0.62 | 0.37 | 0.26 | 0.15 |

### 1.3 Evictions (whole 16-token response, pooled)

| Policy | 8 GiB | 16 GiB | 24 GiB | 32 GiB |
|---|---|---|---|---|
| lru | 2,401 | 1,760 | 1,222 | 465 | 0 |
| engine_priority_lru | 3,747 | 2,962 | 2,076 | 576 | 0 |
| arc | 2,861 | 1,948 | 1,242 | 465 | 0 |
| freq_lru pf0.5 | 2,204 | 1,601 | 1,110 | 437 | 0 |
| freq_lru_warmup | 2,880 | 2,050 | 1,297 | 466 | 0 |
| static_freq / static_layer_freq | 0 | 0 | 0 | 0 | 0 |
| belady | 1,722 | 1,401 | 1,079 | 437 | 0 |

Static placement does zero eviction work by construction — relevant because
Phase 1 measured reservation/LRU-scan overhead inside the 13.0 s
whole-run reservation bucket; pinning eliminates that bookkeeping entirely.

## 2. Per-policy verdicts (mechanical reasons)

1. **lru — current host default.** Validated exactly. Ceiling 53.64 %:
   every repeat sits ≥ 213 distinct records away, so beyond ~1,900 slots
   recency converts nothing new. Keep as the *remainder* semantics but not
   as the placement policy.
2. **engine_priority_lru — the sealed VRAM semantics.** Reproduces v60
   (±2). Its stale priority boost suppresses VRAM hits 2.1× vs plain LRU at
   identical budget (12.9 % vs 26.6 % at 281 slots). Not a candidate: a
   defect. **Repair = plain LRU (drop `priority·2²⁰`), zero-cost, no new
   policy machinery.**
3. **arc.** Applicable mechanically; adapts toward recency; recency is the
   wrong signal here. Within noise of LRU at ≥ 16 GiB, worse below (31.3 %
   vs 40.3 % at 8 GiB). Extra ghost-list state buys nothing. **Rejected.**
4. **lfu (online).** Ties LRU at ≥ 16 GiB; worse at 8 (38.8 %). Online
   counts lag the short window and its LRU tiebreak collapses to LRU
   behavior. **Rejected.**
5. **static_freq (offline top-N).** The pinning ceiling: 91.4 % at 24 GiB,
   100 % at 29.43 GiB. Requires whole-trace counts (not deployable as-is)
   but bounds every frequency policy. Its online proxies are §5 and §8.
6. **static_layer_freq.** Within 0.8–2.7 pp of global static pin at every
   budget; gives per-layer budgets (useful if the host tier is ever
   sharded per-engine with hard per-engine caps, as today's
   8.5+8.5 split is). Equivalent policy value; more implementation surface.
7. **freq_lru (pin + LRU remainder).** Dominated by pure pinning: the LRU
   remainder converts almost nothing on this window (its hits ≈ the plain
   LRU hits it displaces). **Not the default.** Kept as the fallback shape
   (see PHASE2_RECOMMENDATION.md) because it degrades gracefully when the
   pin set is wrong (stream drift) — the remainder still catches recency.
8. **freq_lru_warmup (online pin from prefill counts).** The deployable
   72.0 % at 24 GiB (vs 91.4 % offline ceiling; 79 % of it), 77.7 %
   saturated at 32 GiB. No future knowledge: counts are taken from the
   first forward pass (7 rows), which the run already executes. **Default
   candidate.**
9. **freq_x_recency.** 40.5 % at 8 GiB — indistinguishable from LRU; the
   recency half of the score reproduces the LRU ceiling and the frequency
   half arrives too late (counts build over the window). **Rejected.**
10. **layer_lru.** ≤ 0.6 pp below pooled LRU everywhere; adds per-layer
    bookkeeping for no gain (per-layer working sets are small: 36–99
    records). **Rejected.**
11. **cost_aware.** All records are exactly 13,369,344 bytes ⇒ greedy
    cost-aware eviction is provably LRU; confirmed numerically identical
    (every row matches lru). **Rejected as a distinct policy; recorded as
    the required equivalence check.**
12. **belady.** Offline MIN. *Upper bound only.* Ties LRU here (the trace's
    reuse is frequency-shaped, not recency-shaped) and is beaten by every
    pinned policy — evidence that the reachable optimum is the pin
    ceiling, not MIN.

## 3. VRAM tier matrix (per GPU; cuda0 shown, cuda1 within ±1.5 pp)

| Policy | 1 GiB | 2 GiB | 3.5 GiB | 4 GiB | 6 GiB | 8 GiB |
|---|---|---|---|---|---|---|
| engine_priority_lru (sealed) | — | — | 12.87 | — | — | — |
| lru | 6.44 | 14.31 | 26.58 | 27.73 | 31.05 | 33.65 |
| static top-N | 27.6 | 43.5 | ~37* | ~40 | ~46 | ~51 |

*At 281 slots the global top-281 set is layer-skewed (cuda0 hosts 1,247 of
2,364 records); per-layer top-B placement scores ~35 %. VRAM static pinning
is mechanically identical to the host policy; the tier's budget cap (≤ 4
GiB realistic) is what limits it, not the policy.

## 4. The decision landscape (why exactly one default + one fallback)

- The ceiling of everything realizable is the static pin
  (activation-coverage curve). Among *implementable* policies, the online
  warmup pin captures ~79–85 % of that ceiling at equal RAM.
- The only other policy family with a distinct shape is pin+LRU
  (graceful degradation under drift) — strictly worse here, safer in
  general. That is the fallback trade.
- Everything else is LRU-shaped on this trace and adds state or risk
  without a single pp of gain.
- The VRAM fix is orthogonal (a repair, not a placement policy) and is
  carried as a mandatory rider in the recommendation, not as a separate
  policy choice.

Full machine-readable matrix: `research/phase2-ws-policy/results/sim_rows.csv`
(336 rows: 2 scopes × budgets × 12 policies × pin_frac variants) and
`sim_rows_derived.csv` (same rows + SSD bytes/token + all wall columns).
