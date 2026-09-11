# PHASE2_BYTE_FLOOR.md — cold-byte floors for DSV4-Flash on the measured T4 /tmp bank (v4, corrected)

**Scope.** Offline analysis of dee's own sealed route traffic (v50 canonical
journal, sha256 `665aac3e…`, embedded by hash in the sealed v60
`dee4-v3-trace` metadata). No GPU time, no implementation, no merge.
All numbers derive from the corrected, regime-labeled
`tools/phase2_ws_policy_sim_v4.py` (see `CAUSALITY_AND_INITIAL_STATE_AUDIT.md`
for what changed vs b7b9c7f and why). Host-tier and VRAM-tier counters
reproduce the sealed `fill-live` host-pack and `v60` engine_stats within
±2 counts (`results/validation_v4.json`).

## 1. Physical constants (measured, not assumed)

| Quantity | Value | Source |
|---|---|---|
| DEE4 packed FP4 record | 13,369,344 B = 12.75 MiB | sealed v60 `dee4-metadata.json` (`record_bytes`) |
| Unique (layer, expert) records | 2,364 | v50 journal; v60 metadata `total_experts` |
| Full 16-token working set | **29.43 GiB** = 31,605,129,216 B | v60 metadata `total_bytes` |
| Forward passes | 16 (1 prefill of 7 rows + 15 decode) | journal |
| Engine requests (dedup per layer call) | 5,099 | journal, engine-truth order |
| Prefill (step 0) unique records | 1,229 (= the 80 %-coverage point, exactly) | journal |
| Decode-only activations | 3,870 (258/token) | journal |
| Bank read ceiling (production qd6/l3) | **0.29 GiB/s** | Phase-1 live matrix |
| Bank read ceiling (lanes1 / qd1 riders) | 0.35 / 0.37 GiB/s | Phase-1 live matrix |
| Decode wall (live FILL arm) | 71.479 s | `session-summary.json` |
| v60 decode wall / ITL p50 | 72.6 s / 4.60 s | sealed v60 `result.json` |

The Phase-1 matrix showed all patterns/concurrency settings converge at
0.29–0.37 GiB/s with **zero lane scaling** and pread ≈ 100 % of service:
the ceiling is the /tmp-backed device, not the read path. Per the Phase-1
verdict the fix is FEED-side — fewer/smaller cold fills, earlier submission.

## 2. The structural byte floor (regime A: cold start)

Every miss is one 12.75 MiB record. For a cold-start policy P with h(P)
hits over the 5,099 requests:

- charged misses = 5,099 − h(P) (+ any warmup repopulation, regime B)
- **SSD bytes for prefill** = prefill_misses × 12.75 MiB — identical for
  every policy that starts cold (1,229 first-touch records = 15.28 GiB;
  exactly recovered by plain LRU at 17 GiB: sim prefill misses 1,229)
- **SSD bytes for decode** = (decode_misses + repopulation) × 12.75 MiB
- SSD bytes/token = total ÷ 16; wall = bytes ÷ bank bandwidth.

Decode floors (corrected, regime A/B):

| Policy | Budget | Hits | MiB/tok (prefill+decode) | wall @0.29 | @0.33 | @0.37 |
|---|---|---|---|---|---|---|
| No cache (compulsory only) | 0 | 0 | 4,063.3 | 13.68 s | 12.02 | 10.71 |
| Plain LRU (= today) | 8 GiB | 40.3 % | 2,424.9 | 8.17 s | 7.18 | 6.40 |
| Plain LRU (= today) | 16 GiB | 50.8 % | 1,997.8 | 6.73 s | 5.91 | 5.27 |
| Plain LRU (= today) | 32 GiB (saturated) | 53.6 % | 1,883.8 | 6.34 s | 5.57 | 4.97 |
| Belady/MIN (bound) | 16 GiB | 53.6 % | 1,883.8 | 6.34 s | 5.57 | 4.97 |
| freq_lru_warmup_s4 (B) | 16 GiB | 50.4 % | 2,157.1 | 7.26 s | 6.38 | 5.69 |
| freq_lru_warmup (B) | 24 GiB | 53.1 % | 1,906.9 | 6.42 s | 5.64 | 5.03 |

**The LRU-family floor is 1,883.8 MiB/token and no causal policy can go
below it on this trace** — cold Belady/MIN itself saturates there because
every repeat sits ≥ 213 distinct records away (stack-distance histogram,
`results/reuse_distance.json`).

## 3. The regime-C (prewarmed) floor — a contract, not a cold-start policy

With an explicitly prewarmed top-N set (cross-request residency; one-time
~N × 12.75 MiB prewarm read amortized outside the measured window):

| Prewarm size | Coverage | SSD MiB/token | wall @0.29 | @3 | @12 |
|---|---|---|---|---|---|
| 8 GiB (top-642) | 60.5 % | 1,605.7 | 5.41 s | 0.52 | 0.13 |
| 12 GiB (top-963) | 72.5 % | 1,116.4 | 3.76 s | 0.36 | 0.09 |
| 16 GiB (top-1,285) | 78.8 % | 859.8 | 2.90 s | 0.28 | 0.07 |
| 24 GiB (top-1,927) | 91.4 % | 348.2 | 1.17 s | 0.11 | 0.03 |
| 29.43 GiB (all) | 100 % | 0 | 0 | 0 | 0 |

Caveats that v3 omitted: (1) this requires the prewarm to be resident
*before* the measured request stream (a labeled regime-C contract);
(2) MIN given the same prewarm ties or beats the static pin (91.43 vs
91.43 % at 24 GiB — an exact tie), so the pin vs dynamic choice is not
where the value is — the prewarm *size* is; (3) rank stability across
requests is an assumption about the workload class, not a measured
property of this single window (top-1 % covers only 6.8 % of
activations — the tail is wide).

*(Figure corrected 2026-09-10: prior row used never-used-distance init;
exact contract init per `research/phase2-regime-c` @ 643d3559.)*

## 4. Activation-coverage curve (the prewarm ceiling)

Top-X % of (layer, expert) records by activation frequency → coverage of
the 5,676 activations (full trace; decode-only within ±1 pp):

| Top X % | Records | GiB | Coverage |
|---|---|---|---|
| 1 % | 23 | 0.29 | 6.8 % |
| 2 % | 47 | 0.59 | 12.2 % |
| 5 % | 118 | 1.47 | 23.8 % |
| 10 % | 236 | 2.94 | 37.0 % |
| 20 % | 472 | 5.88 | 53.1 % |
| 30 % | 709 | 8.83 | 64.0 % |
| 50 % | 1,182 | 14.72 | 79.2 % |
| 75 % | 1,773 | 22.08 | 89.6 % |
| 100 % | 2,364 | 29.43 | 100 % |

Inverse form: 80 % coverage needs 1,229 records (**15.30 GiB** — exactly
the prefill's unique-record count, by construction of the window); 90 %
needs 1,797 (22.37 GiB); 95 % needs 2,081 (25.91 GiB).

Per-layer shape: 36–99 unique experts per layer (median 53);
layer-budgeted placement loses ≤ 1.5 pp vs global at 16–24 GiB
(`results/layer_locality.json`).

## 5. Future-hardware floors (3 / 5 / 7 / 12 GiB/s)

Same byte volumes, faster tiers — the *ordering* is unchanged, the walls
shrink. Per-token wall = (SSD MiB/token)/1024 ÷ BW:

| Row | MiB/tok | @3 | @5 | @7 | @12 |
|---|---|---|---|---|---|
| LRU 16 GiB | 1,997.8 | 0.65 s | 0.39 s | 0.28 s | 0.16 s |
| Belady/MIN bound | 1,883.8 | 0.61 s | 0.37 s | 0.26 s | 0.15 s |
| Prewarm 16 GiB | 859.8 | 0.28 s | 0.17 s | 0.12 s | 0.07 s |
| Prewarm 24 GiB | 348.2 | 0.11 s | 0.07 s | 0.05 s | 0.03 s |

**Consequence:** the value of any placement/prewarm work is highest at the
current 0.29–0.37 GiB/s bank and decays with storage speed — a "now"
lever. Any mechanism must be cheap to implement because its advantage
window is the current storage class.

## 6. VRAM tier floor (and the priority artifact)

Per GPU at the sealed v60 budget (3.5 GiB = 281 packed-FP4 slots): the
production score captures 12.6/12.5 % of that GPU's requests; plain LRU
captures 26.0/40.5 %. The repair saves 1,083 device loads and **14.48 GB
of H2D per 16-token response** (cuda0 4.69 GB + cuda1 9.79 GB), with a
host-fill upside bounded at ≤ 6.6 GB NVMe per response. Full audit:
`VRAM_PRIORITY_AUDIT.md`. The VRAM tier remains budget-capped (~3.5–4 GiB
after dense + engine) — the host tier is where multi-GiB/token working
sets live.

## 7. Memory cost accounting

- Host budget B (pooled): today 17 GiB within 22.9 GiB peak RSS on the
  31.35 GiB Kaggle host (v60: 22.48 GiB HWM, 3.5 GiB/GPU VRAM cache,
  8.5+8.5 host packs). 24 GiB pooled projects ≈ 29–30 GiB peak RSS —
  outside the campaign's historical safety margin. 29.43 GiB prewarm
  (100 % coverage) does not fit alongside dense + engine.
- **Marginal value of host RAM (cold/LRU, regime A):** 1,020 slow-MiB per
  added GiB (8→12), 688 (12→16), 242 (16→20), 124 (20→24), 45 (24→32),
  0 beyond 32 — the RAM knee is ≈ 16 GiB for causal operation.
- **Marginal value of VRAM (LRU after repair):** 1,538–2,550 device-load
  MiB per added GiB per GPU in the 2–6 GiB range (per-token working set
  per GPU ≈ 1.6 GiB); beyond ~6–8 GiB it decays (688/644 MiB/GiB).
- Dense/shared/non-expert weights: already resident by design (separate
  from the expert tiers; v60 allocated 6.99/6.76 GiB incl. cache) — no
  resident class competes with the expert cache for RSS except the
  engine's own buffers; see PHASE2_RECOMMENDATION.md §5.

## 8. Verdict (corrected)

1. The byte floor is real and FEED-side: 29.43 GiB unique set vs a
   0.29–0.37 GiB/s bank ⇒ ≥ 84 s of unavoidable bank time if nothing is
   cached; today's LRU halves it; **no causal policy can halve it again**
   (the cold floor is 1,883.8 MiB/token, reached by 32 GiB).
2. The large further reductions (859.8 → 348.2 → 0 MiB/token) belong to
   an explicitly labeled **prewarm contract** (regime C), which is an
   architecture decision (persistence across requests — the dee-serve
   shape), not a cache-policy selection, and must never be scored
   against cold-start rows.
3. The one memory-tier change that is free, causal, and immediately
   justified is the **VRAM priority repair** (2.1–3.2× VRAM hits,
   −14.5 GB H2D per response, robust across GPUs and steps ≥ 3).

Artifacts: `results/sim_rows_v4.csv`, `results/sim_rows_v4_derived.csv`,
`results/ram_slope_v4.csv`, `results/vram_audit.json`,
`results/layer_locality.json`, `results/reuse_distance.json`,
`results/validation_v4.json`, `PHASE2_CAPACITY_CURVES.csv`.
