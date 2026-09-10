# PHASE2_BYTE_FLOOR.md — cold-byte floors for DSV4-Flash on the measured T4 /tmp bank

**Scope.** Offline analysis of dee's own sealed route traffic (v50 canonical
journal, sha256 `665aac3e…`, arrays verified byte-identical to the Phase-1
`fill-live-t4x2-20260909` run). No GPU time, no implementation, no merge.
Everything below derives from `tools/phase2_ws_policy_sim.py`
(`research/phase2-ws-policy/results/`), whose host-tier and VRAM-tier
counters reproduce the sealed `fill-live` host-pack and `v60` engine_stats
numbers within ±2 counts (see `results/validation.json`).

## 1. Physical constants (measured, not assumed)

| Quantity | Value | Source |
|---|---|---|
| DEE4 packed FP4 record | 13,369,344 B = 12.75 MiB | `dee4-metadata.json` (`record_bytes`); = `3·inter·hidden·17/32` with inter 2048, hidden 4096 (`engine.cpp packed_fp4_cache_blob_bytes`) |
| Unique (layer, expert) records touched | 2,364 | sealed v50 journal |
| Full 16-token working set | 2,364 × 12.75 MiB = **29.43 GiB** | journal × record size |
| Forward passes | 16 (1 prefill of 7 rows + 15 decode) | journal |
| Engine requests (dedup per layer call) | 5,099 | journal, engine-truth order |
| Raw routed activations | 5,676 | journal |
| Decode-only activations | 3,870 (258/token) | journal |
| Bank read ceiling (production qd6/l3) | **0.29 GiB/s** | Phase-1 live matrix (`fill_matrix_ingest.json`) |
| Bank read ceiling (lanes1 / qd1 riders) | 0.35 / 0.37 GiB/s | Phase-1 live matrix |
| Rider seq/rand flat over lanes 1–8 | 0.33–0.38 / 0.37–0.51 GiB/s | Phase-1 live matrix |
| Decode wall (live FILL arm) | 71.479 s | `session-summary.json` |

The Phase-1 matrix showed all patterns/concurrency settings converge at
0.29–0.37 GiB/s with **zero lane scaling** and pread ≈ 100 % of service:
the ceiling is the /tmp-backed device, not the read path. Per the Phase-1
verdict the fix is FEED-side — fewer/smaller cold fills, earlier submission —
which is exactly what a host working-set policy buys.

## 2. The structural byte floor

Every miss is one 12.75 MiB record. Because consecutive-token expert overlap
exists (36.0 % of decode slots) but the *unique* working set is 29.43 GiB
while a 16-token response has only 5,099 layer-call requests, the floor of
cold bytes is set by two irreducible components:

1. **Compulsory cold fills: 2,364 records = 29.43 GiB total, 1,839 MiB/token
   over the response.** No policy that starts cold can avoid reading each
   resident record at least once.
2. **Repeat traffic**: 2,735 repeat requests (53.6 % of the request stream).
   How many of these hit RAM/VRAM is the *only* policy-variable quantity.

Therefore, for any cache policy P with hit rate h(P):

- cold records = 5,099 − hits(P)
- SSD bytes/token = (5,099 − hits(P)) × 12.75 MiB ÷ 16
- cold-storage wall = SSD bytes/token ÷ bank bandwidth

## 3. Floors at the measured 0.29–0.37 GiB/s

| Policy (host tier, pooled) | Budget | Hits | SSD MiB/token | Wall @0.29 | @0.33 | @0.37 |
|---|---|---|---|---|---|---|
| No cache (compulsory only) | 0 | 0 | 4,063.3 | 13.7 s | 12.0 s | 10.7 s |
| Plain LRU (= today) | 8 GiB | 40.3 % | 2,424.9 | 8.17 s | 7.18 s | 6.40 s |
| Plain LRU (= today) | 16 GiB | 50.8 % | 1,997.8 | 6.73 s | 5.91 s | 5.27 s |
| Plain LRU (= today) | 32 GiB | 53.6 % | 1,883.8 | 6.34 s | 5.57 s | 4.97 s |
| **Best realizable (static pin)** | **24 GiB** | **91.4 %** | **348.2** | **1.17 s** | **1.03 s** | **0.92 s** |
| Best realizable (static pin) | 32 GiB | 100 % | 0 | 0 | 0 | 0 |
| Belady MIN (unattainable; recency bound) | 24 GiB | 53.1 % | 1,906.1 | 6.42 s | 5.64 s | 5.03 s |
| Belady MIN (unattainable) | 32 GiB | 53.6 % | 1,883.8 | 6.34 s | 5.57 s | 4.97 s |

The decisive structural fact: **the LRU-family (including Belady) saturates at
53.6 %** on this trace because every repeat lies ≥ 213 distinct-experts away
in stack distance (see PHASE2_WORKING_SET.md §4) — the reuse horizon of the
16-token window (≈ 320 requests/token) exceeds any per-token budget that fits
host RAM. Static frequency placement is *not* stack-bounded: pinning the
frequency-ranked set converts far-distance repeats into hits. That is why the
only policies that beat the 53.6 % ceiling are the frequency-pinned family,
and why their ceiling is exactly the activation-coverage curve of §4.

**Wall-clock translation (16-token response, both GPUs pooled).** Today's
live decode is 71.5 s with ≈ 60.5 s of fill wait across engines. At the
production 0.29 GiB/s ceiling:

- LRU 8.5 GiB/GPU (the live configuration): predicted ≈ 1,930 MiB/token of
  bank traffic ≈ 30.5 GiB ≈ the measured ~29 GiB of misses — consistent with
  the Phase-1 42.0 s critical fill bucket.
- Static-pin at 24 GiB total: ≈ 348 MiB/token ≈ 5.3 GiB ≈ 18 s of bank time —
  a ~3.6× reduction in the cold-fill bucket, before any submission-overlap
  gains (which Phase 1 already proved are available: QD6 keeps the disk 96 %
  busy only *while requests exist*; FEED-side starvation gaps remain).

## 4. Activation-coverage curve (the static-pin ceiling)

Top-X % of (layer, expert) records by activation frequency → coverage of the
5,676 activations (full trace; decode-only is within ±1 pp):

| Top X % | Records | GiB | Activation coverage |
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

Inverse form: 80 % coverage needs 1,229 records (15.30 GiB); 90 % needs
1,797 (22.37 GiB); 95 % needs 2,081 (25.91 GiB).

Per-layer shape: each layer touches only 36–99 unique experts
(median 53) out of 256 — layer-budgeted placement loses ≤ 1.4 pp vs global
static pin at 16–24 GiB, so the choice between them is an engineering
(marginal-cost) decision, not a policy-value decision.

## 5. Future-hardware floors (3 / 5 / 7 / 12 GiB/s)

Same byte volumes, faster tiers — the *ordering* of policies is unchanged,
only the absolute walls shrink:

| Policy @ 16 GiB | SSD MiB/token | @3 GiB/s | @5 | @7 | @12 |
|---|---|---|---|---|---|
| LRU | 1,997.8 | 0.65 s | 0.39 s | 0.28 s | 0.16 s |
| Static pin | 859.8 | 0.28 s | 0.17 s | 0.12 s | 0.07 s |
| Static pin @ 24 GiB | 348.2 | 0.11 s | 0.07 s | 0.05 s | 0.03 s |

At 3 GiB/s (a commodity NVMe class) plain LRU's fill wall is already under
0.7 s/token and at 12 GiB/s it is noise; the policy gap collapses from
~2.6× to ~1.6× in wall terms. **Consequence:** the value of placement policy
is highest exactly at the current 0.29–0.37 GiB/s bank and decays with
storage speed — this is a "now" lever, not a forever lever. The
recommended policy must therefore be cheap to implement (it is: see
PHASE2_RECOMMENDATION.md) because its advantage window is the current
storage class.

## 6. VRAM tier floor

Per GPU at the sealed v60 budget (3.5 GiB = 281 FP4 slots): even plain LRU
captures only 26.6 % of that GPU's requests; the sealed engine-priority
semantics capture 12.9 %. The VRAM tier cannot be the primary lever — its
budget is structurally capped by the T4 envelope (~3.5–4 GiB/GPU after dense
weights + engine) while the host tier is the only place where a
multi-GiB-per-token working set can be held. Full per-GPU curves:
`PHASE2_CAPACITY_CURVES.csv` rows `vram,*`.

## 7. Memory cost accounting

- Host-tier budget B (pooled, both engines): B GiB of anonymous pinned-capable
  RAM. The live run held 8.5 + 8.5 = 17 GiB within a 22.9 GiB peak RSS on a
  31.35 GiB box; 24 GiB pooled (12 + 12) projects ≈ 29–30 GiB peak RSS —
  **fits the 2×T4 Kaggle host but not with slack; 32 GiB pooled (29.43 GiB
  working set + dense) does not.** Static-pin's advantage: it reaches the
  LRU-immune 91–100 % coverage *at 24 GiB*, where LRU tops at 53 %.
- VRAM: 3.5 GiB/GPU is already the accepted envelope (peak allocated
  7.13/8.75 GiB incl. dense at v60). Plain LRU needs zero extra VRAM.
- Every row's memory cost is the `ram_cost_GiB` column of
  `PHASE2_CAPACITY_CURVES.csv`.

## 8. Verdict

1. The byte floor is real and FEED-side: 29.43 GiB unique set vs a
   0.29–0.37 GiB/s bank ⇒ ≥ 84 s of unavoidable bank time if nothing is
   cached; today's LRU halves it; static frequency pinning at 24 GiB cuts it
   ~7×.
2. **Increasing RAM beyond ~24 GiB pooled buys nothing for any LRU-family
   policy** (saturated at 53.6 % by 32 GiB) and only buys the last 8.6 pp for
   static pinning (24 → 29.43 GiB = 100 %).
3. The knee of the host capacity curve for the best realizable policy is at
   **≈ 15.3 GiB (80 % coverage) with diminishing returns to 22.4 GiB (90 %)**
   — see PHASE2_RECOMMENDATION.md for the exact selection.

Artifacts: `results/sim_rows.csv`, `results/sim_rows_derived.csv`,
`results/reuse_distance.json`, `results/validation.json`,
`PHASE2_CAPACITY_CURVES.csv`.
