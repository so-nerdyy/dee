# MISS_CRITICALITY.md — were the 52 eliminated misses demand-blocking?

Branch: `experiment/pack-cap-ab` · Date: 2026-09-07
Machine-readable: `results/miss-criticality.json`

## 1. Evidence basis (what exists and what does not)

| Evidence | Status |
|---|---|
| Per-token decode wall | MEASURED (`result.json.per_token_accounting`, all 4 arms) |
| Per-token read-batch wall (`source_read_wall_ms`) | MEASURED |
| Per-token host-pack misses/hits/evictions, storage bytes | MEASURED |
| Aggregate read latency (p50/p95/max), worker overlap % | MEASURED (`expert_store`) |
| **Per-request read submit / complete / demand timestamps** | **NOT RECORDED** (stage profiling disabled in every arm) |
| Page-cache hit counters, major/minor faults | NOT RECORDED |

Because per-request timestamps were not recorded, **every eliminated miss is
UNKNOWN at per-request granularity** — by design, not by omission of
analysis. The classification below is the strongest token-level attribution
the sealed evidence supports.

## 2. Classification rule (pre-stated)

A cap effect at token *t* requires the B−A wall delta to have the **same
sign in both sessions** (candidate faster regardless of which ran first).
Sign flips are position effects; |Δ| ≤ 0.05 s is treated as null.

## 3. Result per token (from `results/miss-criticality.json`)

| tok | d1 = B−A (s1) | d2 = B−A (s2) | elim | rd 2nd−1st | class |
|---|---|---|---|---|---|
| 1 | +0.141 | −0.168 | 0 | +0.52 | no elim |
| 2 | +0.388 | +0.169 | 0 | +0.16 | no elim |
| 3 | +0.465 | +0.055 | 6 | +0.42 | CANDIDATE_REGRESSION |
| 4 | +0.134 | +0.051 | 7 | −0.10 | CANDIDATE_REGRESSION |
| 5 | −0.047 | +0.054 | 4 | −0.22 | NOT_DEMAND_BLOCKING (sign flip) |
| 6 | −0.008 | +0.098 | 2 | −0.19 | NOT_DEMAND_BLOCKING (sign flip) |
| 7 | −0.070 | −0.096 | 5 | +0.29 | CRITICAL_WAIT_CANDIDATE (sub-noise) |
| 8 | −0.067 | −0.096 | 1 | +0.01 | CRITICAL_WAIT_CANDIDATE (sub-noise) |
| 9 | −0.059 | −0.118 | 5 | +0.09 | CRITICAL_WAIT_CANDIDATE (sub-noise) |
| 10 | +0.078 | −0.199 | 3 | +0.26 | NOT_DEMAND_BLOCKING (sign flip) |
| 11 | −0.257 | −0.192 | 5 | −0.01 | CRITICAL_WAIT_CANDIDATE (sub-noise) |
| 12 | −0.125 | −0.142 | 1 | +0.12 | CRITICAL_WAIT_CANDIDATE (sub-noise) |
| 13 | −0.790 | +0.016 | 4 | −1.10 | NOT_DEMAND_BLOCKING (sign flip) |
| 14 | −1.336 | +1.518 | 6 | −3.59 | NOT_DEMAND_BLOCKING (sign flip) |
| 15 | −0.609 | +1.427 | 2 | −2.41 | NOT_DEMAND_BLOCKING (sign flip) |

Eliminated-miss totals by class: 21 flip with position, 17 same-sign
B-faster, 13 same-sign B-slower.

## 4. Interpretation

- **21 of 51** eliminated misses sit on tokens where the wall delta flips
  sign with arm order — the position effect (see ORDER_EFFECT_FORENSICS.md)
  dominates those tokens (read-batch wall swings by −1.1 to −3.6 s at
  tokens 13–15 for the *second-position arm, either arm*).
- The 17 "same-sign B-faster" deltas are all ≤ 0.26 s. Cross-session
  same-arm variation on the same tokens is up to ~0.5 s (e.g. token 4:
  s1A 6.83 vs s2A 7.34), so these are **not distinguishable from session
  noise**. Even taken at face value, 17 misses yielding ≲ 0.26 s per
  ~1–5 misses/token is far below the serial 105 ms/miss prediction.
- The 13 "same-sign B-slower" deltas (tokens 3–4) show the cap doing
  nothing good there either.
- Aggregate support for "not demand-blocking": 62.2% of read-worker time
  was already overlapped (fill_worker 145.9 s vs batch wall 55.1 s on
  cuda0; 62.3% on cuda1), and read batches average ~3.4 concurrent reads
  (fill_batch_wall ≈ max of concurrent reads) — removing 1–2 reads from a
  multi-read batch shortens the batch wall by ~0.

## 5. Bottom line

No token shows a position-robust, beyond-noise wall reduction attributable
to the eliminated misses. **Zero of the 51 (≈52) eliminated misses can be
demonstrated demand-blocking from the sealed evidence**; the majority are
demonstrably absorbed (sign flips) and the rest are sub-noise. Per-request
criticality remains UNKNOWN pending a profiler-enabled run (stage
profiling with `cache_readiness_attribution` / `host_waits` enabled).
