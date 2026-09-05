# HOST_PACK_FRONTIER.md — replay-validated miss frontier and marginal returns

Branch: `research/exact-staging` · Date: 2026-09-04
Machine-readable: `results/host_pack_frontier.csv`, `results/recalibrated_model.json`
(`frontier`, `replay_validation`), raw replay tables in
`results/pack_replay_sweep.json` (regenerable via `run_pack_sweep.py`).

## 1. Replay validation against sealed v65 (Phase E)

The production-equivalent LRU replay (`pack_replay`, semantics unchanged and
re-validated) run at the sealed budget 682 records/GPU on **both** available
journals:

| Journal | Decode misses | Sealed | Per-step max abs diff |
|---|---|---|---|
| v65 sealed | 1251 | 1252 | ≤2 records |
| profiled candidate (current code) | 1251 | 1252 | ≤2 records |

The two journals are byte-different (339,794 vs 337,042 B) but produce
identical read tables — the route stream is stable across the harness
change. Production pack semantics remain: **LRU at 682 records (8.5 GiB/GPU)**,
capped by `LRU_TOTAL_CAP_GIB=17.0` from a 12.87 GiB/GPU request.

## 2. Frontier (Phase C/E)

Decode compulsory floor (infinite budget): **1,135 misses** — a record's
first decode-phase demand. Everything above that is capacity miss.
Per-budget (miss table = replay; wall = **SIMULATED** with the recalibrated
model; memory classes from MEMORY_BUDGET.md):

| GiB/GPU | records | misses | compulsory | capacity | Δmiss/GiB | sim wall (s) | Δwall vs v65 (s) | class |
|---|---|---|---|---|---|---|---|---|
| 8.5 | 682 | 1251 | 1135 | 116 | — | 72.22 | −0.05 | SAFE |
| 9.0 | 722 | 1229 | 1135 | 94 | 44.0 | 71.15 | −1.12 | SAFE |
| 9.5 | 762 | 1209 | 1135 | 74 | 42.0 | 70.17 | −2.09 | SAFE |
| 10.0 | 803 | 1200 | 1135 | 65 | 34.0 | 69.74 | −2.53 | SAFE |
| 10.5 | 843 | 1189 | 1135 | 54 | 31.0 | 69.20 | −3.07 | BORDERLINE |
| 11.0 | 883 | 1184 | 1135 | 49 | 26.8 | 68.96 | −3.31 | BORDERLINE |
| 11.5 | 923 | 1177 | 1135 | 42 | 24.7 | 68.62 | −3.65 | NOT_SAFE |
| 12.75 | 1024 | 1153 | 1135 | 18 | 23.1 | 67.45 | −4.82 | NOT_SAFE |

(intermediate 10.75/11.25/12.0/12.25 rows in the CSV)

Reading the marginal column: the curve is steepest from 8.5→9.5 GiB/GPU
(~42–44 misses/GiB), halves by 10.0 (34), and decays to ~23/GiB at 12.75
while memory danger rises monotonically. **DRAM-for-DRAM, the first 1.5–2 GiB
above 8.5 buy 3× more than the last 2.**

## 3. Where diminishing returns meet the memory wall

- The offline floor (1,135) needs the full 2×16.15 GiB working set —
  unreachable in the 32 GB thesis (12.75 GiB/GPU still leaves 18 capacity
  misses and costs 5.7 GB over budget).
- Within SAFE territory (≤10.0 GiB/GPU), capacity misses fall 116→65
  (−44%); going to 12.75 would only cut the remaining 65→18 while leaving
  the envelope.
- Every wall delta above is SIMULATED under the recalibrated observational
  model (`wall = 754.74 ms + 48.679 ms × misses`, R²=0.979; held-out error
  ≤2.3%); run-to-run spread of the same config is ~1.0–1.9 s, so deltas
  under ~2 s require a matched A/B to confirm. No TPS claims.
