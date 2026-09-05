# MEMORY_BUDGET.md — full host-memory accounting and the 32 GB verdict

Branch: `research/exact-staging` · Date: 2026-09-04
Companion machine-readable: `results/memory_frontier.json`,
`results/seal_host_reuse/derived_memory_digests.json` (all inputs sha256-gated
to seal `14864034a7354e0e29e11c1c09f18b0863afe6a0`).

## 1. MEASURED non-pack memory (five seal runs, 2×T4, MemTotal 31.35 GiB)

| Component | Value | Source |
|---|---|---|
| Process VmHWM | 22.58–22.80 GiB | memory.json × 5 runs |
| Pack occupied (both GPUs) | 16.983 GiB | profile.json host_pack bytes |
| **Non-pack process HWM** | **5.60–5.82 GiB** | VmHWM − pack occupied |
| Min checkpoint MemAvailable | 7.76–7.97 GiB | minimum_checkpoint_host_mem_available_gib |
| MemTotal | 31.35 GiB | system_final_gib |
| System rest (others + kernel) | 0.91 GiB | MemTotal − process − min-avail |

Unknown sub-splits (aggregate-only measurement): pinned/CUDA/Python shares
inside non-pack HWM; continuous peak (final + per-token checkpoints only).
The non-pack HWM *decreased* slightly with reuse ON (5.82 → 5.61–5.81 GiB) —
no significant memory penalty from the reuse candidate.

## 2. DERIVED pack memory (the knob)

Production requests 12.87 GiB/GPU (`NATIVE_HOST_PACK_GPU{0,1}_BYTES`) but the
harness clamps the **total** with `LRU_TOTAL_CAP_GIB = 17.0`
(`deepseek_v4_native_generate.py:1347`), yielding 8.5 GiB/GPU = 682 records.
The live A/B knob is the runtime cap, not the requested bytes.

## 3. The conservative envelope and classification

    projected_total(GB) = (2·budget + 5.817 + 0.91 + 0.5) GiB × 1.0737
                          └── pack ──┘  └measured┘ └unk┘ └Gib→GB┘

`0.5 GiB` is an explicit UNKNOWN growth allowance (allocator/page-cache at
larger pack). Classification: `SAFE_FOR_32GB` requires ≥2.0 GB headroom AND
≥2.0 GiB projected min-available AND ≥2.0 GiB margin to the nearest clean OOM
observation; `BORDERLINE_FOR_32GB` = fits arithmetic but fails one gate;
else `NOT_SAFE_FOR_32GB`.

Empirical OOM ledger (harness header comments, earlier campaign runs —
measured on this host class): 25.74 GiB pack total **survived** (v8);
27.68 GiB **OOM-killed** (v9); v11's 26.0 GiB OOM is confounded by madvise
DONTNEED re-fault thrash.

| Budget/GPU | Total GB projected | Headroom to 32 GB | Min avail (GiB) | OOM margin (GiB) | Class |
|---|---|---|---|---|---|
| 8.5 | 27.92 | 4.08 | 7.12 | 10.68 | SAFE |
| 9.0 | 29.09 | 2.91 | 6.12 | 9.68 | SAFE |
| 9.5 | 30.23 | 1.77 | 5.12 | 8.68 | SAFE |
| 10.0 | 31.36 | 0.64 | 4.12 | 7.68 | SAFE |
| 10.5 | 32.52 | −0.52 | 3.12 | 6.68 | BORDERLINE |
| 10.75 | 33.10 | −1.10 | 2.62 | 6.18 | BORDERLINE |
| 11.0 | 33.68 | −1.68 | 2.12 | 5.68 | BORDERLINE |
| 11.25 | 34.26 | −2.26 | 1.62 | 5.18 | BORDERLINE |
| 11.5 | 34.85 | −2.85 | 1.12 | 4.68 | NOT_SAFE |
| 12.0 | 36.01 | −4.01 | 0.12 | 3.68 | NOT_SAFE |
| 12.25 | 36.59 | −4.59 | −0.38 | 3.18 | NOT_SAFE |
| 12.75 | 37.71 | −5.71 | −1.38 | 2.18 | NOT_SAFE |

## 4. Verdicts

- **Is 12.75 GiB/GPU safe for the 32 GB thesis? NO.** Projected 37.71 GB
  (decimal) — 5.7 GB over the limit, and within 2.2 GiB of the nearest
  measured clean OOM (27.68 GiB pack total). It was a good research point;
  it is not a production candidate. REJECTED for the main thesis.
- **Largest SAFE point: 10.0 GiB/GPU** (cap 20 GiB total): projected
  31.36 GB, headroom 0.64 GB — fits, but note headroom <2 GB makes 10.5+
  BORDERLINE by rule; 10.0 passes all three gates.
- **Recommended default (risk-balanced): 9.5 GiB/GPU** (cap 19 GiB):
  projected 30.23 GB, headroom 1.77 GB, min-available 5.12 GiB, OOM margin
  8.68 GiB, and it already captures ~74% of the SAFE-tier miss reduction
  (42 of 51 misses) that 10.0 offers — the marginal curve is flattest
  exactly where memory pressure starts.
- **Scaling point (NOT a requirement):** on 48/64 GB hosts, 12.75 GiB/GPU
  becomes acceptable (37.7 GB < 48 GB with margin) and reaches the
  offline floor region (capacity misses 18 vs 116 at 8.5).

Rule honored: RAM was not enlarged merely to improve hit rate — the
recommendation stays inside the conservative 32 GB envelope with margin.
