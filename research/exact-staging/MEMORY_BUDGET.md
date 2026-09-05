# MEMORY_BUDGET.md — full host-memory accounting and the 32 GB verdict

Branch: `research/exact-staging` · Date: 2026-09-04 (unit-contract revision)
Authoritative unit contract: `MEMORY_UNIT_CONTRACT.md`.
Companion machine-readable (SOURCE OF TRUTH):
`results/memory_frontier.json`,
`results/seal_host_reuse/derived_memory_digests.json` (all inputs sha256-gated
to seal `14864034a7354e0e29e11c1c09f18b0863afe6a0`).

> **Revision note (fcc8ca2 → this commit):** the previous revision of this file
> contained a hand-maintained table that disagreed with
> `results/memory_frontier.json` (e.g. 10.0 GiB/GPU shown as 31.36 GB /
> 0.64 GB headroom vs the JSON's 29.23 GB / 2.77 GB). Root cause: the table
> silently added the 2.0-unit safety headroom into the projected totals and
> used a stale conversion. The table below is now **generated from the JSON**
> between sentinel markers and is compared row-by-row against the JSON in
> `tests/test_recalibration.py`; hand-edits to the generated block fail tests.

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

Projection (all terms measured except the explicitly-labeled UNKNOWN
allowance), in GiB:

    projected_total_gib = 2·budget + 5.817 (nonpack HWM max)
                          + 0.91 (system rest) + 0.5 (unmeasured growth, UNKNOWN)

Every memory number carries its unit in its field name. Two authoritative
safety systems are evaluated per row (definitions in
`MEMORY_UNIT_CONTRACT.md`):

- **MEASURED_HOST_MEMTOTAL** — host-execution safety on the actual Kaggle
  2×T4 host (MemTotal 31.35 GiB measured in all 5 seal runs).
  `headroom_gib_to_measured_memtotal` = 31.35 − projected_total_gib, which by
  construction equals `projected_min_MemAvailable_gib`.
- **STRICT_32_DECIMAL_GB** — the project's "~32 GB total host RAM" thesis as a
  hard 32.0 decimal GB envelope (nominal host class).
  `headroom_decimal_gb_to_strict_contract` = 32.0 − projected_total_decimal_gb.

Classification gates (identical in both systems, each in its own unit):
`SAFE_FOR_32GB` requires headroom ≥ 2.0 AND projected min-available ≥ 2.0 GiB
AND margin to the nearest clean OOM observation ≥ 2.0 GiB; failing exactly one
gate while fitting arithmetic → `BORDERLINE_FOR_32GB`; otherwise
`NOT_SAFE_FOR_32GB`. The row's `classification` is the conservative
intersection (worst) of the two systems.

Empirical OOM ledger (harness header comments, earlier campaign runs —
measured on this host class): 25.74 GiB pack total **survived** (v8);
27.68 GiB **OOM-killed** (v9); v11's 26.0 GiB OOM is confounded by madvise
DONTNEED re-fault thrash.

<!-- BEGIN GENERATED:memory-table -->
| Budget GiB/GPU | Pack total GiB | Projected total GiB | Projected total decimal GB | Headroom GiB to measured MemTotal (31.35) | Headroom decimal GB to strict 32 GB contract | Projected min MemAvailable GiB | OOM margin GiB | Class (measured MemTotal) | Class (strict 32 decimal GB) | Combined class |
|---|---|---|---|---|---|---|---|---|---|---|
| 8.5 | 17.0 | 24.227 | 26.01 | 7.12 | 5.99 | 7.12 | 10.68 | SAFE | SAFE | SAFE |
| 9.0 | 18.0 | 25.227 | 27.09 | 6.12 | 4.91 | 6.12 | 9.68 | SAFE | SAFE | SAFE |
| 9.5 | 19.0 | 26.227 | 28.16 | 5.12 | 3.84 | 5.12 | 8.68 | SAFE | SAFE | SAFE |
| 10.0 | 20.0 | 27.227 | 29.23 | 4.12 | 2.77 | 4.12 | 7.68 | SAFE | SAFE | SAFE |
| 10.5 | 21.0 | 28.227 | 30.31 | 3.12 | 1.69 | 3.12 | 6.68 | SAFE | BORDERLINE | BORDERLINE |
| 10.75 | 21.5 | 28.727 | 30.85 | 2.62 | 1.15 | 2.62 | 6.18 | SAFE | BORDERLINE | BORDERLINE |
| 11.0 | 22.0 | 29.227 | 31.38 | 2.12 | 0.62 | 2.12 | 5.68 | SAFE | BORDERLINE | BORDERLINE |
| 11.25 | 22.5 | 29.727 | 31.92 | 1.62 | 0.08 | 1.62 | 5.18 | BORDERLINE | BORDERLINE | BORDERLINE |
| 11.5 | 23.0 | 30.227 | 32.46 | 1.12 | -0.46 | 1.12 | 4.68 | BORDERLINE | NOT_SAFE | NOT_SAFE |
| 12.0 | 24.0 | 31.227 | 33.53 | 0.12 | -1.53 | 0.12 | 3.68 | BORDERLINE | NOT_SAFE | NOT_SAFE |
| 12.25 | 24.5 | 31.727 | 34.07 | -0.38 | -2.07 | -0.38 | 3.18 | NOT_SAFE | NOT_SAFE | NOT_SAFE |
| 12.75 | 25.5 | 32.727 | 35.14 | -1.38 | -3.14 | -1.38 | 2.18 | NOT_SAFE | NOT_SAFE | NOT_SAFE |
<!-- END GENERATED:memory-table -->

## 4. Verdicts

- **Is 12.75 GiB/GPU safe for the 32 GB thesis? NO** under either system:
  projected 32.727 GiB = 35.14 decimal GB > 32.0 strict contract, and only
  2.18 GiB below the nearest measured clean OOM (27.68 GiB pack total), with
  projected min-available **negative** (−1.38 GiB). It was a good research
  point; it is not a production candidate. REJECTED for the main thesis.
- **Largest SAFE point (both systems): 10.0 GiB/GPU** (cap 20 GiB total):
  projected 27.227 GiB = 29.23 decimal GB, headroom 4.12 GiB to measured
  MemTotal / 2.77 decimal GB to the strict contract — passes all gates.
- **Risk-balanced default: 9.5 GiB/GPU** (cap 19 GiB): projected 26.227 GiB =
  28.16 decimal GB, headroom 5.12 GiB / 3.84 decimal GB, and it captures
  ~82% of the SAFE-tier miss reduction (42 of 51 misses) that 10.0 offers.
- **Scaling point (NOT a requirement):** on 48/64 GB hosts, 12.75 GiB/GPU
  becomes acceptable (35.1 GB < 48 GB with margin) and reaches the
  offline floor region (capacity misses 18 vs 116 at 8.5).

Rule honored: RAM was not enlarged merely to improve hit rate — the
recommendation stays inside the conservative envelope with margin under
BOTH safety systems.
