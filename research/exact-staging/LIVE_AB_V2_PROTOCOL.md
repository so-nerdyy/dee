# LIVE_AB_V2_PROTOCOL.md — the ONE next A/B: pack cap 17 → 20 GiB total

Branch: `research/exact-staging` · Date: 2026-09-04 · Prepared for Codex/Astra.
Machine-readable: `results/next_ab.json`. All expected deltas are SIMULATED
(recalibrated model + validated replay); the A/B is the only arbiter.
Performance acceptance only under the formal campaign rules.

## 1. Hypothesis

The production host-pack LRU (8.5 GiB/GPU, 682 records — the `LRU_TOTAL_CAP_GIB=17.0`
clamp) evicts 116 decode-demanded record-slots per run that a slightly larger
pack would retain. Raising the cap to **20 GiB total (10.0 GiB/GPU, 803
records)** keeps 51 of those misses from re-reading 13.37 MB from the store:
replay says **1,252 → 1,200 storage misses**. If `miss_service_ms` is real
wall service, decode wall drops ~2.5 s (central estimate, ±5.6% statistical
on the miss term; baseline run spread ~1.0–1.9 s).

Why 10.0 and not neighbors:
- 9.5 GiB/GPU captures 42/51 of the effect with 1.13 GB more headroom —
  pick this if you want the conservative memory variant (one knob, same
  protocol; it is the risk-balanced default in MEMORY_BUDGET.md).
- 10.5+ crosses the 2 GB-headroom gate (BORDERLINE) and buys only ~11 more
  misses/GiB.
- 12.75 is REJECTED for the 32 GB thesis (NOT_SAFE; MEMORY_BUDGET.md §4).

## 2. Exact configurations

| | Baseline | Candidate |
|---|---|---|
| `LRU_TOTAL_CAP_GIB` (harness) | 17.0 | **20.0** |
| effective pack/GPU | 8.5 GiB (682 rec) | 10.0 GiB (803 rec) |
| everything else | identical | identical |

Keep identical: prompt, `n_tokens=16`, route authority, `cache_budget 3.5
GiB/GPU`, `source_read_lanes/queue_depth`, reuse flag, madvise/discard
settings, correctness gates, hardware (2×T4), kernel version class.
Implementation: the harness clamps requested bytes by the cap
(`deepseek_v4_native_generate.py:1347-1361`); set the cap only — requested
12.87 GiB/GPU stays as-is and the clamp distributes symmetrically.

## 3. Required live metrics (both arms)

- decode wall, decode TPS, total wall, ITL p50/p95
- memory: per-token VmRSS/VmData checkpoints, final VmHWM/RSS,
  min checkpoint MemAvailable, MemTotal
- host_pack per GPU: entries, bytes, hits, misses, evictions
- per-step `storage_requests` (SSD reads) and `h2d_copies`
- all sealed correctness gates + ids/text equality
- run hashes: harness, source, run_config, journals

## 4. Success / falsification (pre-registered)

Success (both required):
1. Correctness: both arms ACCEPT_CORRECTNESS, identical ids+text.
2. Performance: candidate decode wall < baseline decode wall by ≥1.0 s
   (baseline run spread), **replicated** in a second matched pair; per-step
   SSD reads reduced ~51 (±few, matching replay per-GPU ~27/24 split).
   Formal performance acceptance is a separate campaign decision.

Falsifies the host-pack hypothesis:
- pack misses fall as replayed but decode wall does not improve beyond
  run noise in two matched pairs (⇒ `miss_service_ms` is not wall
  service on this path), OR
- memory exceeds the projected envelope (projected total 27.227 GiB =
  29.23 decimal GB at the 10.0 GiB/GPU candidate — see the generated table
  in MEMORY_BUDGET.md; abort if min checkpoint available < 1.5 GiB), OR
- any correctness divergence (stop immediately).

Pair count and stopping are **pre-registered** in AB_EXPERIMENT_DESIGN.md:
run exactly 2 matched pairs; futility (abort after pair 1) if the candidate
wall exceeds its baseline by more than the measured baseline SD (0.926 s);
no early accept on a favorable pair 1. Success statistics (both deltas
negative AND mean delta ≥ baseline MAD 0.905 s) come from
`results/ab_noise.json` methodology.
