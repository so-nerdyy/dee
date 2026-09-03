# ULTRALOWBIT_STORAGE_REPORT — DeepSeek-V4-Flash on a consumer machine

Branch: `research/ultralowbit-storage` — parallel research track.
**No changes were made to the MXFP4 runtime, its semantics, or any sealed
evidence.** All simulation consumed sealed artifacts read-only.

Companion audit: `research/ultralowbit/SHERRY_AUDIT.md` (verified vs inference
vs hypothesis tags for every upstream claim).
Tools: `tools/expert_hierarchy_sim.py`, `tools/run_sweep.py`,
`tools/mixed_precision_optimizer.py`, `tools/stq_codec_proto.py`.
Raw results: `research/ultralowbit/results/` (sweep_table.csv 421 rows,
phase_d_required_hit_rates.json, 3 heatmaps, sweep_results.json).

> **Every TPS number in this report is a STORAGE-ONLY CEILING**: the token rate
> at which the storage/PCIe hierarchy can *deliver* expert bytes. It excludes
> compute, attention, kernel, scheduling, and dequant costs. It is **not** an
> end-to-end TPS prediction.

---

## Inputs (all sealed / measured — nothing estimated ad hoc)

| Input | Value | Source |
|---|---|---|
| Route trace | 688 records, 16 forward steps (1 prefill + 15 decode), 5,099 grouped expert requests | `v62-terminal-evidence-20260902T003709Z/routed_experts.jsonl` (journal sha `665aac3e…`, run `p2.3-route-journal-v50`) |
| Per-expert MXFP4 record | **13,369,344 B (12.75 MiB)** = FP4 payload 12 MiB + e8m0 scales 0.75 MiB, group 32 | `v62…/dee4-metadata.json` component table |
| Logical weights per expert | 25,165,824 (w1 [2048,4096] + w3 [2048,4096] + w2 [4096,2048]) → **4.25 effective bpw** current | same |
| Trace store subset | 2,364 selected experts, 31.61 GB (29.44 GiB) | `dee4-metadata.json` `total_bytes` |
| Full model experts | 43 layers × 256 = 11,008 → 137.0 GB MXFP4 | config (43 layers, 256 experts) × record size |
| v62 live counters (2×T4, 3.5 GiB/GPU device cache) | cuda0: 328 resident hits, 2,285 cold loads, 2,004 evictions, 30.5 GB H2D | `v62…/native-generate-result.json` |
| Simulator validation | 5,099 simulated requests vs 5,101 staged requests sealed in P2.3 replay; VRAM hit @8 GiB/GPU-equivalent budgets consistent with P2.3 replay (50.3% LRU / 53.6% Belady at 8 GiB) | P2.3_FP4_VRAM_ANALYSIS.md |

Simulator geometry notes mirror the live scheduler: duplicate expert IDs
grouped once per layer; prefill group-loads serve all token rows; two-tier
VRAM-over-RAM; decode-phase steady-state metrics reported separately from
one-time warmup staging.

---

## Q1. Current MXFP4 bytes/expert

**13,369,344 bytes (12.75 MiB) per routed expert** — sealed value, not an
estimate. Effective 4.25 bits/weight (4.0 FP4 + 0.25 e8m0 scale overhead at
group size 32). Full 11,008-expert working set: **137.0 GB**; the sealed
trace-store subset (2,364 experts): **31.61 GB**.

## Q2. Plausible STQ bytes/expert

**4,128,768 bytes (3.9375 MiB)** at **1.3125 bpw** total:
- payload: 25,165,824 w × 1.25 b / 8 = 3,932,160 B (5 bits per 4-weight group,
  3:4 structured sparsity, 32 permutations saturating the 5-bit code);
- scales: 98,304 blocks × 2 B fp16 (block 256) = 196,608 B (+0.0625 bpw).

Format verified from Sherry paper (arXiv:2601.07892) + llama.cpp STQ1_0 block
(`qs[32] + sign[8] + d` = 42 B per 256 w). "Plausible" caveats: the *format*
is verified; applying it to DeepSeek-V4-Flash experts is **idealized** — no
public checkpoint exists and quality is unresolved (see Q11). IQ2_XXS
equivalent: 6,488,064 B (6.1875 MiB) at 2.0625 bpw.

## Q3. Cache capacity multiplier

| Budget | MXFP4 experts | STQ1_0 experts | IQ2_XXS experts | Multiplier |
|---|---|---|---|---|
| 2 GiB VRAM | 157 | 508 | 323 | ×3.24 / ×2.06 |
| 4 GiB VRAM | 314 | 1,017 | 647 | ×3.24 / ×2.06 |
| 6 GiB VRAM | 471 | 1,525 | 970 | ×3.24 / ×2.06 |
| 8 GiB VRAM | 628 | 2,033 | 1,294 | ×3.24 / ×2.06 |
| 12 GiB RAM | 942 | 3,050 | 1,941 | ×3.24 / ×2.06 |
| 16 GiB RAM | 1,257 | 4,066 | 2,588 | ×3.24 / ×2.06 |
| 20 GiB RAM | 1,571 | 5,083 | 3,236 | ×3.24 / ×2.06 |

(Exact divisors: 13,369,344 / 4,128,768 / 6,488,064 bytes per expert.)

## Q4. SSD bytes/token reduction (decode steady state)

At matched generous budgets (8 GiB VRAM + 16 GiB RAM, warm-started, LRU):

| Codec | SSD GB/decode-token | vs MXFP4 |
|---|---|---|
| MXFP4 | 1.11 | 1× |
| IQ2_XXS | 0.49 | 2.26× |
| STQ1_0 | 0.31 | **3.55×** |
| MIXED 10/30/60 | 0.42 | 2.62× |

STQ reaches the trace's compulsory-miss floor: 1,135 decode-touched experts
were never seen in prefill, so ≥ 1,135 × 3.94 MiB = 4.47 GB of one-time
traffic is unavoidable *within this trace*; with RAM ≥ 12 GiB those experts
stay resident and steady-state SSD traffic → ~0 on longer runs. The 0.31 GB
value is that floor amortized over the 15-token window; the MXFP4 floor would
be 15.1 GB (unreachable at these budgets), hence the 3.55× gap is a **lower
bound** on the long-run SSD reduction.

## Q5. H2D (PCIe) bytes/token reduction

Same budgets: MXFP4 **1.61 GB/token** → STQ **0.40–0.60 GB/token**
(**2.7–4.0×**), IQ2 0.78–1.10 GB/token (1.5–2.1×), MIXED 0.48 GB/token
(3.3×). STQ also enables *narrower* VRAM caches (2–4 GiB reaches the same
hit rate as 8 GiB MXFP4), which is what frees the 5070 Ti's VRAM for KV and
dense state.

## Q6. Effect on the RTX 5070 Ti 16 GB geometry

Persistent dense state ≈ 8.84 GB (compressed, from the sealed ledger in
T4_FEASIBILITY.md; the same checkpoint weights apply on any GPU), leaving
~6–7 GB of VRAM for everything else in a single-GPU configuration. With
bounded expert caches of 6–8 GiB:
- MXFP4: 471–628 experts resident — far below the 1,135-expert decode cold
  set; per-token H2D stays ~1.6 GB → PCIe becomes the binding constraint
  above ~10 TPS even at 16 GB/s.
- STQ1_0: 1,525–2,033 experts resident in the *same* VRAM — covers the full
  prefill-warmed hot set; H2D drops to ~0.4–0.6 GB/token, and the VRAM cache
  could be *shrunk* to 2–4 GiB with no SSD-traffic penalty, returning 4–6 GiB
  to KV cache / dense / MTP.

## Q7. Effect on the 32 GB RAM geometry

The MXFP4 trace subset (31.61 GB) **does not fit** in 32 GB RAM alongside the
OS, app, and CUDA host staging (v62's own host process peaked at 430 GiB
VmPeak on the Kaggle host, but a 32 GB consumer box has ~24–26 GB usable).
Practical MXFP4 RAM caches are ≤ 20 GiB → 1,571 experts.
STQ1_0 shrinks the same subset to **9.77 GB** — it fits with >12 GiB of
headroom. The entire prefill+decode working set (2,364 experts, 9.77 GB)
becomes RAM-resident after a one-time 9.77 GB sequential read (~3–10 s on a
consumer NVMe), after which SSD traffic tends to zero. This is the single
biggest structural change: **expert server → RAM-resident server**.

## Q8. Storage-only roofline at 1/2/5/10/20 TPS

Required sustained SSD bandwidth = (SSD GB/decode-token) × TPS
(best-budget rows; MIXED at 8 GiB VRAM + 16 GiB RAM):

| Codec | SSD GB/tok | 1 TPS | 2 TPS | 5 TPS | 10 TPS | 20 TPS |
|---|---|---|---|---|---|---|
| MXFP4 | 1.11 | 1.11 | 2.22 | 5.55 | 11.1 | 22.2 |
| IQ2_XXS | 0.49 | 0.49 | 0.98 | 2.45 | 4.91 | 9.82 |
| STQ1_0 | 0.31 | 0.31 | 0.62 | 1.55 | 3.10 | 6.21 |
| MIXED | 0.42 | 0.42 | 0.84 | 2.11 | 4.23 | 8.46 |

Corresponding **STORAGE-ONLY CEILING TPS** (max over swept budgets; PCIe
capped at 16 GB/s; consumer NVMe columns):

| Codec | 1 GB/s | 2 GB/s | 3.5 GB/s | 5 GB/s | 7 GB/s | 10 GB/s |
|---|---|---|---|---|---|---|
| MXFP4 | 0.90 | 1.80 | 3.15 | 4.51 | 6.31 | 9.01 |
| IQ2_XXS | 2.04 | 4.07 | 7.13 | 10.19 | 14.26 | 20.37 |
| STQ1_0 | 3.20 | 6.40 | 11.20 | 16.00 | 22.41 | 32.01 |

(Heatmaps: `results/heatmap_{MXFP4,IQ2_XXS,STQ1_0}.png` — VRAM × SSD grids at
best RAM budget. All labeled STORAGE-ONLY CEILING.)

## Q9. Cache hit rate required at each target

Required combined RAM+VRAM **avoidance rate** (share of expert requests
served without an SSD read) so that SSD traffic fits the bandwidth budget
(`results/phase_d_required_hit_rates.json`; requests per decode token =
258; per-expert bytes from Q1/Q2):

At 3.5 GB/s SSD:

| Target TPS | MXFP4 | IQ2_XXS | STQ1_0 |
|---|---|---|---|
| 5 | 79.7% | 58.2% | 34.3% |
| 10 | 89.9% | 79.1% | 67.1% |
| 20 | 94.9% | 89.6% | 83.6% |

At 10 GB/s SSD:

| Target TPS | MXFP4 | IQ2_XXS | STQ1_0 |
|---|---|---|---|
| 5 | 42.0% | 0% (any) | 0% (any) |
| 10 | 71.0% | 40.3% | 6.1% |
| 20 | 85.5% | 70.1% | 53.1% |

Simulated **achievable** combined hit rate at 8 GiB VRAM + 16 GiB RAM
(decode-phase, warm-started LRU): MXFP4 **65.6%**, IQ2/STQ **70.7%** (the
trace's compulsory floor — 2,735 of 3,870 decode requests touch experts
already seen in prefill). Belady decode-only warm ceiling with the same
total capacity: **70.7%** (capacity no longer binding once RAM ≥ 12 GiB under
STQ; under MXFP4, Belady at 8 GiB VRAM only = 53.6% vs LRU 40.4%).

Read directly: **20 TPS under MXFP4 needs 85.5% avoidance at 10 GB/s but LRU
delivers 65.6% — capacity-bound, not policy-bound (Belady 70.7% confirms).**

## Q10. Is 20 TPS physically plausible?

- **MXFP4: physically implausible on this geometry.** Even a 10 GB/s NVMe
  caps at 9.0 TPS; the required 85.5% avoidance exceeds what the trace's
  reuse structure supports at feasible capacities (LRU 65.6%, oracle 70.7%).
- **IQ2_XXS: marginal.** 20.4 TPS ceiling at 10 GB/s (needs 70.1% avoidance;
  achievable 70.7% — zero margin, any routing drift breaks it). Comfortable
  at 7 GB/s for 14.3 TPS-class targets.
- **STQ1_0: plausible as a storage problem.** 32.0 TPS ceiling at 10 GB/s;
  20 TPS needs only 53.1% avoidance vs 70.7% achievable — 17.6 points of
  headroom; even a 5 GB/s NVMe yields a 16 TPS ceiling and 7 GB/s yields
  22.4. **MIXED 10/30/60: 23.6 TPS at 10 GB/s** — plausible with a smaller
  quality bet.
- Caveats: these are delivery ceilings. End-to-end 20 TPS additionally
  requires a ternary/IQ2 GEMV path fast enough on consumer GPUs (no CUDA
  kernel exists today — Phase F shows the format decodes at 13 M weights/s
  vectorized on CPU; a 5070 Ti-class kernel is unmeasured), dense+attention
  headroom, and KV capacity. Nothing here measures those.

## Q11. What would require QAT

1. **Shipping STQ1_0 (1.3125 bpw) experts at acceptable quality on this
   model.** Sherry's published accuracy is QAT-only: 10B tokens, STE, and the
   Arenas annealing bypass; its own ablation shows naive 3:4 ternary training
   *collapses* without Arenas. No public calibration-only (PTQ) variant of
   Sherry/STQ1_0 exists (verified: AngelSlim docs + PR #22836 contain none).
2. A PTQ attempt on a model whose experts are already MXFP4-quantized is
   **not lossless by construction**: e2m1 has 5 magnitudes per exponent;
   ternary ±d per 256-block collapses them; the paper's PTQ-style baselines
   at 1.67 bpw already lose 7 points on 1B models (AbsMean 0.489 vs BF16
   0.558).
3. Not-QAT items: codec *assignment* policy (which experts get which codec),
   cache policy, and format plumbing are all inference-time work. A
   **calibration-only** middle path (magnitude projection + per-block scale
   fitting + GPTQ-style Hessian compensation, evaluated per expert) is
   untested for this model and is the cheapest thing that could retire the
   QAT requirement — see Q12/Q13.
4. IQ2_XXS at 2.0625 bpw is the conservative tier: mainline llama.cpp PTQ
   ecosystem exists, but DeepSeek-V4-Flash-specific quality is still
   unmeasured here (no quality data was fabricated in this branch).

## Q12. What could be tested immediately

1. **Single-expert PTQ probe (no QAT, no training):** take one real expert
   (e.g. layer-6 expert-0 from shard model-00008, per CHECKPOINT_DOWNLOAD_PLAN),
   apply magnitude-projection STQ1_0 and IQ2_XXS, dequantize, run the existing
   DS7 FP32-reference comparison (`scripts/deepseek_v4_expert_reference.py`)
   and record max/mean/p99 relative error + SwiGLU-output MSE per codec.
   Costs one Kaggle session; needs no new kernels.
2. **Decode throughput microbenchmark of the Phase F decoder** ported to a
   CUDA scaffold (isolated, never merged into Engine): measures unpack+GEMV
   feasibility on T4 (sm75) and estimate for consumer GPUs.
3. **Consumer NVMe QD4 random-read at 3.9 MiB / 6.2 MiB / 12.75 MiB record
   granularity** on the actual target machine — the sweep assumes 1–10 GB/s
   sustained; expert-record reads are the real access pattern.
4. **Longer route traces** (thousands of decode steps across prompts/domains)
   to replace the 15-decode-token compulsory-floor estimate with a stable
   reuse-distance distribution.
5. **Admission-policy A/B in the live engine** (2-hit admission vs pure LRU)
   using the existing harness knobs — zero new dependencies.

## Q13. Exact next experiment Codex should run later

**`DS-STQ-1 — single-expert ultra-low-bit projection error measurement.**

- **Setup:** Kaggle dual-T4 (existing kernel lineage), commit pinned to the
  current `freebuff/deepseek-v4-flash-0731-t4` HEAD; download shard
  model-00008-of-00048 (contains `layers.6.ffn.experts.0.*`) via the existing
  verified downloader.
- **Procedure:** (a) FP32 reference forward of that expert on N=512 random
  hidden states using `deepseek_v4_expert_reference.py`; (b) quantize the
  same weights to (i) magnitude-projection STQ1_0 (Phase F encoder,
  per-block fp16 scale), (ii) IQ2_XXS-equivalent codebook quant, (iii) MXFP4
  control (round-trip through the official FP4 table); (c) dequantize and
  forward; (d) record max/mean/p99 rel error, output MSE vs FP32, and the
  same gates DS7 used (max_abs 2.0, mean_rel 0.01, p99_rel 0.05).
- **Accept/fail:** *report-only* — no accept gate on quality; the decision
  gate is a later milestone. MXFP4 round-trip must match DS7's tolerance
  (sanity check on the harness itself).
- **Output:** one JSON evidence bundle + a table mapping codec → error
  profile, which becomes the first *real* sensitivity data point for the
  Phase E optimizer (replacing the MEMORY-ONLY SIMULATION label).
- **Estimated cost:** one Kaggle session (< 1 h), no model-wide download.
- **Explicitly out of scope:** any change to the MXFP4 runtime; any quality
  claim about the full model.

---

## Method notes & limitations

- Sim consumes the sealed v62 trace (16 steps). Reuse statistics from 15
  decode tokens are indicative; the compulsory-floor calculation (1,135
  unseen decode experts) is exact for this trace but trace-dependent.
- Warm-start model = perfect warmup (prefill-touched experts pre-staged in
  RAM at one-time SSD cost). Cold-start rows are in `sweep_table.csv`
  (`*_cold`) for comparison.
- MIXED runs are **MEMORY-ONLY SIMULATION**: codec assignment is
  frequency-ranked; no quality/sensitivity data exists in this repo and none
  was fabricated. `mixed_precision_optimizer.py` accepts an explicit
  sensitivity JSON for future budgeted runs.
- PCIe assumption 16 GB/s (conservative for a PCIe 5.0 x16 5070 Ti, whose
  copy bandwidth is likely higher — H2D ceilings in the sweep are therefore
  conservative).
- No Belady-optimal admission was combined with LRU; Belady rows are
  ceilings, not implementable policies.
- Phase F prototype: 6/6 tests pass (round-trip vs fp16-rounded scale, 3:4
  sparsity invariant, 42 B/1.3125 bpw constants, re-encode idempotence, edge
  blocks, numpy equivalence). CPU decode: 13.1 M weights/s vectorized numpy;
  1.6 M weights/s pure-Python reference. Licensing: llama.cpp is MIT and PR
  #22836 is public, but it is **still open/unmerged** — the prototype here is
  a from-spec reimplementation, and no CUDA kernel exists upstream to port
  (PR is CPU/NEON only).
