# Phase 5 — Serving Path Results (dee-serve v0)

Branch `research/phase4-cache-hierarchy`. Campaign kernel
`nivind/dee-cpp-dsv4-phase5-cohort` @ `9d6d403` (16 units, 2xT4, ~4h52m).
Mechanism/forensic kernels `nivind/dee-cpp-dsv4-p5b-mechanism` v1–v6.
Evidence: `p5-kaggle-out/` (campaign, 3.85 GB), `p5b-v{2..6}-out/`
(mechanism arms), `P5_KAGGLE_WATCHER_REPORT.md`, `P5_EXACTNESS_FORENSICS.md`,
`P5B_MECHANISM_REPORT.md` (root cause + fix).

## What Phase 5 was for

Phases 0–4 established that exact bounded-memory inference works on one
request at a time. Phase 5 answers the serving question: can one process
run *repeated and concurrent* requests through the shared
NVMe→RAM→VRAM hierarchy while preserving per-request outputs bit-exactly —
and does hardware demand scale with the working set, not the checkpoint?

## Verdict

**Serving machinery: PROVEN. Warm-process exactness defect: FOUND, ROOT-CAUSED,
FIXED (`cccf17f`), VALIDATED (v6). Exactness contract: RESOLVED and evidenced
(P5c v2) — singleton bitwise, cohort numerical-equivalence.**

| Component | Result |
|---|---|
| Cohort execution (K=1..8) | 16/16 units ACCEPT_CORRECTNESS, zero infra failures |
| Cross-process determinism | c0 anchor == sealed P4 sha; fresh processes reproduce each other bit-exactly across kernels/VMs |
| Cross-request dedup | real — dedup_ratio 1.21 (K=2) → 1.85 (K=4); ~27% fewer unique staged experts at K=8 |
| Warm-process exactness | **was broken** — every unit ≥1 after `clear_host_cache` served stale/zero expert bytes |
| Root cause | `clear_host_cache` zeroed `fp4_region_nbytes` → zero-length gathers → stale pinned-slot bytes shipped to VRAM |
| Fix | preserve record geometry on clear + restore on reused staging entries + `staging_region_bytes` tripwire |
| Post-fix validation | v6 micro suite — see below |
| Cohort re-run (v2, fixed build) | corruption gone (c1 stable, c0 anchored); residual = batch-shape numerics, not a defect |
| Exactness contract | K=1 bitwise; K>1 = exact bytes + deterministic numerical equivalence — evidenced by P5c tensor bisect |

## The defect that failed the exactness gate

The campaign's 24 `row == c1 exact` failures — and the instability of the
c1 reference itself (c1-c0 vs c1-c7 diverged on byte-identical input) —
were all one bug, not numerics:

1. `Engine::clear_host_cache()` (cold reset between units) nulled each
   persistent `staging_int8_` entry's region pointers **and zeroed
   `fp4_region_nbytes`** — record geometry that is immutable for a given
   store format.
2. On the next fill, `prepare_fp4_experts()` found the *existing* entry
   and skipped `configure_fp4_quantized` — the only writer of
   `fp4_region_nbytes`. `point_fp4_regions` stamped six `{ptr, 0}`
   regions.
3. `AsyncPrefetcher::cuda_submit`'s pinned-slot gather memcpy'd **zero
   bytes**; the H2D then shipped whatever the slot last held — the last
   two fills' records — into the device block. Blocks that never got a
   slot stayed all-zero.
4. The transfer path reported success; the device cache marked the bad
   block resident. Every subsequent read of that expert consumed wrong
   bytes. Corruption accumulated per reset (v5: dev_zero 42→54→58→62
   across successive cold-reset iterations).

Signature proof (v5 byte fingerprints, `0dd39ef`): `pack_sha == store_sha`
on every expert in every arm (host fill innocent); `dev_sha` mismatched on
960/1024 fp4 blocks — wrong blocks held *exactly* expert 62's or 63's
record (the last two staged) or were all-zero. Identical under
`CUDA_LAUNCH_BLOCKING=1` (not a race) and under fp16 cache (not
FP4-decode-specific). First iteration always clean — entries don't exist
yet, so fresh metadata was used.

The forensics doc's cuBLAS hypothesis is **falsified**: this was a
data-integrity bug in the host-cache-invalidation path, not library
nondeterminism. It was also *invisible* to every earlier phase because
nothing before Phase 5 ran two inference units per process with a cold
reset between them.

## Fix (`cccf17f`)

- `clear_host_cache` invalidates pointers + generation only; record
  geometry (`fp4_region_nbytes`, `fp4[p]`, `fp4_total_nbytes`) is
  immutable and now survives the clear — matching the pack-eviction
  observer's existing rule.
- `prepare_fp4_experts`' reused-entry branch restores the geometry from
  the freshly-derived metadata, so even an entry whose fields were
  invalidated any other way self-heals on refill.
- `debug_expert_fingerprint` now exposes `staging_region_bytes` (sum of
  stamped region lengths); the driver tallies `staging_zero_regions` as a
  permanent tripwire for this bug class.
- CPU coverage: `peek_bytes` non-perturbing contract test (LRU order +
  counters untouched) — `6ee7249`, ALL PASS.

## v6 validation — PASS (kernel v6 @ `6ee7249`, includes `cccf17f`)

`P5B VERDICT: PASS`. Preflight all PASS (clone, cmake, dee_core,
test_dee4_segmented, pydee, 46-segment store). Results against the
predeclared acceptance criteria:

| Criterion | Required | Got (mK/mL/mM) |
|---|---|---|
| `dev_eq_pack` per iter (fp4) | 64 | **64** (1024 total) both fp4 arms |
| `dev_zero` | 0 | 0 |
| `dev_ne_pack` | 0 | 0 |
| `staging_zero_regions` | 0 | 0 — `staging_region_bytes` = 13,369,344 everywhere |
| `distinct` raw hash per config | 1 | **1** on all 4 configs — every iteration sha `17fbc848…`, identical to v5's clean iteration-0 |
| mL under `CUDA_LAUNCH_BLOCKING=1` | clean | clean |
| mM under fp16 cache | clean | clean (`dev_zero=0`) |

The convicted boundary — device bytes diverging from pack bytes after a
host-cache clear — is fully closed, and warm-process refills now
reproduce the exact fresh-run output bit-for-bit.

## What this means for dee-local / dee-serve

- **The architecture is right.** The hierarchy, dedup, journals, evidence
  chain, cohort scheduler, and cold resets all worked as designed through
  16 units; the only defect was one invalidated-metadata field. Nothing
  about the failure argues against the NVMe→RAM→VRAM design.
- **Exactness under warm-process serving is now a tested invariant,** not
  an assumption: the fingerprint instrument (three-boundary hashing per
  expert per iteration) is a permanent part of the mechanism suite, and
  the zero-length-gather tripwire fails loudly instead of silently
  shipping stale bytes.
- **Serving economics on T4-class hardware are honest:** host-pack-bound
  ~0.2–0.3 tok/s aggregate; K=8 dedup cut unique staged experts ~27% but
  could not beat the VRAM capacity law (~258-expert working set ≫ 281
  slots at K≥4). The sim's prediction that host RAM is the serving tier
  held; the full 95%-host-hit point needs ~64 GiB/device — deferred to
  hardware follow-on, by design.

## Cohort re-run (v2) + exactness contract — resolved

The post-fix campaign re-run (`dee-cpp-dsv4-phase5-cohort` v2, ~7.5h)
removed the corruption noise entirely: c1's 8 warm units stable
(c1-c7 == c1-c0 bitwise), c0 anchored, all machinery checks green,
clean dedup curve K=1 1.15 → K=8 1.61.  The residual 22/24
`row == c1` gate failures were then isolated to **batch-shape floating
point**, not a defect: the P5c isolation kernel
(`dee-cpp-dsv4-p5c-iso` v2) dumped per-(step,layer) tensors for the
same padded prompt at K=1 vs inside a K=8 cohort and measured:

- **Injection point:** layer-0 attention GEMM output (max_abs 3.1e-2)
  with bitwise-identical inputs — different GEMM row shapes (K·L* vs
  L*) select different kernels/reduction orders.
- **Amplification:** router-score perturbation grows ~400× over depth
  (5e-4 → ~0.2); expert-id flips begin at l3 (first learned router),
  ~20/layer at the saturated tail; hash-routed l0-2 cannot flip.
- **Boundary causality:** 92.5% of flips have swap-expert score delta
  ≥ the rank-6/7 margin; 97.7% are single-expert swaps — flips occur
  where the measured perturbation can cross the gap.
- **Determinism:** two identical K=8 units reproduced all 503 dumped
  tensors, 344 journal records, and 8 token shas bitwise.
- **Tokens:** member0 and member5 emitted bit-identical tokens to
  their K=1 singletons in the 8-token window — routing flips stayed
  sub-argmax; over the 128-token campaign horizon, crossings
  accumulate and streams legitimately bifurcate.

Full detail: `P5B_MECHANISM_REPORT.md` §P5c + `p5c_analysis.json`.

**Final contract (user-directed):** K=1 remains bitwise
reference-equivalent in tokens and authoritative routing.  K>1 cohort
serving holds: exact expert/store/host/device bytes, no
stale/zero/cross-request contamination, same-shape bitwise
reproducibility, and characterized near-tied-boundary routing flips —
but not bitwise batch-invariance.  Batch-invariant kernels remain a
possible future research direction, not a Phase-5 requirement.

## Remaining scope (not done here)

- Ragged cohort admission / per-row positions (P5b scope, deferred).
- `dee-local` packaging (SSD-heavy minimum-hardware profile) and
  `dee-serve` (global expert caches, continuous batching, admission
  control on the measured capacity law) — engineering build-out, now on
  a validated exact base.
