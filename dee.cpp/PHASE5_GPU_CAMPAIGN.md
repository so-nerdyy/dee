# Phase 5 — GPU Campaign Preflight (cohort serving / dee-serve v0)

Branch `research/phase4-cache-hierarchy`. Commits under test:
`c6ae477` (W1 cohort execution) + `3e93e98` (W2 serve driver/runner/session
driver) + `e97dbd6` (W3 verifier). All numbers marked SIM =
`tools/phase4/p4_replay_sim.py --cohort` over the committed Phase-4 journals
(reproduces the a2_fp4 live counters tol=0 at K=1); P4 = sealed Phase-4
evidence bundle.

## What this run is for (and is not)

Phases 0-4 established that exact bounded-memory inference works. This run
does NOT re-ask "is it possible" — it validates the dee-serve v0 execution
path on hardware and measures the serving-economics curve the W0 sim
predicted. The deliverable is a system property: cohort execution that
provably preserves per-request outputs, plus the residency/store-read
scaling law live.

## Hypotheses (predeclared)

- **H1 (exactness under batching):** every cohort row produces a token
  stream byte-identical to the same padded input run sequentially (c1
  reference), and each row's extracted route stream equals the c1
  singleton journal record-by-record. This is the load-bearing gate —
  the native engine's per-expert token grouping + shared-cache path must
  not perturb any individual request.
- **H2 (VRAM cliff):** pooled VRAM residency falls with K per the sim's
  fixed-capacity law — on 3.5 GiB fp4 (281 slots): K=1 ~45-48%, K=2 ~35-40%,
  K=4 ~0%, K=8 ~0% (SIM table below). The per-sequence reuse distance
  (~258 uniques) sits just under capacity; concurrency multiplies the
  working set past the cliff.
- **H3 (host-tier rescue direction):** the enlarged host pack in c8h
  raises host-hit fraction vs the same-K arm at baseline capacity — the
  serving tier is host RAM, not VRAM. (The full 95% host-hit point needs
  ~64 GiB/device — outside the ~29 GiB session; c8h proves direction.)
- **H4 (cross-campaign determinism):** c0_anchor's unpadded q0 stream
  reproduces the committed Phase-4 a2_fp4 q0 token sha — proving the
  rebuilt stack (W1/W2) preserves Phase-4 semantics end to end.
- **H5 (dedup is real but bounded):** cross-request expert dedup ratio
  per forward lands near the sim's ~26% at K=8 (measured as unique
  experts staged / K*topk request slots) — shared residency exists but
  cannot beat the capacity law.

## Arm matrix (as implemented in phase5_session_driver.py)

| arm | mode | groups | pad_to | host pack | purpose |
|-----|------|--------|--------|-----------|---------|
| c0_anchor | sequential b=1 | q0 only, unpadded | — | 8.5 GiB/GPU | P4 determinism anchor |
| c1 | cohort K=1 x8 | [[0]..[7]] | "max" | 8.5 GiB/GPU | padded sequential reference + K=1 point |
| c2 | cohort K=2 x4 | [[0,1],[2,3],[4,5],[6,7]] | "max" | 8.5 GiB/GPU | K=2 point |
| c4 | cohort K=4 x2 | [[0..3],[4..7]] | "max" | 8.5 GiB/GPU | K=4 cliff point |
| c8h | cohort K=8 x1 | [[0..7]] | "max" | ~10.5 GiB/GPU | K=8 + host-rescue direction |

All arms: dee4_segmented store, fp4 residency, lru eviction (the winning
a2 config), lanes=4, qdepth=6, cold reset per unit, N_TOKENS=128,
NATIVE_IGNORE_EOS=1, RUN_ID constant.

Padding contract: cohort rows are left-padded to the workload-global L*
(pad_to="max"); c1 sees the same padded inputs, so cohort-vs-c1 compares
identical model inputs. c0 alone runs the unpadded prompt so it can
anchor to P4's committed (unpadded) sha.

## Metrics (per cohort unit)

- Exactness: per-row token_ids_sha256 vs c1 same-prompt sha; journal
  record_count/token_rows canonicality; extracted per-row route streams
  vs c1 (verify.py); chain_sha256 recompute; checkpoint fan-out.
- Cache: VRAM resident/cold/evictions, host pack hits/misses, store
  source_reads/bytes — cohort-scoped deltas (counter snapshot at cohort
  start subtracted), process-cumulative alongside.
- Sharing: dedup_stats — unique experts staged per (step,layer) vs
  K*topk request slots; per-forward staged-unique series.
- Perf: cohort wall, per-forward decode timings, emitted-token
  amortization (storage bytes / K*N), H2D bytes.
- Memory: host RSS, pack occupancy, VRAM budget compliance.

## Predictions (SIM, fp4+lru, 3.5 GiB VRAM/GPU, cold per cohort)

| K | VRAM resident | host hit (8.5 GiB/dev) | store reads/request |
|---|---------------|------------------------|---------------------|
| 1 | ~48% | ~66% | ~11,900 |
| 2 | ~38% | ~56% | ~14,800 |
| 4 | ~0% | ~41% | ~18,800 |
| 8 | ~0% | low | ~25,200 |

c8h (larger host pack): VRAM stays ~0% (cliff is VRAM-side); host hit
rises vs the same-K baseline arm — direction, not magnitude, is the gate
(session RAM << the 64 GiB/device sim point that hit 95%).

## Accept / reject criteria

- **PASS-minimum:** all cohort rows ACCEPT_CORRECTNESS + every row sha ==
  c1 same-prompt sha (all cohort arms) + c0 sha == P4 a2 q0 + journal
  chain/order/token_rows audits pass in verify.py. Residency ordering
  c1 > c2 > c4 (monotone decrease per H2, tolerances noted below).
- **STRONG:** H2 ordering AND magnitudes within ~10pp of sim (K=1 ~45%,
  K=4 ~0-5%); c8h host-hit > its baseline-capacity counterpart by a
  measurable margin; dedup ratio at K=8 within 5pp of sim (~26%).
- **REJECT-EXACTNESS:** any row token/route mismatch vs c1, journal
  chain break, wrong token_rows, lookup_failures != 0 => arm FAIL,
  results quarantined, cohort path considered broken.
- **REJECT-SIM:** residency INCREASES with K or stays flat ~45% through
  K=8 => the capacity law is wrong; revisit the model before designing
  admission control on it.
- **INFORMATIVE-NULL:** K=8 shows 0% VRAM AND ~0% host hit => on this
  hardware cohort serving is store-bound regardless; document honestly,
  Phase-5 conclusion becomes "serialize on T4-class hosts".

## Why GPU is required (local can't answer)

The CPU tests prove scheduling/exactness semantics on the fp32-direct
path; they cannot measure the native fp4 batched-GEMM path's bitwise
behavior under tokens=K, the real cache-tier behavior under cohort load,
wall/H2D, or the host pack's live capacity. The P4 journals exist but a
live K-row forward's journal does not.

## Budget + mechanics

- T4x2, one Kaggle session (the AUTHORIZED Phase-5 run). Rough wall
  ESTIMATE: c0 ~12min, c1 ~90min, c2 ~70min, c4 ~80min, c8h up to ~2.5h
  (thrash), + build/seal ~30min => ~6-7h. ARM_TIMEOUT_S=3h with partial
  harvest (TIMEOUT arms still yield counters/journals up to the kill).
- Driver: phase5_session_driver.py (P0 gate -> build -> mount+seal ->
  arms -> verdict). Artifacts per cohort into p5-out/{arm}/.
- Abort rules identical to Phase 4: runner nonzero exit / REJECT_
  classification -> harvest, record FAIL, continue; always exit 0.
- Commit pin: set P5_PINNED_COMMIT at launch (runner records `commit` in
  every artifact regardless).

## Known limitations / non-goals this run

- Equal-length cohorts only (left-pad to global L*); ragged admission
  and per-row positions are P5b scope.
- c8h's host pack is the largest the session can carry (~21 GiB total),
  NOT the sim's 64 GiB/device full-rescue point — it validates direction.
- FUSE storage wall times are environment-shaped; the residency/sharing
  ratios are the transferable numbers.
- Prefill records carry token_rows=K*L* (batched prompt tokens) — the
  verify.py extractor slices them per row; journal file shas are NOT
  cross-arm comparable by construction (token_rows differ), row-level
  route streams are.
