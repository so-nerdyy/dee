# Live pipeline readiness

Verdict: **LIVE_PIPELINE_READY = true** (mock end-to-end green; T4 values
still required). Terminal classification stays **BLOCKED_LIVE_GPU** — the
two statements mean different things (pipeline proven vs GPU absent).

## Gate checklist (all must hold; all hold)

1. **Mock end-to-end passes**: 7/7 scenarios through `run_evidence.py`
   (`test_mock_campaign_e2e.py`).
2. **Every promised artifact populated**: attribution (8 records), CSV
   (header + 8), per-token (2 tokens), closure, ABC, ranking, next-ab,
   bundle (`test_scenario_1_*` asserts each file non-empty and consistent).
3. **Ranking from parsed evidence**: `rank_from_evidence()` consumes only
   ingested closure + mechanics; UNMEASURED appears only when fields are
   absent (scenario 4/5 paths tested).
4. **next-ab derives from ranking**: single-candidate priority rule +
   empty case; no hardcoded winner (`test_next_ab_single_and_empty`).
5. **Malformed evidence fails closed**: scenarios 6/7 exit 1 with
   INVALID_PROFILE_PAIR / INVALID_PROFILE_EVIDENCE; unit-level malformed
   matrix in `test_profile_ingestion.py`.
6. **Host-independent tests**: environment matrix fully mocked (no/one/two
   T4, wrong CC, CUDA-without-torch, no torch); entry-point tests force the
   probe. Suite passes on CPU-only and (by construction) dual-T4 hosts.
7. **Dry-run-live works**: 10 readiness checks without model execution;
   verified locally (source/markers/modules/schemas pass; GPU/command fail
   honestly with exit 2).
8. **Resume tested**: verified reuse, tamper-forced rerun, missing-artifact
   rerun, partial-never-reused (`test_resume.py`).

## Remaining true GPU-only unknowns

Measured span values (all twelve fields), perturbation magnitude, `hidden_ms`
for case A on real transfers, contention `c` on real GEMMs, event-vs-sync
costs on T4, closure fraction on real data, and any promotion decision.
None of these can be resolved without the live session; none block it —
the dry-run exists precisely to de-risk the packaging side first.
