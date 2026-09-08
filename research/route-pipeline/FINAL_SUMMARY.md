# Route-pipeline goal track — final summary (MUSE)

Previous dependency audit accepted and built on (lead-0 spine, R1–R6 in
`sim.py`). This installment turns the three legal overlaps into
implementation-quality isolated prototypes. No production, tolerance,
campaign, sealed-evidence, or `research/t4-kernel-next` change. No merge.

## Model-faithfulness fix (review-driven)

Candidate A initially resolved one six-ID list and reused it across L0–L2.
Official semantics are per-layer tables (`tid2eid` is a per-Gate
parameter), so the prototype now contracts `ids_L0/L1/L2` independently
(`resolve_all_hash_layers`, missing table raises), keys suppression on
`(layer, expert_id)` tuples only, tags telemetry rows with table identity,
and ships `load_tid2eid_rows()` for real per-layer fixture extraction.
Regression tests lock: distinct rows, correct six per layer, cross-layer
numerics never suppressed, same-record suppression, weights gate, no L3+
prefetch. Maximum accounting revalidated unchanged: 18 records,
240,648,192 B. A remains LIVE_MICROBENCH_NEXT.

## Profiler cross-check (read-only, sealed A/B)

Newest valid profiled candidate (`host-reuse` seal): decode 71.179 s /
16 tokens, storage 2.07 GB/token, H2D 3.71 GB/token; summed CUDA-event
device time ≈ 3.5 s of 71 s wall. Critical path is host-side staging +
synchronization — the exact surface A/B/C attack. Shared-expert time is NOT
in the profile (torch path → UNKNOWN); individual sync costs are UNKNOWN
(inside the host-side gap). No speedup is derived from launch counts.

## Ranking

| Candidate | Rank | Why |
|---|---|---|
| A hash L0–L2 early staging | LIVE_MICROBENCH_NEXT | Legality + exactness + implementation all closed host-side (table-gated IDs, evidence-gated weights, immutable records, dup suppression, cancel accounting). Only unknown is measured hidden ms. Max 240.6 MB/token early. |
| B shared overlap | WORTH_PROTOTYPING | Join exactness closed (buffered, order-preserved). Value hinges on unmeasured shared runtime + contention `c` (runner case B). Likely hides under host-bound staging, not under GEMM — must be measured, not assumed. |
| C event handoff (+D2H narrow) | WORTH_PROTOTYPING | Numerically free, lifecycle-expensive: needs the per-consumer reader/eviction/error proof (listed, not done). Prize bounded by unmeasured unrelated-work + event costs (runner case C). Never remove-a-sync without that proof. |

## The 10 required answers

1. **Yes** — hash IDs for L0–L2 at token start (table lookup; fail-closed).
2. **≤18 reads / ≤240.6 MB per token**, minus resident/host-packed/pending
   duplicates (suppression implemented + tested).
3. **Lead per layer**: L0 = embedding+dense_0+router_0 window; L1/L2 = full
   predecessor-layer compute windows for staging submits (consumption still
   chained on weights). Exact ms pending runner case A.
4. **Yes** — shared needs only `h_L`; prototype runs it across routed
   staging/compute with buffered exact join.
5. **Possibly, for the GEMM region — which is why the expected value sits in
   the staging region**: routed staging uses no SMs/DRAM, so shared hiding
   there is near-free; shared-vs-GEMM splits bandwidth (`c` measured live).
6. **Per-layer `cudaStreamSynchronize`** (forced at `engine.cpp:748-754`) is
   wider than necessary in principle; sync-free precedent exists
   (`:1927-1935`). Route-D2H full-stream sync likewise. Neither removed here.
7. **Yes in principle** — pinned copy + copy-event wait; prize = unrelated
   work skipped, copy wait unavoidable. Formal verdict after case C.
8. **Lifecycle risk for events**: premature reuse/eviction, double record,
   swallowed errors — each with a negative test; plus the unsealed
   per-consumer reader proof. Numerical risk: none.
9. **Next microbench: `kaggle_runner_abc.py` full A/B/C run** (one session;
   A gates the staging prize, B yields `c`, C bounds the barrier prize).
10. **Return to C1/C2 when** the campaign profiler shows
    decode/materialization or expert compute on the end-to-end critical path
    AFTER storage work improves (profiler evidence, no fixed X).
    `research/t4-kernel-next @ eddb752` preserved untouched.

## Verification

- `tests/test_route_pipeline_abc.py` — 18 passed (this track's cases).
- `tests/test_route_pipeline.py` — 10 passed (prior sim, untouched).
- Note: `tests/test_legal_overlap.py` (untracked files from a concurrent
  worker in this shared worktree: `legal_overlap.py`, `shared_reorder.py`)
  has 1 failure in THEIR work-in-progress; left completely untouched, not
  committed, unrelated to this track (no shared imports).

---

# Host/sync profiler installment (MUSE)

Profiling-only: C++ HostSpan API + engine markers (RAII, default-off),
`DEE_HOST_PROFILE=1` Python rows, schema/closure/decision engine
(`host_profiler.py`), Kaggle matched-pair pack. No sync removed, no
reorder, no arithmetic/routing change, no staging/overlap/cache/tolerance
change. `research/t4-kernel-next` and Flash's branch untouched.

## The 10 required answers

1. **Directly measurable now**: route-D2H split (copy submit/wait
   start/wait end + bytes/copies), native-call wall, source/fill/enqueue/
   readiness/decode/compute/gather-scatter host spans, per-layer output-sync
   wait, shared host wall (+CUDA-event interval where completable), combine,
   orchestration/handoff (schema-reserved, caller-wired).
2. **Route-D2H separation**: copy-submit timestamp splits submission from
   waiting; the wait span covers sync exit minus entry. Attribution of "how
   much was unrelated work" needs the event timeline alongside (reported,
   not subtracted) — the narrowing decision uses the copy floor + timeline.
3. **Yes**: `NativeOutputSync` wraps the existing final sync per
   layer/token; removal/narrowing is NOT done here.
4. **Yes, without perturbation**: host wall needs no sync; device interval
   uses deferred record-only events with elapsed resolved at dump;
   incomplete events go UNKNOWN (never waited on). Any future sync-added
   measurement must be labeled PROFILE_PERTURBING.
5. **Workers carry no wall timers** — only the calling thread's waits are
   timed (byte counters + CUDA events elsewhere); nested spans are excluded
   from closure sums by rule, enforced in code and tests.
6. **Potentially all of it**: schema covers every known wait class; honest
   closure target is 0.85 or better, unknown carried explicitly, never
   force-filled.
7. **Overhead**: one branch per marker + two RAII objects per native call;
   one flag check per Python forward; zero added syscalls/CUDA/syncs.
   Residual perturbation UNKNOWN until the matched off/on Kaggle pair runs.
8. **Exact run**: `kaggle_profile_pack.py --run` with the canonical decode
   command + prompt hash + tokens (2xT4/source/gates/denylist verified,
   matched off/on, four artifacts out). See KAGGLE_PROFILE_PROTOCOL.md.
9. **Promote B** iff closure is 0.85 or better AND measured shared fraction
   is 0.05 or better with legal idle overlap (runner case-B `c` supporting).
10. **Promote C/D2H** iff closure is 0.85 or better AND output-sync (resp.
    route-D2H excess over copy floor) fraction is 0.05 (resp. 0.02) or
    better. Thresholds are explicit parameters, not verdicts.

## Verification

- `tests/test_host_sync_profiler.py` — 17 passed: default-off (C++ member,
  EngineConfig, runtime Python flag), diff-scan proving instrumentation-only
  engine/candidate changes, nesting rules, UNKNOWN-safe closure, malformed
  fails closed, attribution, hold-by-default + evidence-gated promotion.
- Prior suites untouched and green (`test_route_pipeline*.py`).

---

# Evidence-run installment (experiment/host-sync-profile, MUSE)

One-command entry point `dee.cpp/experiments/route_pipeline/run_evidence.py`
(profile-off/on validation, attribution, closure, ABC mechanics, ranking;
exit 3 with a BLOCKED bundle when no live GPU; never invents timings).
Executed on this host: no nvidia-smi, `torch.cuda.is_available() == False`,
no nvcc, no Kaggle — bundle written to
`research/route-pipeline/evidence/` with zero invented timings.

## Final main classification: BLOCKED_LIVE_GPU

## The 10 required answers (all honest: unmeasured without T4)

1. **Accounted fraction: UNKNOWN** (no execution; closure engine tested on
   fixtures only, target 0.85 or better on real data).
2. **Largest measured host wait: UNKNOWN** (sealed profile shows ~67 s of
   71 s wall outside CUDA-event device time — the gap this profiler fills).
3. **Route-D2H wait: UNKNOWN** (split timers built; copy floor vs unrelated
   work needs the live timeline).
4. **Final-sync wait: UNKNOWN** (per-layer NativeOutputSync span built).
5. **Shared expert: UNKNOWN** (deferred-event capture built; device
   interval resolvable at dump without perturbation).
6. **Profiler perturbation: UNKNOWN** until the matched off/on pair runs;
   by construction one branch per marker, zero added syncs/CUDA calls.
7. **Hidden hash staging: UNKNOWN ms** (mechanics runner ready; max
   240.6 MB/token structural bound stands).
8. **No candidate wins**: all UNMEASURED; rankings stay provisional.
9. **No candidate implemented** (correctly: gates need live data) —
   NO_OVERLAP_OPTIMIZATION_JUSTIFIED is not yet returnable either, for the
   same reason. BLOCKED is the honest terminal state.
10. **ONE future A/B**: after Flash's pack-cap experiment, run
    `run_evidence.py --command <canonical decode> --prompt-hash <sha>
    --tokens 16` on dual-T4; if closure passes and exactly one gate in
    `candidate-ranking.json` promotes, package that single candidate's
    matched A/B next. No gain claimed until then.

## Verification (this branch)

- `tests/test_run_evidence.py` — 5 passed (no-GPU probe, BLOCKED schema,
  zero-timings invariant, live-arg gating, UNMEASURED defaults).
- Full suite: 56 passed warning-free (this clean tree has no foreign files).
- Evidence bundle committed with `timings_invented: 0`.

---

# End-to-end readiness installment (experiment/host-sync-profile, MUSE)

Review finding accepted: the BLOCKED classification was correct but
`compute_closure([])` and unparsed subprocess tails meant LIVE_PIPELINE_READY
was false. This installment closes it: `run_evidence.py` now orchestrates
pack → pair validation → ingestion → closure → ABC → ranking → next-ab
through one code path; `mock_campaign.py` (7 deterministic scenarios)
exercises the identical path with fixture data. No production, tolerance,
campaign, sealed-evidence, or `t4-kernel-next` change. No merge.

## The 10 required answers

1. **Accounted fraction: UNKNOWN live** (no T4 here); the closure engine is
   proven on fixtures (scenario 1: 0.99; scenario 5: 0.33 with
   PROFILE_INCOMPLETE, never force-filled).
2. **Largest measured host wait: UNKNOWN live** (sealed gap stands: ~67 s
   of 71 s outside device events).
3. **Route-D2H wait: UNKNOWN live** (split timers + copy floor ready).
4. **Final-sync wait: UNKNOWN live** (per-layer span ready).
5. **Shared expert: UNKNOWN live** (deferred-event capture ready).
6. **Perturbation: UNKNOWN live** (matched off/on + perturbation fraction
   wired; by construction one branch per marker, zero added syncs).
7. **Hidden hash staging: UNKNOWN ms live** (mechanics runner ready).
8. **No candidate wins**: all UNMEASURED on real data; mock matrix promotes
   exactly the designed winner per scenario (1→EVENT, 2→SHARED, 3→D2H,
   4→none, 5→incomplete-hold).
9. **Nothing implemented** (correctly: gates need live data).
10. **ONE future A/B**: `run_evidence.py --command <canonical decode>
    --prompt-hash <sha> --tokens 16` on dual-T4 after Flash's pack-cap
    experiment (dry-run first); the emitted `next-ab.json` then names the
    single candidate, mechanism, files, telemetry, gates, and replication.

## Final main classification: BLOCKED_LIVE_GPU
## LIVE_PIPELINE_READY = true

The two coexist by definition: the pipeline is proven (mock E2E green,
artifacts populated, ranking derived, malformed fails closed,
host-independent tests, dry-run works, resume tested); the GPU is absent.

## Verification (this branch, mock-proven stage)

- Full suite: 84 passed warning-free (`test_mock_campaign_e2e` 7
  scenarios, `test_profile_ingestion` 12, `test_resume` 4,
  `test_run_evidence` 12 incl. mocked env matrix, prior suites green).
- `compute_closure([])` placeholder is GONE (pack assembles real records;
  grep-verifiable).
- Dry-run verified locally (source/markers/modules/schemas pass; GPU and
  command checks honestly fail with exit 2).

---

# Live T4 profile installment (experiment/host-sync-profile, MUSE)

RUN FOR REAL on dual-T4 SM75 (kernel v8, ABA OFF1/ON/OFF2, 17 GiB cap).
Source identity: 217a333 + profiler patch aae0f41a (6 files, additive
except one const relaxation; per-arm proof recorded). All arms
ACCEPT_CORRECTNESS, identical IDs/text. Full numbers:
LIVE_PROFILE_RESULTS.md; bundle: evidence-live/ (688 merged records).

1. **Source identity**: 217a33359b06 + patch aae0f41a (proof in evidence).
2. **Profile-off wall**: 71.447 s (OFF1; OFF2 68.669 s).
3. **Profile-on wall**: 66.233 s.
4. **Perturbation**: −5.2 s (−7.3%), ORDER-CONFOUNDED (OFF2−OFF1 −2.8 s
   same session; prior session −5.4 s). Unresolvable below ±3–5 s noise;
   PROFILER_PERTURBING: NO by construction.
5. **Closure**: 0.861 decode-only (57.0/66.2 s); prefill ≈0.98 separately.
6. **Top five**: fill 42.0 s (63%), stage-enqueue 9.4 s (14%), sync 4.9 s
   (7%), unknown bucket 9.2 s (dense attn + orchestration + journal),
   combine 0.14 s. Route-D2H 15 ms total.
7. **Storage**: worker 33.2 GB/2481 req/batch-sum 91 s (overlapped, not
   wall) vs critical fill 42.0 s; H2D ≤5.3 s concurrent; effective
   0.74 GB/s. −52 misses ≈ 0.9 s < noise ⇒ pack-cap paradox resolved:
   deterministic per-miss law falsified, blocking-fill mechanism confirmed.
8. **Route-D2H**: 15.4 ms total; p50 0.022 / p95 0.035 / max 0.16 ms.
9. **Final sync**: 4.894 s; p50 7.99 / p95 8.08 / max 8.12 ms; uniform
   required drain; top layers ≈120 ms each.
10. **Shared**: host UNKNOWN (instrumentation gap found live; fix
    committed for future runs); device 308 ms decode (+24.1 s prefill).
11. **ABC**: A hidden −98 ms (batching worse; mechanics only); B c=0.021
    worst-case bound; C 8.5 µs vs 5.4 µs empty.
12. **Winner**: EVENT_HANDOFF_PROMOTED per gate (7.4% ≥ 5%, closure ok).
13. **ONE next experiment**: event-handoff-ONLY matched A/B with abort
    criteria (abort if first-pair delta < resolution); predicted outcome
    ≈ 0 because syncs wait on required completion with no legal host work
    available — the A/B would confirm the model, not beat it.
14. **Final classification**: EVENT_HANDOFF_PROMOTED (mechanical) with
    NO_OVERLAP_OPTIMIZATION_JUSTIFIED as the live alternative the root
    should weigh: the data says the wall is fill-throughput-bound (42 s)
    and no A/B/C candidate reduces fill service.
