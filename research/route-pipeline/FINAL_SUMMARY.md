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
