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
