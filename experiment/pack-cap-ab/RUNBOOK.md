# RUNBOOK — pack-cap A/B execution and analysis

Branch: `experiment/pack-cap-ab` (base `ce6cb81`) · Worktree `../dee-pack-ab`
All numbers except measurements are SIMULATED/PROJECTED per the contract.

## 1. What runs

Two Kaggle sessions, each running BOTH arms as fresh OS processes with a
shared, deterministic DEE4 trace bank:

| session | kernel slug (20260905) | order | status |
|---|---|---|---|
| 1 | `nivind/dee-cpp-dsv4-pack-cap-s1r-20260905` | A → B | dispatched v1 (repaired) |
| 2 | `nivind/dee-cpp-dsv4-pack-cap-s2r-20260905` | B → A | dispatched v1 (repaired) |

Failed first dispatches (unrepaired slugs `...-s1-20260905` v1/v2,
`...-s2-20260905` v1) are preserved as INVALID_EXPERIMENT evidence: both arms
died identically at `git checkout` before any GPU work — no results observed.

## 2. GitHub force-push repair (2026-09-05)

The remote branch `freebuff/deepseek-v4-flash-0731-t4` was force-pushed to
`7b137846` after the host-reuse seal. Commit `217a3335` (and parent
`14864034`) no longer exist on GitHub, so the seal-era incremental bundle
(prerequisite = then-tip) could not apply on a fresh clone: first dispatch
failed `reference is not a tree: 217a3335...` on BOTH arms identically.

Repair (`bundle/repair-217a3335-prereq-7b137846.bundle`,
sha256 `76e1b437...`): new incremental bundle carrying the **bit-identical**
commit `217a3335` with prerequisite = **current** remote tip `7b137846`
(every fresh clone provides it). Validated locally end-to-end: fresh clone →
fetch → `checkout 217a3335` OK. The harness changes are exactly the two
embedded source-integrity constants (`SOURCE_BUNDLE_SHA256`, `SOURCE_BUNDLE_B64`);
the fetch refspec and checkout pin are untouched. Arm A vs seal-era harness:
only the repair; arm B vs arm A: only `LRU_TOTAL_CAP_GIB 17.0 → 20.0`
(18 tests enforce all of this, `tests/test_pack_cap_tooling.py`).

## 3. Commands (all from the worktree root)

```bash
# build packages (verifies pinned base harness sha; refuses on drift)
python experiment/pack-cap-ab/tools/make_harness.py

# tests (contract lint, harness provenance, driver behavior, decision rule)
python -m pytest experiment/pack-cap-ab/tests/ -q

# dispatch / poll / fetch (recipe matches the seal-era run_control.py)
python experiment/pack-cap-ab/tools/dispatch.py dispatch \
    --dir experiment/pack-cap-ab/kernels/session1
python experiment/pack-cap-ab/tools/dispatch.py status \
    --kernel nivind/dee-cpp-dsv4-pack-cap-s1r-20260905
python experiment/pack-cap-ab/tools/dispatch.py fetch \
    --kernel nivind/dee-cpp-dsv4-pack-cap-s1r-20260905 \
    --out experiment/pack-cap-ab/results/live/s1
```

A session takes ~25–45 min (trace bank ~15 min + 2 arms × ~7–10 min).
Poll `status`; when `KernelWorkerStatus.COMPLETE`, `fetch` the output.

## 4. After both sessions complete

```bash
python experiment/pack-cap-ab/tools/classify_experiment.py \
    --live-dir experiment/pack-cap-ab/results/live \
    --out-dir experiment/pack-cap-ab/results
```

This mechanically applies the pre-registered rule
(`experiment-contract.json`) and writes `combined-comparison.json`,
`comparison-sessionN.json`, `memory-validation.json`, `miss-validation.json`,
including the final classification from:

```
PACK_CAP_SUPPORTED | PACK_CAP_NULL | PACK_CAP_ORDER_CONFOUNDED
PACK_CAP_MEMORY_REJECTED | INVALID_EXPERIMENT | BLOCKED_LIVE_GPU
```

Decision rule (frozen in the contract, never rewritten post hoc):
- informative positive = both pair deltas (B−A decode wall) < 0 AND
  mean ≤ −0.905 s (baseline MAD)
- order confounded if exactly one pair improves beyond noise AND the
  improving arm ran second in its session
- CASE 1 (misses don't fall ≥10) → replay/model failed;
  CASE 2 (misses fall ≥30 but wall null) → service model failed;
  CASE 3 (both) → supported

## 5. Expected (SIMULATED) vs measured

| quantity | SIMULATED prediction | measured |
|---|---|---|
| host-pack budget | 8.5 → 10.0 GiB/GPU (cap 17 → 20 GiB) | `memory.json.host_pack_budget_bytes` |
| storage misses (per GPU) | 1390→1338, 1091→1046 (replay −51 total) | `result.json.host_pack.cuda{0,1}.misses` |
| decode wall | −2.48 s (48.679 ms/miss observational model) | `result.json.decode_wall_s` |
| total process GiB at 20 cap | 27.227 GiB (29.23 decimal GB) | `memory.json` peaks |
| min MemAvailable | ≥ 4.12 GiB projected | `memory.json.minimum_checkpoint_host_mem_available_gib` |

Abort thresholds (pre-registered): min checkpoint MemAvailable < 1.5 GiB,
VmHWM > 30.0 GiB, or host OOM → stop, classify per contract.

## 6. Pread rider status — RESOLVED (2026-09-07)

The session-2 rider failed harmlessly (`store too small for one record`):
the driver had passed `--store <bank>/metadata.json` where the bench
requires the multi-record file. FIXED (one line in `tools/session_driver.py`:
`--store experts.dee4 --journal-meta metadata.json`; regression test
`test_session_driver_rider_targets_experts_dee4`).

A standalone rider kernel (`dee-cpp-dsv4-pread-rider-storage-only-post-a-b`,
built by `tools/make_rider.py`, dispatched AFTER all four arms — contract-
true) then ran the corrected measurement on the same T4 storage class:
repair-bundle clone → checkout `217a3335` verified → v50 bank rebuilt
(492.6 s) → bank identity validated (`data_sha256 c83462ba…` PASS) → bench
at depths 1–16, seq+dispersed, coldish+warm passes. Results and
interpretation: `forensics/PREAD_RIDER_RESULT.md`; raw JSON:
`forensics/results/pread-rider.json`. Headline: cold aggregate saturates
at ~2.9 GB/s by depth 2–4 (flat to depth 16), warm page-cache reads run
~3.2–5.7× faster; decode only ever demanded ~470 MB/s aggregate.

## 7. Evidence retention

Fetched kernel outputs live in `results/live/<s1,s2>/`; the dispatch log is
`results/live/dispatch-log.jsonl` (append-only). Failed first-dispatch
artifacts are retained under the same tree. Nothing here is sealed campaign
evidence; sealed evidence was never touched.
