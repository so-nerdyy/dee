# archive

Code that is no longer part of the supported engine, kept so that older
evidence and reports still point at something real. Nothing here is needed to
build or run DEE.

## `ornith-prototype/`

The first, July 2026 prototype: Modal scripts that collected router data from
Ornith-1.0-35B and Qwen3-30B-A3B, trained the `oracle.pt` expert predictor, and
tested a Python expert cache and a Triton kernel. These files used to live at
the repository root (`modal_step1_datacollection.py` ... `modal_step4_triton_kernel.py`,
`modal_verify_*.py`, `modal_minimax_*.py`, `inspect_header.py`,
`test_step3_cache.py`, `ornith_idx.json`). The native engine in `dee.cpp/`
replaced them.

## `kaggle-supervisors/`

Local supervisors that launched and monitored the Milestone 3/4 and Phase 1
Kaggle T4 runs. These used to live in `tmp/` at the repository root, and
evidence files and reports from those runs refer to them as `tmp/<name>.py`.
Their tests are in `dee.cpp/tests/test_m3_supervisor_v6.py`,
`dee.cpp/tests/test_m4_supervisor_phase2.py`, and next to the scripts.

One-off patch scripts and `.bak` copies that were in `tmp/` were removed; they
remain in git history.

## Moved documents

| old path | new path |
|---|---|
| `RESULTS.md` | [`docs/phase3/RESULTS_PHASE3.md`](../docs/phase3/RESULTS_PHASE3.md) |
| `PHASE3_BUILD_PLAN.md` | [`docs/phase3/PHASE3_BUILD_PLAN.md`](../docs/phase3/PHASE3_BUILD_PLAN.md) |
| `PHASE3_FULL_EXPERT_STORE.md` | [`docs/phase3/PHASE3_FULL_EXPERT_STORE.md`](../docs/phase3/PHASE3_FULL_EXPERT_STORE.md) |
| `PHASE3_SEGMENTED_READER.md` | [`docs/phase3/PHASE3_SEGMENTED_READER.md`](../docs/phase3/PHASE3_SEGMENTED_READER.md) |
| `JSON_MIN_NULL.md` | [`docs/phase3/JSON_MIN_NULL.md`](../docs/phase3/JSON_MIN_NULL.md) |
