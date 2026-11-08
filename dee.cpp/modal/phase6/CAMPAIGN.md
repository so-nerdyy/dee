# Phase-6 campaign runbook (Modal)

Phase 6 runs the exact-MoE engine on rented GPUs and grades every run against the predictions
registered before any run existed: `theory/PREDICTIONS_PHASE6.md`, frozen 2026-09-30, plus
**Addendum 3** (2026-11-04), which registers Modal rows at the run's measured B_SSD and the VRAM
and container sizes `modal_p6.py` actually provisions (`theory/predecl_modal.py`). Every Modal
run is scored twice: against the frozen row as written, and against its Addendum-3 row. The frozen
matrix was written for **Modal**: 5 cells (1xL4, 2xL4, 1xA10, 1xL40S, RTX PRO 6000) × 2 models ×
b ∈ {1, 8} × host {16, 64, 256} GiB = 60 scored cells. Modal is the active platform again
(2026-11-04). The AWS port (`aws/phase6/`, Addendum 2) is paused. Its scorer and helpers are
shared.

This file gives the order of work, the exact commands, what each run costs, and how a run becomes a
graded row. Every command that bills is marked **BILLS**. Commands run from `dee.cpp/`.

> The GPU containers clone `research/phase6-modal` at its head (`modal_p6.py` `BRANCH`). Merge
> this work into that branch before the first run, or pass `--pinned <commit>`.

## 0. What "Phase 6 done" means

1. Each provisionable frozen cell has a scored run. Each run has `scorecard.md` next to its
   evidence and one line in `SCOREBOARD.jsonl`.
2. `PREDICTIONS_PHASE6.md` gets an append-only **Addendum 4** (results) with each row's frozen
   and Addendum-3 grade. Killed rows are
   marked `KILLED (date, evidence path)` and the theory is corrected, not defended (section 0).
3. Cells that can't be provisioned (container RAM, budget) are listed as descoped with the reason.

REPORT_PHASE7 section 7 names the minimum useful deliverable: one L4 run at b=1 (and b=8 if budget
allows) with at least 16 tokens, the stage profile and measured B_SSD. That run alone scores P1-P5.

## 1. Preflight (free unless marked)

```
pip install modal pytest
python -m pytest -q aws/phase6/test_score.py aws/phase6/test_phase6_aws.py
modal run modal/phase6/modal_p6_ops.py::store_status --model dsv4-flash   # store complete?
```

| check | needed | how |
|---|---|---|
| payment method on the Modal workspace | GPU functions refuse to start without one | Modal dashboard → Settings → Billing |
| checkpoint on `dee-models` | `DeepSeek-V4-Flash-0731/` | `modal run modal/phase6/modal_p6_ops.py::stage_ckpt --model dsv4-flash` (**BILLS**, CPU) |
| segmented store on `dee-stores` | `dsv4-flash/metadata.json`, journal complete | `modal run --detach modal/phase6/modal_p6_ops.py::build_store --model dsv4-flash` (**BILLS**, CPU, resume-safe) |
| MiMo-Flash (second model) | same two steps with `--model mimo-flash` | only after the DSv4 slice is scored |

## 1b. Budget rules (2026-11-04: $30 total, shared by every model)

- **Preflight before every GPU launch** (~$0.10 CPU, same image and commit as the run):
  `modal run modal/phase6/modal_p6.py::preflight --gpu L4 --configs "$CFGS"`.
  It builds the image, runs the harness build and the runner's own build, imports torch
  (must be 2.10.0+cu128 on Python 3.12, the sealed build) and pydee, and checks the store
  table, checkpoint, prompts and configs. Don't launch a GPU unless it prints `preflight_ok: true`.
- **One container per GPU type.** `--configs` runs several rows in one container, so build,
  store staging and the B_SSD probe are paid once. Each config is a fresh runner process,
  with page cache dropped (best effort, and recorded), and is written to
  `runs/<run_id>-<tag>`. A failing config doesn't stop the rest.
- **Store seal during the copy.** Staging sha256-checks every segment against the metadata
  table and sets `DEE4_STORE_SKIP_SEAL=1` for the runner. That's the same check, and it
  saves ~17 min per run.
- **Cost guards:** 20 min per b=1 config, 45 min per b=8 config, 150 min per container.
- **b=1 diagnostics:** capture journal + router top-8 values per (step, layer), so a token
  mismatch names its layer and how close the tie was.

The L4 slice in one container:

```
CFGS='[{"tag":"b1-h64","host_gib":64,"n_tokens":16},
       {"tag":"b1-h16","host_gib":16,"n_tokens":16},
       {"tag":"b8-h64","host_gib":64,"n_tokens":65,"ignore_eos":true,"cohort":"0-7","prompts_file":"dee.cpp/aws/phase6/prompts_phase4_8.json"},
       {"tag":"b8-h16","host_gib":16,"n_tokens":65,"ignore_eos":true,"cohort":"0-7","prompts_file":"dee.cpp/aws/phase6/prompts_phase4_8.json"}]'
modal run --detach modal/phase6/modal_p6.py::bench_L4 --run-id p6m-l4-r5 --configs "$CFGS"
```

## 2. Run order (original plan, superseded by 1b for cost)

Cheapest first, so a harness bug costs a dollar, not ten. The first run is the frozen headline cell.

| # | frozen cell | b | host | ids it scores | command (**BILLS**) |
|---|---|---|---|---|---|
| 1 | 1xL4 | 1 | 64 | **P1** (headline), P2-P4 | `modal run modal/phase6/modal_p6.py::bench_L4 --n-tokens 16 --host-gib 64 --run-id p6m-l4-h64-b1` |
| 2 | 1xL4 | 1 | 16 | **P2** short window | as #1 with `--host-gib 16 --run-id p6m-l4-h16-b1` |
| 3 | 1xL4 | 8 | 16, 64 | P2-P4 steady | `modal run modal/phase6/modal_p6.py::bench_L4 --n-tokens 65 --ignore-eos --cohort 0-7 --prompts-file dee.cpp/aws/phase6/prompts_phase4_8.json --host-gib 64 --run-id p6m-l4-h64-b8` |
| 4-7 | 1xA10, 1xL40S | 1, 8 | 16, 64 | P2-P4 | `bench_A10` / `bench_L40S`, same flags |
| 8-9 | 2xL4, RTX PRO 6000 | 1, 8 | 16, 64 | P2-P4 | `bench_2xL4` / `bench_RTXPRO6000`, same flags |
| 10+ | all cells | 1, 8 | **256** | **P1b, P14, P15b, P15c** | `bench_<GPU>_h256` (288 GiB container, registered in Addendum 3), e.g. `modal run modal/phase6/modal_p6.py::bench_L40S_h256 --n-tokens 65 --ignore-eos --cohort 0-7 --prompts-file dee.cpp/aws/phase6/prompts_phase4_8.json --run-id p6m-l40s-h256-b8` |

Notes on the flags:

- **b=1** uses the runner's default anchor prompt (no `--prompt`), 16 generated tokens = the
  `finite_16tok` horizon (one prefill + 15 decode steps).
- **b=8** needs at least 64 decode steps (`--n-tokens 65`) and `--ignore-eos`, so no row stops
  early. Decode tok/s is aggregate over the last 32 steps.
- **`--host-gib` is always explicit.** The default (17 GiB, the sealed T4 split) is not a frozen
  budget, and the scorer marks it UNTESTABLE. The function refuses when host + 12 GiB reserve
  exceeds the container's RAM, instead of clamping.
- Leave `--budget-mib` at the GPU default (3,584 MiB on L4/A10, 26,624 on L40S, 61,440 on RTX; the L40S default was 28,672 until the VRAM-fit check showed 16.2 GiB torch peak + 28 GiB cache overflows 44.4 GiB). The
  frozen rows assumed 22 / 48 / 96 GiB, which can't fit next to the 8.238 GiB dense backbone on
  L4/A10 (Addendum 2 item 2). The scorecard reports this as a warning on every run.
- The function now runs a **fio B_SSD probe** after staging the store (13,369,344 B blocks,
  O_DIRECT, libaio, QD6 × 3, the anchor's definition) and records `measured_b_ssd_gib_s`. If the
  container refuses O_DIRECT, it falls back to buffered I/O and flags it. If `/tmp` can't hold the
  store, it is served from the volume, which is also flagged.

## 3. Cost (frozen price schedule A7, an ASSUMPTION)

Modal bills GPU + CPU (8 cores × $0.0000131/s) + RAM ($0.00000222/GiB/s on the container size in
`GPU_SPECS`). Estimates assume ~45 minutes per run (build, ~147 GiB store copy from the volume,
probe, generation). That is a guess until run #1 reports `wall_s`.

| function | container RAM | ≈ USD/h all-in | est. per run (45 min) |
|---|---|---|---|
| bench_L4 | 96 GiB | 1.94 | ~1.46 |
| bench_A10 | 96 GiB | 2.25 | ~1.68 |
| bench_2xL4 | 96 GiB | 2.74 | ~2.06 |
| bench_L40S | 128 GiB | 3.35 | ~2.51 |
| bench_RTXPRO6000 | 192 GiB | 4.94 | ~3.71 |
| bench_<GPU>_h256 | 288 GiB | L4 3.48 · A10 3.78 · 2xL4 4.28 · L40S 4.63 · RTX 5.71 | ~2.6-4.3 |

Runs 1-3 (the L4 slice, 4 runs): **~$6**. All 16 GiB / 64 GiB DSv4 cells (5 GPUs × 2 b × 2 hosts =
20 runs): **~$46**. The 10 host-256 runs: **~$33** more. That is ~$79 for the whole DSv4 matrix,
before MiMo, and assumes Modal grants a 288 GiB container on each GPU type (unverified). The function timeout is 6 h, so the worst case
for one L4 run is ~$11.6.

## 4. Score a run (free)

```
RUN=p6m-l4-h64-b1
modal volume get dee-p6-evidence runs/$RUN runs/
# optional but registered (section 0 item 1): add the billed cost from the Modal usage page
python - <<EOF
import json, pathlib; p = pathlib.Path("runs/$RUN/summary.json")
s = json.loads(p.read_text()); s["billed_usd"] = 0.00  # <- from the dashboard
p.write_text(json.dumps(s, indent=2))
EOF
python aws/phase6/score.py runs/$RUN --write --ledger SCOREBOARD.jsonl
```

For Modal runs `score.py` grades the frozen rows straight from `theory/data/predecl_matrix.csv`
(stdlib only), then the Addendum-3 row from `theory.predecl_modal` (needs numpy + scipy; without
them that section says NOT SCORED). For the frozen rows:

- **Prerequisites** (any failure → UNTESTABLE): frozen cell and host budget, host budget honored
  by the runner, measured B_SSD present, run complete, stage profile present, ≥16 tokens (b=1),
  ≥64 decode steps and no early EOS (b=8).
- **B_SSD gate:** decode tok/s and $/1k are scored only when measured B_SSD is 0.9-1.1× the cell
  spec (L4 2.5, 2xL4 5.0, A10 3.0, L40S 6.0, RTX 8.0 GB/s). Otherwise those two metrics are
  UNTESTABLE, while host hit and cold rec/tok are still scored.
- **ID tightening** (section 4): P1 0.9-2.1 and P1b 8-28 on decode tok/s, P2 0.45-0.57 on host hit
  at 16 GiB, and P14 $0.05-0.20 (plus TPS 8-28). P15b (ratio vs dense ≥ 0.95 kills) and P15c (< 0.85
  kills) are graded against the expensive dense reading. The cheap-reading ratio is reported
  alongside.
- **$/1k** uses `billed_usd / emitted tokens` when present, otherwise `1000 × rate_s / TPS` with the
  frozen `serve_cost_frontier.csv` rate, plus a warning.

The Addendum-3 section follows Addendum 2's rules: the prediction is evaluated at the measured
B_SSD (so no B_SSD gate), P2 applies at 16 GiB, the b=8 steady gate applies, and the run must use
the registered VRAM budget and container (`bench_<GPU>_h256` at 256 GiB).

Read by hand from the stage profile (not automated): P5 (eta_s 0.15-0.45), P8b/P8c, P16/P17
(limiter identity).

## 5. Record results

1. Commit `runs/<run_id>/scorecard.{md,json}` and the `SCOREBOARD.jsonl` line on
   `research/phase6-modal`. Keep raw evidence on the volume, and commit only what the scorecard
   cites, as earlier phases did under `benchmark_reports/`.
2. Append a dated row to Addendum 4 in `theory/PREDICTIONS_PHASE6.md`. Never edit sections 0-5 or
   the frozen tables.
3. On a KILLED row: write the measured value and the evidence path, then fix the theory in a new
   commit on `research/phase7-theory`.

## 6. Known weaknesses, stated before the runs

- **Frozen decode tok/s and $/1k may be UNTESTABLE everywhere** if Modal's real disk speed isn't
  within ±10% of the assumed spec. Addendum 3 is what keeps those two metrics scorable. It only
  counts as a pre-registration because it is committed before the first run, so keep it that way.
- **The frozen VRAM provisioning is unrealizable on L4/A10** (22 GiB assumed vs 3.5 GiB usable next
  to the backbone). Cold rec/tok and host hit are scored as registered, with the deviation shown.
- **The frozen b=8 rows carry the units defect** (Addendum 2 item 3): scored as registered, and
  expected to read low against measured aggregate throughput.
- **b=1 $/1k reads high by construction:** the prediction uses steady-horizon TPS (section 3b item
  9), while the run measures a 16-token window.
- **The 8-prompt workload repeats prompt 0 as prompt 7** (the Phase-4/5 determinism probe), which
  adds a little cross-row sharing at b=8.
- **Not yet run on Modal:** the probe, cohort and host-budget code in `modal_p6.py`, and the
  scorer, are tested only on synthetic fixtures. Check the first scorecard against the raw
  `result.json` by hand.
