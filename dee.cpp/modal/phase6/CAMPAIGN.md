# Phase-6 campaign runbook (AWS EC2, us-east-2)

Phase 6 runs DSv4-Flash on rented GPUs and grades every run against the predictions
registered before any run existed: `theory/PREDICTIONS_PHASE6.md` (frozen 2026-09-30) and its
Addendum 2 (AWS cells, registered 2026-10-08, function `theory/predecl_aws.py`). This file is
the order of work, the exact commands, what each run costs, and how a run becomes a graded row.

Every command that bills is marked **BILLS**. Everything else is free.

## 0. What "Phase 6 done" means

1. Every provisionable registered AWS cell has a scored b=1 run, and a b=8 run where budget
   allows. Each run has `scorecard.md` next to its evidence and one line in `SCOREBOARD.jsonl`.
2. `PREDICTIONS_PHASE6.md` gets an append-only **Addendum 3** with each row's grade. Killed rows
   are marked `KILLED (date, evidence path)` and the theory is corrected, not defended (section 0).
3. Cells that can't be provisioned (quota, budget) are listed as descoped with the reason.

REPORT_PHASE7 section 7 names the minimum useful deliverable: one L4-class run at b=1 (and b=8 if
budget allows), with at least 16 tokens, the stage profile and measured B_SSD. That run alone
scores P2-P5 on new hardware.

## 1. Preflight (free)

```
cd dee.cpp/aws/phase6
python -m pip install boto3 "botocore[crt]" pytest numpy scipy
python -m pytest -q test_phase6_aws.py test_score.py      # harness + scorer
python infra.py status                                     # bucket, role, SG, GPU quota
aws s3 ls s3://dee-p6-147224001180-use2/stores/dsv4-flash/metadata.json --profile dee
aws s3 ls s3://dee-p6-147224001180-use2/dense/dsv4-flash/manifest.json  --profile dee
```

Blockers to clear before the first GPU run:

| check | needed | how |
|---|---|---|
| G/VT on-demand vCPU quota (`L-DB2E81BA`) | **8** for the 2xlarge cells, **32** for g6.8xlarge / g5.8xlarge, **64** for g6e.16xlarge | Service Quotas console, us-east-2. At 2026-10-08: 0 approved, 8 requested |
| segmented store in S3 | `stores/dsv4-flash/metadata.json` present and complete | `python launch_cpu.py --job store --model dsv4-flash --launch` (**BILLS**, CPU) |
| dense-only checkpoint in S3 | `dense/dsv4-flash/manifest.json` (8.238 GiB, 1,564 tensors) | `python launch_cpu.py --job dense --model dsv4-flash --launch` (**BILLS**, CPU) |
| HF token | in your shell as `HF_TOKEN` for CPU jobs | written to SSM by `launch_cpu.py --launch` |

## 2. Run order

Cheapest first, so a harness bug costs a dollar, not ten. Every launch is a dry run first.

| # | cell (Addendum 2) | b | host | quota | command (add `--launch` to bill) |
|---|---|---|---|---|---|
| 1 | aws-L4-g6.2xlarge | 1 | 16 | 8 | `python launch.py --gpu L4 --host-gib 16 --n-tokens 16 --run-id p6-l4-h16-b1` |
| 2 | aws-L4-g6.2xlarge | 8 | 16 | 8 | `python launch.py --gpu L4 --host-gib 16 --n-tokens 65 --ignore-eos --cohort 0-7 --prompts-file prompts_phase4_8.json --run-id p6-l4-h16-b8` |
| 3 | aws-A10-g5.2xlarge | 1 | 16 | 8 | `python launch.py --gpu A10 --host-gib 16 --n-tokens 16 --run-id p6-a10-h16-b1` |
| 4 | aws-A10-g5.2xlarge | 8 | 16 | 8 | as #2 with `--gpu A10 --run-id p6-a10-h16-b8` |
| 5 | aws-L40S-g6e.2xlarge | 1 | 16 | 8 | `python launch.py --gpu L40S --host-gib 16 --n-tokens 16 --run-id p6-l40s-h16-b1` |
| 6 | aws-L40S-g6e.2xlarge | 8 | 16 | 8 | as #2 with `--gpu L40S --run-id p6-l40s-h16-b8` |
| 7-10 | aws-L4-g6.8xlarge, aws-A10-g5.8xlarge | 1, 8 | 64 | 32 | as above with `--host-gib 64` |
| 11-12 | aws-L40S-g6e.4xlarge | 1, 8 | 64 | 16 | as above with `--gpu L40S --host-gib 64` |
| 13-14 | aws-L40S-g6e.16xlarge | 1, 8 | 256 | 64 | as above with `--gpu L40S --host-gib 256` |

Notes on the flags:

- **b=1** uses the runner's default anchor prompt (no `--prompt`), 16 generated tokens = the
  `finite_16tok` horizon (one prefill + 15 decode steps).
- **b=8** needs at least 64 decode steps, so `--n-tokens 65`, and `--ignore-eos` so no row stops
  early. The scorer grades decode tok/s and $/1k on the last 32 steps only.
- `--host-gib` is always explicit. The launcher refuses rather than clamps. 256 GiB is not
  provisionable on L4/A10 (Addendum 2).
- `--spot` is fine for cost. Spot runs are still scored at the on-demand rate (Addendum 2 (iv)).
- Leave `--budget-mib` at the GPU default (3,584 MiB L4/A10, 28,672 MiB L40S). Any other value
  is not a registered cell and the scorer marks the run UNTESTABLE.

## 3. Cost (ASSUMPTION: on-demand prices read 2026-10-08)

`launch.py` prints the hard cap `max_cost_usd = USD/h x --max-hours` (default 3 h; user-data arms
`shutdown -h` at that time). The estimates below assume about 45 minutes of wall time per run:
clone and build, a ~147 GiB S3 sync to instance NVMe, fio probe, then generation. This is a guess
until run #1 reports `wall_s`. Use run #1's `summary.json` to replace it.

| cell | USD/h | est. per run (45 min) | cap per run (3 h) |
|---|---|---|---|
| g6.2xlarge (L4) | 0.9776 | ~0.73 | 2.93 |
| g5.2xlarge (A10) | 1.212 | ~0.91 | 3.64 |
| g6e.2xlarge (L40S) | 2.242 | ~1.68 | 6.73 |
| g6.8xlarge / g5.8xlarge | 2.014 / 2.448 | ~1.51 / ~1.84 | 6.04 / 7.34 |
| g6e.4xlarge | 3.004 | ~2.25 | 9.01 |
| g6e.16xlarge | 7.577 | ~5.68 | 22.73 |

Runs 1-6 (8-vCPU quota only): about **$6.6 estimated, $26.6 worst case**. All 14: about
**$30 estimated, $105 worst case**. S3 storage for the store (~147 GiB) is extra and not priced here.

## 4. Score a run (free)

```
RUN=p6-l4-h16-b1
aws s3 sync s3://dee-p6-147224001180-use2/evidence/$RUN/ runs/$RUN/ --profile dee
python score.py runs/$RUN --write --ledger SCOREBOARD.jsonl
```

`score.py` checks the prerequisites first (registered cell and host budget, VRAM budget, host
budget honored, measured B_SSD, run complete, stage profile present, enough tokens/steps, no early
EOS at b=8). Any failure makes the run **UNTESTABLE**, and the scorecard lists why. It then evaluates
`predecl_aws.predict(cell, b, host, measured_B_ssd)` and grades the four metrics:

| grade | meaning (frozen section 0) |
|---|---|
| PASS | inside the registered `[lo, hi]` |
| WEAKENED | between a range edge and its kill edge |
| KILLED | outside the kill band (P2's 0.45-0.57 intersected in at host 16 GiB) |
| UNTESTABLE | missing prerequisite, or (b=8) the window never reached the modeled regime |

The cell verdict is the worst grade: KILLED, then WEAKENED, then PARTIAL (some metrics
untestable), then PASS.

Read by hand from the stage profile in `result*.json` (not automated): P5 (eta_s 0.15-0.45),
P8b/P8c (per-touch dispatch), P16/P17 (limiter identity vs the predicted limiter in the scorecard).

## 5. Record results

After each scored run:

1. Commit `runs/<run_id>/scorecard.{md,json}` and the `SCOREBOARD.jsonl` line on `research/phase6-aws`.
   Keep raw evidence in S3, and commit only what the scorecard cites, as earlier phases did under
   `benchmark_reports/`.
2. Append a dated row to Addendum 3 in `theory/PREDICTIONS_PHASE6.md`. Never edit sections 0-5 or
   the frozen tables.
3. On a KILLED row: write what was killed, the measured value, and the evidence path. Then fix the
   theory in a new commit on `research/phase7-theory`.

## 6. Known weaknesses, stated before the runs

- **b=1 $/1k reads high by construction.** The prediction uses the steady-horizon TPS (frozen
  economics convention, section 3b item 9). The measurement uses the run's finite 16-token decode
  TPS, so expect WEAKENED or KILLED on cost where the finite and steady TPS differ. This is
  registered. Report it, don't patch it.
- **b=8 steady cold rate is ~0 at host >= 64 GiB** (Addendum 2 (i)). Arbitrary prompts touch
  experts the sealed trace never saw, so the b=8 gate may close (TPS/$ UNTESTABLE). That gate is
  the registered rule, not a scorer bug.
- **The 8-prompt workload repeats prompt 0 as prompt 7** (the Phase-4/5 determinism probe). This
  raises cross-row sharing slightly at b=8. It is the established workload. Note it in Addendum 3.
- **Untested against a real run.** The scorer's field names come from
  `kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py` (`result*.json`,
  `generated_checkpoint-c*.jsonl`), and it is tested on synthetic fixtures only. Check the first
  real scorecard against the raw `result.json` by hand.
- **MiMo-Flash is not on AWS** (no dense-only extraction yet), so its frozen rows stay UNTESTABLE.
