# Milestone 2.5 takeover audit — interim state

This note records the verified state of the Milestone 2.5 investigation at the
moment the current agent took it over, after exhausting all no-cost avenues.
It exists so the work is not lost and so the next continuation knows the exact
real situation (which differs materially from the prior agent's reported
state).

## Verified repository state

- Branch `opt/real-model-t1`, local HEAD `e1bbd36118bd98cb775609402a92dca1756d2555`
  exactly equals `origin/opt/real-model-t1` (remote is in sync).
- HEAD is two commits *ahead* of the prompt's stated reference points:
  - `46ba273` — Milestone 2.5 instrumentation revision (7,535 added lines
    across native + Python scripts + tests, including deep per-layer,
    per-expert, per-transfer, host-memory, GPU-memory instrumentation).
  - `e4434ab` — add pinned Milestone 2.5 Kaggle forensic job.
  - `e1bbd36` — target explicit dual-T4 forensic kernel (HEAD).
- Working tree is clean w.r.t. real content. The two `tests/data/*.expected.json`
  "modifications" are whitespace/EOL-only (CRLF on Windows), confirmed via
  `git diff -w` empty. All `benchmark_reports/` and `.venv*` are gitignored by
  design; evidence outputs were deliberately never committed.
- The prior `prior-30-tps-audit.md` (382 lines) is committed and complete; it
  fully answers required deliverable #10 (what the ~30 TPS result measured)
  with commit provenance, exact metric semantics, byte identity, and the
  mandatory correct label.

## Kaggle kernel identity (verified from source)

- Canonical slug: `nivind/dee-cpp-ornith-milestone-2-5-forensics`
  (from `dee.cpp/kaggle/ornith-milestone25/kernel-metadata.json`),
  `accelerator: NvidiaTeslaT4`, `gpu_class: dualT4`, private, internet on,
  dataset `nivind/ornith-1-0-35b`.
- Notebook pins implementation commit `46ba273`, builds CUDA (`sm_75`), runs
  native + Python tests, then `run_milestone25_matrix.py --require-dual-gpu`,
  router parity, layer-0 regression, and `analyze_milestone25_matrix.py` to
  emit all required deliverable files (exact list in notebook cell 9:
  `MILESTONE_2_5_VERIFIED.md`, `milestone-2.5-report.json`,
  `memory-timeline.json`, `host-memory-breakdown.json`,
  `gpu-memory-breakdown.json`, `layer-timing.json`, `expert-trace.jsonl.gz`,
  `expert-cache-analysis.json`, `transfer-analysis.json`,
  `profiler-summary.md`, `prior-30-tps-audit.md`, `bottleneck-ranking.json`).

## Known Kaggle forensic-run status (verified from downloaded logs)

The previously-reported "Version 3, dual-T4 verified running" is **not
accurate**. Every downloaded forensic-kernel log shows the same hard failure
at notebook cell 1:

```
AssertionError: dual-T4 assignment required, got 1 GPUs
```

- `milestone-2.5/kaggle-v1-output/...forensics.log`: 1 x Tesla P100-PCIE-16GB
  (sm_60); PyTorch 2.10+cu128 rejects sm_60 (min supported 7.0).
- `milestone-2.5/kaggle-v2-output/...forensics.log`: same single-P100 failure.
- `milestone-2.5/kaggle-v15-output/...dual-t4-proof.log`: same single-P100
  failure on the proof kernel too (this run).
The explicit `NvidiaTeslaT4` accelerator in the metadata was silently
ignored by the Kaggle scheduler for these submissions. Note that the genuine
Milestone 2 proof kernel `nivind/dee-cpp-ornith-milestone-2-dual-t4-proof`
**version 14** did successfully obtain dual T4, so dual-T4 assignment IS
achievable on this account; the forensic kernel's P100 assignment is
intermittent/retry-able, not a fundamentally broken metadata.

## Genuine dual-T4 Milestone 2 evidence (already collected, verified)

From `benchmark_reports/kaggle-milestone2/v14-output/ornith-milestone2-evidence/`
(kernel `nivind/dee-cpp-ornith-milestone-2-dual-t4-proof` v14, status
`complete`, result `PASS`, dual Tesla T4). This is the source of every
headline number in the prompt:

- Full token parity vs Transformers, max abs error 0 across full trace.
- Permanent layer-0 regression PASS; 40 genuine layers; 31,666 tensors;
  70,214,363,872 checkpoint bytes.
- Warm: TTFT 2.370449 s, prefill 0.421861 tok/s, decode 0.465670 tok/s,
  per-token decode 2.139/2.149/2.154 s; cold per-token 484/344/279 s.
- Whole-process peak RSS 38,818,201,600 B; per-GPU sampled peak VRAM
  5,015,207,936 B; per-phase peak VRAM 4,006,477,824 B per GPU.
- Warm expert cache: 272 hits / 1008 misses / 21.25% hit rate /
  6,341,787,648 H2D bytes = exactly 6,291,456 B per miss.
- Cache split: warm 1008 misses; aggregate (cold+warm) 1185 hits / 5824 loads
  / 5504 evictions / 36,641,439,744 H2D; per-device cuda:0 565 hits and
  cuda:1 620 hits.

This evidence is **aggregate only** (per-phase totals, peak RSS/VRAM, cache
hit/miss counts). It does **not** contain the per-token, per-layer,
per-expert, per-transfer forensic attribution that Milestone 2.5 requires
(`expert-trace.jsonl`, `layer-timing.json`, per-step host-memory-breakdown,
transfer-overlap, GPU concurrency, etc.).

## Verified locally (no GPU required)

- 28/28 forensic + analyzer Python tests pass
  (`tests/test_analyze_milestone25_expert_trace.py`,
  `tests/test_analyze_milestone25_matrix.py`, `tests/test_milestone25_memory.py`,
  `tests/test_run_ornith_forensics.py`) — pytest 9.0.2 / Python 3.13.5.
- 6/6 support tests pass (`tests/test_ornith_support.py`).
- So the *analysis layer* of the instrumentation is verified correct: once a
  dual-T4 run produces instrumented matrix output, the committed analyzers
  will produce the required deliverables. The gap is only the **run**, not the
  analytics.

## Blocker: forensic evidence requires a paid-resource action

- The Lightning devbox (`s_01ky0cg3593rmmtssaaeadzkjx@ssh.lightning.ai`) is
  **offline**. SSH publickey is rejected at the gateway
  (`Permission denied (publickey)`; key
  `SHA256:2eLXYbvhZt4bhXUel19yGRuqcwD9R/cjOw6FX1QEZ/w` offered but refused),
  which is Lightning's behavior for a stopped instance.
- Kaggle credentials (`kaggle.json`) and the configured `kaggle` CLI exist
  **only on the devbox**. Verified absent on this Windows host
  (`~/.kaggle` does not exist; no `KAGGLE_*` env vars). Therefore the
  required first read-only Kaggle action,
  `kaggle kernels status nivind/dee-cpp-ornith-milestone-2-5-forensics`, cannot
  be run from here.
- No required local data exists only on the Windows host that is at risk; the
  remote git already contains the instrumentation and Kaggle metadata.
- The missing deliverables (per-layer/per-expert/per-transfer forensic
  attribution) can be produced only by running the instrumented
  `run_milestone25_matrix.py --require-dual-gpu` on dual Tesla T4, via either
  (a) the devbox's own GPUs (needs the devbox started), or (b) a successfully
  dual-T4-assigned resubmission of the forensic Kaggle kernel (needs the
  devbox's Kaggle credentials).
- Restarting a paid Lightning instance is an explicit ask-before action.
  No autonomous path to the remaining measured evidence exists without it.

## Next action required from the user

Pick one:
A. Start (restart) the Lightning devbox so it can be SSH'd into; then resume
   polling the Kaggle forensic kernel and/or drive the instrumented matrix
   locally on the devbox GPUs, then publish and shut down.  **OR**
B. Provide the Kaggle credentials (`kaggle.json`) on this Windows host so the
   read-only status poll and an in-place dual-T4 kernel resubmission can be
   driven from here without touching the devbox.

Either unblocks the full evidence collection; everything downstream
(analysis, bottleneck ranking, fail-closed audit, commit/push, devbox
shutdown) is then autonomous.
