# PHASE2_T4_BATCH1_PAYLOAD.md — GPU Batch #1 payload manifest (W1-T11)

Status: **payload assembled; not launched.** Implements the mechanical
campaign plan in `PHASE2_T4_CAMPAIGN.md` (research/phase2-t4-campaign @
ab358bf, worktree `.freebuff/wt/p2r`) on top of `integration/phase2-campaign`
@ 942c75a (T1 audit-hardening + T3 pydee-arming merged, verified green).
Nothing here spends budget; the §8 pre-launch checklist still gates the push.

## Payload contents (this branch vs 942c75a)

| File | Change | Purpose |
|---|---|---|
| `dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_phase2_campaign.py` | NEW (~960 lines) | Batch-#1 kernel `code_file`: P0–P6 phase machine (§2.4), 4-arm loop + drift bracket (§2.3), mechanism segment (§3.1), bench segment (§3.3), §4.8/4.9 verdicts + aborts, §4.10 evidence layout + manifest + registry record |
| `dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py` | MODIFIED (additive) | `NATIVE_REUSE_TREE=1` fast-path: skips clone/build/tests/pydee + full 29.4 GiB bank re-hash; per-rep cheap bank re-verify (metadata sha256, data size, record-index contiguity, integrity line count). Default OFF — standalone behavior unchanged |
| `dee.cpp/kaggle/deepseek-v4-flash-0731/run_config.json` | MODIFIED | Campaign defaults: lanes 4 / qd 8 / fp4 / dee4_trace / sealed v60 run_id; informational `campaign` block |
| `dee.cpp/kaggle/deepseek-v4-flash-0731/kernel-metadata-phase2-campaign.json` | NEW | Kernel metadata for the campaign `code_file` (same kernel family, 2xT4 `machine_spec`, shards dataset) |
| `dee.cpp/tools/pinned_direct_bench.cpp` | NEW (~740 lines) | P3 microbench (§3.3): pinned-direct `prefetch_host_lease` vs gather-copy enqueue decomposition at 13,369,344 B; pinned pool registration cost; counts 64/256/512/1024; emits `pinned_direct_bench.json` + predeclared HOLD-gate verdict |
| `dee.cpp/CMakeLists.txt` | MODIFIED (additive) | `pinned_direct_bench` target under `if(DEE_CUDA)` — absent from `DEE_CUDA=OFF` builds |
| `docs/CAMPAIGN_FAULT_INJECTION.md` | NEW | SPEC ONLY for the §3.2(a) fault-injection hook + probes (b)/(c) wiring; post-merge implementation |
| `PHASE2_T4_BATCH1_PAYLOAD.md` | NEW | this manifest |

## Arm matrix executed by the driver (§2.2, §2.3)

| Arm | `phase2.enabled` | `host_enabled` | `vram_priority_fix` | Reps |
|---|---|---|---|---|
| A0 baseline | off | off | off | smoke(2) + r1–r3(16) + r4 bracket |
| A1 vram_only | on | off | on | smoke + r1–r3 |
| A2 host_only | on | on | off | smoke + r1–r3 |
| A3 combined | on | on | on | smoke + r1–r3 |

17 generations total, each a fresh subprocess (`NATIVE_REUSE_TREE=1`):
strongest regime-A cold start + per-rep crash isolation (§4.9). All measured
reps run under the sealed v60 `run_id` — required because `run_id` is hashed
into every route-journal record and the §4.5 gate is the sealed terminal
chain `d8539b6e…`. Per-arm host geometry: `budget_bytes=9,126,805,504`,
`dynamic_slots=682`, `slot_bytes=0` (engine-derived 13,369,344),
`alignment=4096`, `policy_slots=0`, `try_pin=true`, identity
`deepseek-v4-flash-0731@9e165c30…` — §2.2 verbatim.

## Evidence layout emitted to `/kaggle/working` (§4.10-conforming)

```text
progress.log                          # driver + inner runs append; never blocks
environment.json                      # P0 probe incl. hardware_ok_2xt4
run_config.json                       # campaign-level resolved config
run_registry_record.json              # RUN_REGISTRY-shaped job record
segment_mechanism.json                # P2: 8 ctests + 3 audit-probe records
pinned_direct_bench.json              # P3: enqueue split + HOLD-gate verdict
dee4-trace-validation.json            # P4: full bank validation (once)
arms/A0_baseline/run_config.json      # per-arm resolved config record
arms/A0_baseline/generated_checkpoint.A0.<rep>.jsonl
arms/A0_baseline/routed_experts.A0.<rep>.jsonl
arms/A0_baseline/result.A0.<rep>.json          # RESULT + engine_stats +
                                             #   phase2 + host_pack +
                                             #   expert_store + stage_profile
arms/A0_baseline/{run_config,integrity,profile,memory,environment}.A0.<rep>.json
arms/A0_baseline/error.A0.<rep>.txt            # when the inner run raised
result.<arm>.summary.json                      # per-arm rollup + band verdicts
native-generate-result.json                    # top-level verdict
manifest.json + manifest.sha256                # sha256 over every artifact
kernel.log                                     # raw stdout/stderr (Kaggle)
```

## Verdicts implemented (§4.8/§4.9)

- `MECHANISM_FAIL` — any of the 8 CUDA ctests non-PASS in P2 (SKIP counts as
  FAIL on a GPU box); arms unreachable.
- `INVALID_HARDWARE_LOTTERY` — P0 != 2xT4; P2+P3 still banked, arms skipped.
- `INVALID_EXPERIMENT` — A0.r1 counters outside the sealed band (±12 on
  `resident_hits`/`cold_loads`/`h2d_*`/`evictions` per engine).
- `TIMEOUT_PARTIAL` — cumulative wall > 5 h, or 3 consecutive tokens beyond
  3× sealed ITL p95 (6,843.79 ms) in a rep.
- `REJECT_EXACTNESS` (per arm) — any measured rep failing §4.5 gates
  (16/16 tokens, sealed text, terminal chain `d8539b6e…`, all non-hardware
  contract gates); remaining reps of that arm skipped, others continue.
- Per-arm annotations: VRAM band verdict (`CONTRACT_REPRODUCED` /
  `REJECT_MECHANISM` (counters at sealed = switch inert) / `PROVISIONAL`)
  on A1/A3; host-tier sanity (`pin_failures=0`, `failures=0`,
  `device_failures=0`, token denominator) on A2/A3.
- `COMPLETE` — mechanism all-pass + ≥3 exact A0 reps + no abort.

## What remains before `kaggle kernels push` (§8 checklist)

1. Push the campaign commit somewhere the kernel can clone (this work was
   committed locally only, per instructions) and pin it via `NATIVE_COMMIT`.
2. CPU 1/10 remote build gate: `DEE_CUDA=ON` compile of dee_core + all test
   targets + pydee + `pinned_direct_bench` on the pinned commit; host ctest
   suite all-PASS.
3. Ledger stub: mark GPU 1/2 committed; quota headroom ≥ ~8 h
   (est. 2.5–3.2 h wall ≈ 5.0–6.4 GPU-h — subprocess-per-rep adds ~1 min/rep
   over the doc's 2.0–2.8 h model).
4. Cron guard paused-goal + watchdog cadence per the v2a runbook tail.
5. `kaggle kernels push` with `kernel-metadata-phase2-campaign.json`
   (code_file = `deepseek_v4_phase2_campaign.py`), then record
   `kernel_version` into the registry stub.

## Local verification (this machine — no GPU/nvcc)

- `python -m py_compile` on both harness scripts: PASS.
- MinGW `DEE_CUDA=OFF` build + ctest: see the commit message / report.
- `pinned_direct_bench` compiled standalone without `DEE_CUDA` and run on the
  mock path (`--host-only`): emits measured:false timings + gate verdict.

## Known limitations / honesty labels

- Host-arm wall numbers remain **perf-lower-bound** (T9 concurrent fill never
  landed; §6.1 degraded design is in force — A0↔A1 is the decisive pair).
- Audit probe (a) is `NOT_RUN` until the fault-injection hook lands
  (`docs/CAMPAIGN_FAULT_INJECTION.md`); probe (c) is `PARTIAL` via the
  existing `cudaLaunchHostFunc` gate; probe (b) is `COVERED` by
  `test_phase2_cuda`'s `run(false)` half.
- The inner per-rep run re-checks `check_gpu_allocation` + torch repair on
  every rep (≈seconds; also a per-rep admission gate).
