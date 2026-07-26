# Milestone 3 evidence reconstruction

Reconstructed 2026-07-26 before launching another Kaggle experiment.

## Repository and execution state

- Branch/HEAD: `opt/real-model-t1` at `a2c0870`, matching
  `origin/opt/real-model-t1`.
- The historical M3 commits exist. The handoff's `bcd194c` reference is
  `dcd194c` in this repository.
- No M3 supervisor is alive. Every `tmp/m3_v62_supervisor*.pid` is stale.
- The pre-existing worktree modification in `tmp/m3_supervisor_v6.py` is a
  54-line Kaggle credential propagation repair. It has been preserved.
- Latest authoritative Kaggle state is `ERROR`, not `FINISHED`
  (`tmp/m3_supervisor_v6_state.json`).

## Artifact map

- M3 v4: `tmp/m3_v4_output/ornith-milestone3-evidence`
  - seven run directories exist;
  - six experiments are committed as PASS in `matrix-summary.json`;
  - `dual-long-prompt/run-report.json` says PASS, but its child process aborted
    during cleanup before the matrix driver could append the seventh result;
  - the raw failure is in `logs/matrix-driver.log` immediately after late-layer
    eviction output.
- M3 v5: instrumentation commit `4d8ccf2`; no separate experiment bundle.
- M3 v6/v6.1/v6.2: `tmp/m3_v6_output*`
  - only preflight logs and local analyzer output exist;
  - no bundle contains a matrix tree.

The untracked local `dee.cpp/pydee/pydee_core*.so` predates v5 and contains no
trace marker strings. It is not evidence that traced CUDA code built or ran.

## Why v6.2 produced no matrix evidence

Kaggle kernel version 14 printed:

```text
RUN_ID=LOCAL_RUN
COMMIT_EXPECTED=4d8ccf2
```

It then cloned `a2c0870` and failed notebook commit preflight. The native build
and matrix cells occur later and never executed.

The supervisor passed identity only in the local `kaggle` subprocess
environment. Kaggle's installed CLI has no `--env-vars` push option, so those
values could not become remote notebook environment variables. The supervisor
also:

- preserved the `KernelWorkerStatus.ERROR` prefix, so terminal ERROR was not
  recognized;
- parsed successful version output as `None`;
- allowed concurrent supervisors to archive/download/analyze the same paths;
- deleted partial downloads before attempting the terminal download;
- returned success for Kaggle COMPLETE without a required-artifact gate.

`NO_TRACE_ABORT` was therefore computed over a preflight-only log with zero
trace markers. It proves neither teardown safety nor instrumentation
connectivity.

## Native instrumentation connectivity defects

The v5 tracing source itself could not pass a fresh CUDA build:

- `DEE_TA_INSERT` received a moved `QuantizedExpert` object rather than
  `quantized.pinned` at the INT8/INT4 insertion sites.
- Persistent BF16/INT8/INT4 buffers used raw `cudaHostAlloc`, while teardown
  freed them through the traced wrapper. A connected run would classify them
  as unallocated pointers.
- Oracle GPU weights used raw `cudaMalloc`/`cudaFree` and were invisible.

These are instrumentation defects, not proof of the original teardown root
cause.

## Trust gate for the next run

The next push is allowed only after:

1. local supervisor and Python harness regressions pass;
2. a fresh remote CUDA build succeeds;
3. the imported extension path/hash matches the just-built binary;
4. all four `DEE_TA_SELFTEST` marker strings exist in that binary;
5. the traced 64-byte allocation/free reconstructs one allocation ID;
6. a real `Engine::init` increases the non-self-test allocation count;
7. the seven selected v4 variants advance through crash-safe progress records;
8. Kaggle COMPLETE is rejected unless the downloaded run-specific manifest is
   complete.

No teardown root cause or Milestone 3 completion is claimed by this report.
