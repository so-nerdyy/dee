# CAMPAIGN_FAULT_INJECTION.md — fault-injection hook SPEC (post-merge work)

Status: **specification only — do not implement on this branch.** Written for
the GPU Batch #1 campaign payload (`feat/phase2-campaign-payload`). The hooks
below are T1-adjacent code changes that land on a follow-up commit; until then
the campaign's audit-probe segment (PHASE2_T4_CAMPAIGN.md §3.2) runs the
probes that are reachable today and records the rest as observational notes.

Context: the verification audit (research/phase2-host-tier-verification @
e63a3bc, `PHASE2_VERIFICATION_AUDIT.md`) returned GO conditioned on three live
exercises static review could not settle. Probe (b) and half of probe (c) are
already reachable through existing knobs; probe (a) needs a real fault seam.

## Design rules (inherited, non-negotiable)

1. **Never compiled into production.** The seam exists only under a test
   build flag (`-DDEE_FAULT_INJECT=ON`, default OFF) or inside a test-only
   translation unit (`dee.cpp/tests/fault_inject.h`). No env var, no config
   field, and no code path may arm it in a `dee_cli`/engine production build —
   a stuck injector must be impossible in a scored run.
2. **Exactness contract untouched.** A fault injection changes *which CUDA
   call fails and when*, never which bytes move or which expert executes.
   Injected failures must be indistinguishable to the engine from a real
   `cudaError*` return.
3. **Deterministic.** Injection is keyed on a call ordinal
   ("fail the n-th `cuda_submit_host`"), never on timing or randomness —
   the same rep must be reproducible.
4. **Evidence-shaped.** Each probe emits a small JSON/stdout verdict line the
   P2 mechanism segment can capture verbatim into `segment_mechanism.json`.

## Injection point 1 — `cuda_submit_host` failure → event cleanup (probe a)

Target: `dee.cpp/src/async_prefetcher.cpp` `AsyncPrefetcher::cuda_submit_host`
(~lines 724-782 on the campaign base).

The suspected defect: on an injected `cudaMemcpyAsync` failure the error path
(line ~749 `if (!ok)`) drains the stream, sets `dma_complete`, returns false —
but the `cudaEvent_t` created at line ~733 (`DEE_TA_EVENT_CREATE_FLAGS ...
"phase2_event"`) is stored on `transfer.event` and only destroyed in
`reset()` (line ~659). N injected failures ⇒ N leaked event handles until
teardown. The audit wants that proven or disproven live.

**Hook shape (spec):**

```cpp
// async_prefetcher.h — only declared under DEE_FAULT_INJECT
#ifdef DEE_FAULT_INJECT
    // Arm a one-shot failure: the n-th cuda_submit_host call's
    // cudaMemcpyAsync returns cudaErrorUnknown without enqueueing work.
    // Thread-safe; test-only; no effect when count is already past n.
    void fault_inject_submit_fail_at(long n);
    // Test-visible handle accounting for the leak assertion.
    long fault_events_live() const;   // events created minus events destroyed
#endif
```

```cpp
// async_prefetcher.cpp — inside cuda_submit_host, replacing the direct call:
#ifdef DEE_FAULT_INJECT
    if (fault_submit_at_ == next_submit_ordinal_++)
        ok = false, last_cuda_error_ = cudaErrorUnknown;   // skip the real call
    else
#endif
    ok = DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(...), "phase2 packed H2D");
```

Behavior required of the hook: the failure must surface *through the same
`ok` boolean* the real call would produce — the function's existing error
tail (`cudaStreamSynchronize` + `dma_complete` + `return false`) executes
verbatim. `fault_events_live()` counts `cudaEventCreate*` minus
`cudaEventDestroy` across the prefetcher's lifetime so the test asserts
flatness after N injected failures and one subsequent good transfer.

**Probe binary:** `tests/test_phase2_fault_inject.cpp` (new; built only when
`DEE_FAULT_INJECT=ON` AND `DEE_CUDA=ON`). Sequence:

1. Build the test_phase2_cuda fixture (host tier + VramCacheManager on the
   cuda arena + prefetcher with `init(true)` + `DeviceExpertTier` scope).
2. `fault_inject_submit_fail_at(0..2)`; issue three `device.stage()` calls —
   each must fail cleanly (`stage` returns false, `device_failures`
   increments).
3. Assert `fault_events_live()` is flat (0 net after the aborted transfers,
   modulo any events still legitimately owned by in-flight good transfers).
4. Issue one good `stage` + `wait` — must succeed; byte-exact copyback.
5. `synchronize_all()` + `reset()` — `in_flight()==0`, host `leased_slots==0`.

Success (per campaign §3.2a): N injected failures leave event-handle count
flat and the prefetcher still serves a subsequent good transfer.

## Injection point 2 — forced pageable fallback (probe b) — mostly exists

Already reachable **without a hook**:

- `HostTierConfig.try_pin=false` — no registration attempted; slots stay
  usable pageable, `pin_failures` stays 0. Engine-level probe:
  `phase2_host_try_pin=false` in run_config / `NATIVE_PHASE2_TRY_PIN=0`.
- A `HostMemoryBackend` whose `pin` returns false — exactly what
  `test_phase2_cuda`'s `run(false)` half does — produces the
  `pin_failures>0` variant.

What the probe asserts on the live run: `experimental_pageable_wait_ms_` /
TierMetrics `pageable_fallback_wait_ms` is nonzero and monotonically grows
under load; no crash; byte-exact outputs. The P2 segment cites
`test_phase2_cuda` PASS as covering case (ii); an engine-level `try_pin=false`
probe rep is optional Batch-#1 evidence (one extra A2-style rep, unscored).

No new hook required. If reviewers want the *engine* to simulate
registration failure under `try_pin=true`, add a
`HostMemoryBackend::pin` wrapper at the *test* layer only — the backend
struct is already injectable, so this is a test-side policy wrapper, not a
production seam.

## Injection point 3 — in-flight `synchronize_all`/`reset` (probe c)

Target: `synchronize_all` (~lines 618-652) and `reset()` (~654-669). The
audit LOW finding: the `std::terminate` at line ~624 is keyed on the armed
flag (`experimental_host_tier_`), not on actual in-flight managed work.

Existing coverage: `test_phase2_cuda`'s pinned half parks a managed H2D
behind a `cudaLaunchHostFunc` gate — proving `wait_on_stream` cannot release
the host source early (`leased_slots==1` while gated). Not yet covered:
calling `synchronize_all()` / `reset()` while a managed transfer is
genuinely in flight.

**Probe shape (no new hook needed — reuse the gate):**

1. Arm the same `cudaLaunchHostFunc` gate used by `test_phase2_cuda`.
2. `device.stage()` a managed transfer (it queues behind the gate —
   genuinely in flight).
3. Call `prefetcher.synchronize_all()` on a second thread; assert it blocks
   until `gate.release()`, then completes all managed transfers without
   `std::terminate`; `in_flight()==0`; host `leased_slots==0`; a subsequent
   `stage` still works.
4. Variant: `reset()` while gated — must drain + destroy events + free
   leases cleanly; log the path if a sticky CUDA error ever trips the
   armed-flag terminate.

If a genuinely *failing* in-flight transfer is wanted (to reach the
`std::terminate` line), combine with injection point 1's hook armed at the
same ordinal — the spec deliberately leaves that combination to the
follow-up implementation.

## Wiring summary for the implementer

| Piece | Where | Gate |
|---|---|---|
| `fault_inject_submit_fail_at` / `fault_events_live` | `AsyncPrefetcher` public test seam | `#ifdef DEE_FAULT_INJECT` + CMake option `DEE_FAULT_INJECT` default OFF |
| probe binary | `dee.cpp/tests/test_phase2_fault_inject.cpp` | `if(DEE_CUDA AND DEE_FAULT_INJECT)` in CMakeLists; ctest name `test_phase2_fault_inject` |
| env gating | none required — compile-gated only | env-var arming explicitly rejected (rule 1) |
| P2 segment entry | campaign driver `p2_mechanism()` | add `test_phase2_fault_inject` to `CUDA_MECHANISM_TESTS` when the binary exists |

Until this lands, the campaign driver emits `NOT_RUN`/`PARTIAL` probe
records (see `segment_mechanism.json` fields `audit_probes`) — a verdict
note, never a gate, per PHASE2_T4_CAMPAIGN §3.2's degradation rule.
