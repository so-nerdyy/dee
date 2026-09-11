# Phase 2 matched 2xT4 A/B runbook

Status: prepared, not launched. The first remote run stays blocked until Flash
returns a corrected causal host-policy result and exactly one policy is selected.
No codec, compression, original `b7b9c7f` frequency policy, or unvalidated VRAM
priority fix is eligible for this A/B.

## Pair definition

Run baseline and candidate from the same immutable dee.cpp commit, checkpoint
identity, shard/store integrity identity, driver/container, GPU assignment,
prompt/token workload, route inputs, seed, and process placement. Use two T4s in
both arms. The only intended software difference is:

| Arm | Phase-2 state | Policy state |
|---|---|---|
| Baseline | `phase2.enabled = false` | Existing runtime behavior |
| Candidate | `phase2.enabled = true` | Exactly one corrected Flash-selected host policy; optional VRAM fix only if independently reconfirmed |

The candidate must use an explicit immutable `model_identity`, explicit host
slot geometry/budget, and the selected `HostPlacementPolicy`. There is no
implicit environment or CLI enablement. If the selected policy is unavailable,
the candidate fails closed and the pair is not scored.

## Preflight gates

1. Rebuild the exact commit with CUDA sm_75 support and run
   `test_phase2_host_tier`, `test_expert_store`, all existing FP4 CUDA tests,
   and `test_phase2_cuda`.
2. Verify the baseline and candidate produce identical route records, packed
   bytes, and executor outputs on the local exactness fixture. Record the
   checkpoint/store identity and the full Phase-2 configuration in each arm.
3. Verify both arms have the same two-GPU visibility, device ordering, memory
   limits, prefetch depth, and host process limits. Do not change allocation or
   quota in response to transient launch failures.
4. Predeclare warmup count, measured repetition count, timeout, and failure
   handling. A failed or incomplete repetition is unresolved; it is not silently
   retried or converted into a score.

## Measurements

Capture the existing Engine acceptance/performance fields and the Phase-2
snapshot for the same lifetime interval:

- inclusive and per-layer latency, throughput, and p50/p95;
- `SSD_bytes_per_token`, `H2D_bytes_per_token`, and their denominator;
- host/device hit and miss counts, coalesced fills, evictions, fills, failures,
  and capacity rejections;
- host/device wait boundaries, pageable fallback waits, and GPU waits when
  measured (unknown remains unknown);
- host allocated/pinned/resident and peak resident bytes, plus current and peak
  VRAM arena bytes;
- exact route/output comparison and any unresolved or failed repetitions.

Do not add service-time sums, nested waits, or non-additive CUDA totals to a
critical-path speedup. A local mock or CPU run validates mechanism only; it is
not 2xT4 performance evidence.

## Scoring and stop conditions

Seal the pair only if every measured repetition passes exactness and mechanism
gates, the candidate's host policy is the single corrected causal selection,
and the VRAM change (if any) is reconfirmed in the same evidence. Report the
replicate distribution and conservative/evaluated-only coverage separately.
Do not claim a speedup from a single favorable repetition, from a mock result,
or from a changed workload. If causal attribution is not isolated, mark the
result provisional and return to Flash rather than selecting another policy.

No Kaggle/remote job is part of this local implementation turn. Before any
future remote launch, pause the active goal and create the requested cron guard
before starting the next job.
