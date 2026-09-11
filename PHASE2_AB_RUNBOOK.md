# Phase 2 matched 2xT4 A/B runbook

Status: prepared, not launched. The candidate is rooted at `ca8abd0`; no
remote/T4 campaign is part of this turn.

Use the same immutable commit, checkpoint/store identity, container, driver,
GPU placement, route inputs, seed, prompt/token workload, budgets, and
prefetch depth in both arms. The only intended software switches are:

| Arm | `phase2.enabled` | `host_enabled` | `vram_priority_fix_enabled` |
|---|---:|---:|---:|
| Baseline | false | false | false |
| Host-only | true | true | false |
| VRAM-only | true | false | true |
| Combined | true | true | true |

The first matched pair should be baseline versus the single arm selected by
the local causal review. Host policy is plain LRU; no frequency pinning,
compression, regime-C prewarming, or unvalidated custom device score is
eligible.

Before launch, require the local C++ and CUDA mechanism gates, exact packed-byte
and output parity, explicit host budget/slot geometry, and two identical T4
assignments. Predeclare warmups, measured repetitions, timeout, and failure
handling. Do not retry failed repetitions into a score.

Capture inclusive/per-layer latency, throughput, exact route/output status,
host/device hits and misses, fills, evictions, waits, SSD/H2D bytes per token,
host resident/peak bytes, and VRAM current/peak bytes. Keep service-time sums,
nested waits, and overlapping CUDA totals out of a critical-path speedup.

Do not claim an improvement from one favorable run or from the CPU/mock suite.
Seal only after every measured repetition passes exactness and the selected
switch attribution remains isolated.
