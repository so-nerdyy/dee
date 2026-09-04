# Consumer harness — final summary

Branch: `research/consumer-harness`. Worktree: `../dee-consumer`.
Track: hardware migration / measurement prep. No model semantics touched, no
KT production CPU execution enabled, no merges, sealed campaign evidence intact.

## What was built

| Deliverable | Path |
|---|---|
| Host qualification (one command) | `tools/qualify_host.py` |
| Storage benchmark | `tools/bench_storage.py` |
| H2D transfer benchmark | `tools/bench_h2d.py` |
| Memory budget estimator | `tools/memory_budget.py` |
| Benchmark schema | `research/consumer-harness/BENCHMARK_SCHEMA.md` |
| Migration checklist | `research/consumer-harness/MIGRATION_CHECKLIST.md` |
| This summary | `research/consumer-harness/FINAL_SUMMARY.md` |
| Tests (28, all passing) | `tests/test_{memory_budget,bench_storage,bench_h2d,qualify_host}.py` |

`tools/qualify_host.py` is the Phase A capability tool (human summary + JSON
artifact) and the Phase D one-command qualification. It orchestrates the
storage / H2D / budget modules and separates every output into MEASURED /
DERIVED / UNKNOWN.

## Canonical geometry used (not assumed)

From `dee.cpp/.../official-source/config.json`, `T4_FEASIBILITY.md`, P2.3 doc:
DSV4-Flash-0731, hidden 4096, MoE-inter 2048, 256 routed experts, 43 layers,
top-6, MXFP4 packed record **13,369,344 B (12.75 MiB)**, FP16 48 MiB, INT8
24.75 MiB, dense ≈ 8.84 GB. Reference derivation: 623 MXFP4 slots per
16 GiB-style budget vs 11,008 routed experts (~23% cached with a 24 GiB RAM
budget) — eviction is mandatory, stated as derived arithmetic.

## Phase F — cost-model export

`host.json → cost_model` exports `t_h2d_ms_pinned_by_size_bytes` (measured
pinned-H2D medians at 1/4/8/13/16/32 MiB), SSD seq-read by block + filesystem
+ cold-cache note, CPU model/counts/ISA, and `gpu_execution: null` (dee
kernels deliberately not benchmarked here). Status:
`EXPORT_ONLY_NOT_INTEGRATED` — nothing is wired into production scheduling or
any KT q* cost model yet, by design.

## Verification on this dev machine (honest, not a target result)

- `python -m pytest tests/ -q` → **28 passed** (Windows, no CUDA).
- `tools/qualify_host.py` degrades as specified: system measured (Ryzen 7
  5825U, 8C/16T, 15.3 GiB RAM), GPU/transfer UNKNOWN (no CUDA), NTFS storage
  measured, ISA UNKNOWN on Windows without py-cpuinfo (labeled, not inferred).
- No checkpoint was needed to qualify the host.

## Claim discipline (held throughout)

- No DSV4 tok/s predicted from bandwidth; no 20 TPS feasibility claim; 4-bit
  weights never presented as native FP4 execution (CC-derived feature flags
  carry an explicit unpack-cost disclaimer); no 5070 Ti / 4090 / 5090 result
  claimed — only the procedure to measure one.
- STQ/IQ2-style inputs allowed via `--bits-per-param` strictly as
  approximate/research-only arithmetic with no quality claim.
- `qualify_host` verdicts refuse performance prediction in both human and JSON
  output; tests assert the refusal structurally.
