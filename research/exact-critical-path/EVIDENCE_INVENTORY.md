# EVIDENCE_INVENTORY.md — sealed exact evidence for the critical-path model

Track: `research/exact-critical-path` (offline analysis only).
Every number below lists its source file. Nothing here was re-measured by
this track; nothing sealed was modified. Missing metrics are marked
**UNKNOWN**, not inferred.

## Primary evidence set

### E1. v65 sealed terminal run — `dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/`

Newest sealed exact DSV4 evidence on the dual-T4 Kaggle host. ACCEPT_CORRECTNESS.

| Item | Value | Provenance |
|---|---|---|
| run_id | `p2.3-dee4-storage-readahead-ab1-candidate-q6l6-v65` | result.json |
| commit | `c3182fc4ecfad5f5bae8f8ad548087cc16a1e978` | result.json |
| decode wall | **72.267 s** | result.json `decode_wall_s` |
| decode tokens | 15 | result.json |
| decode TPS | 0.208 | result.json |
| prefill | 95,693.9 ms (7-token prompt) | profile.json |
| per-step walls | 15 values, 3,573–6,876 ms | profile.json `decode_timings_ms` |
| route journal | 688 records, 16 forwards, chain-sealed, sha `ee5c9568…` | result.json route_journal |
| selections | 5,676 total (43 layers × top-6 × 16 forwards … per-record `token_rows` vary) | routed_experts.jsonl |
| expert record | 13,369,344 B packed fp4, uniform | dee4_trace_validation.record_bytes |
| store | 2,364 unique layer-expert records, 31.61 GB | dee4_trace_validation |
| engine config | cache fp4-packed, 3.5 GiB/GPU budget, 6 lanes, qd 6 | result.json engine_config |
| resident hits | 328 (cuda0) / 327 (cuda1) | engine_stats |
| cold loads | 2,285 / 2,159 | engine_stats |
| H2D | 30.55 GB / 28.86 GB over 2,285 / 2,159 copies | engine_stats |
| SSD reads | 1,390 / 1,091 records; p50 215.5 / 220.8 ms; overlap 77.9 % | expert_store |
| host pack | hits 1,223 / 1,395, misses 1,390 / 1,091, evictions 708 / 409 | host_pack |
| per-step accounting | requests/hits/evictions/storage_bytes/H2D per step | profile.json per_token_accounting |
| bridge | 1,376 host syncs, 688 native calls, 688 route d2h | result.json bridge_counters |
| stage profile | **disabled** (`profile_stages: false`) → GPU compute UNKNOWN for this run | profile.json |

Per-step decode accounting (both GPUs summed), from `per_token_accounting`:

| step | requests | resident hits | H2D copies | SSD reads (storage_requests) |
|---|---|---|---|---|
| 1..15 | 258 each | 67,74,71,49,38,41,28,31,31,43,40,35,34,37,36 | 191,184,187,209,220,217,230,227,227,215,218,223,224,221,222 | 96,69,80,111,125,114,96,80,56,64,79,61,69,91,61 |

Consistency checks performed (this track, read-only):
- `storage_requests × 13,369,344 B = storage_bytes` exactly, every step.
- `h2d_copies` ≈ resident misses; `h2d_bytes` matches `h2d_copies × record_bytes` per step (±128 B rounding in the journal summary).
- decode-only storage traffic = 1,252 records = **16.74 GB** (the 33.17 GB `storage_bytes_total` includes prefill's 1,229 records).

### E2. v52 stage-profiled run — `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/v52-evidence-20260830T133749Z/`

Only available stage decomposition of the real fp4 DSV4 pipeline (same route
journal; profiler instrumentation inflates wall: 138.7 s vs v51's 133.0 s).
Used for **stage rates**, never for wall reconstruction.

| Stage (cuda0) | Measured | Provenance |
|---|---|---|
| GPU gate/up/silu/down projection | 467.8 / 207.7 / 16.7 / 302.7 ms | profile.json gpu_ms |
| GPU d2d gather+scatter | 24.9 + 16.9 ms | profile.json gpu_ms |
| GPU h2d (CUDA-event) | 2,534.9 ms over 2,285 copies | profile.json gpu_ms |
| CPU mmap→pinned (per cold) | 6,523.7 ms total → **2.855 ms/record** | profile.json cpu_ms |
| CPU tensor_resolution | 127,085.5 ms (profiler-inflated; NOT used as-is) | profile.json cpu_ms |
| CPU sync | 3,471.4 ms | profile.json cpu_ms |
| eviction_selection | 53.8 ms | profile.json cpu_ms |

Derived routed-compute rate (v52, per record): (467.8+207.7+16.7+302.7+41.8)/2285
= **0.4537 ms/record** — includes dequant scratch work on the fp4 path.

### E3. v63 / v64 / v65 lane A/B — `dee.cpp/tmp/v63-terminal-fetch-20260903T0048Z/`, `v64-…`, `v65-…`

| Run | lanes | decode wall | p50 read | lanes × p50 |
|---|---|---|---|---|
| v63 baseline q6l3 | 3 | 71.315 s | 109.1 ms | 368 MB/s |
| v64 record-order candidate | 3 | 71.804 s | 103.7 ms | 363 MB/s |
| v65 readahead candidate | 6 | 72.267 s | 215.5 ms | 372 MB/s |

Measured conclusion (theirs, sealed): 3→6 lanes is **neutral** on wall; the
aggregate disk throughput (~370 MB/s under concurrent reads) is the invariant.
This is the signature of a saturated shared disk.

### E4. campaign-idle4 microbench — `dee.cpp/benchmark_reports/campaign-idle4/`

Synthetic fp16 single-GPU T4 microbench (32-token steps). Used only as
stage-rate anchors, clearly labeled synthetic:

- pinned H2D: 4.009 GB / 0.745 s = **5.38 GB/s** (prepack-controlled/end-to-end.json).
- copy/compute overlap exists: overlap_fraction 0.222, copy_active 743.4 ms,
  compute_active 539.3 ms (timeline-9a520b3.json summary).
- pinned-ring depth sweep 8/12/16/24: wall 1.224–1.242 s, spread ≈ noise
  (ring-*.log) → pinned pool depth is not a lever at this scale.

### E5. Historical bounds — `dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/`

- STORAGE_ROOFLINE.md: `/kaggle/working` SSD p50 123–128 MB/s sequential
  single-stream; dataset mount ~13 MB/s. FP4-era runs read from a repacked
  trace store under `/tmp` (overlay), not `/kaggle/input`.
- P2.3_FP4_VRAM_ANALYSIS.md: FP4 replay hit rates and Belady ceilings by
  budget; 3.5 GiB/GPU → 33.7 % predicted routed hits (replay), motivating the
  3.5 GiB budget used since v47.

## UNKNOWN register (not inferred anywhere in this track)

| Metric | Status |
|---|---|
| GPU dense/non-MoE work per token (DSV4) | UNKNOWN — no stage profile isolates it |
| GPU routed compute in the unprofiled v65 run | UNKNOWN — stage profile off; v52 rate used as proxy |
| Host orchestration per layer (bridge sync, dispatch) | UNKNOWN directly; fitted 1.2 ms/layer in the model |
| Effective H2D rate under dual-GPU load + compute contention | UNKNOWN; fitted 5.54 GB/s ≈ microbench 5.38 GB/s |
| Per-read flash latency vs bandwidth split | UNKNOWN — model uses aggregate-bandwidth drain with zero fixed latency |
| In-flight H2D↔D2D contention penalty | UNKNOWN |
| Exact intra-step staging lead achieved by v65's fill pipeline | UNKNOWN — model fits lead=0 (on-demand) |
| Codex host-buffer A/B results | IN PROGRESS (Codex) — ingestion path ready, not consumed |
