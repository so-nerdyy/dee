# MODEL_DESIGN.md — discrete-event critical-path simulator

Tool: `tools/exact_critical_path_sim.py` (standalone, stdlib-only).
Sweeps: `tools/critical_path_sweeps.py`. A/B ingestion: `tools/ingest_ab.py`.

## Scope

Simulates the **decode** phase of dee's exact DSV4 pipeline on the sealed v65
route journal (15 decode steps; prefill excluded — it is one 95.7 s cold pass
whose structure differs). GPUs are simulated in lockstep; the host bridge
synchronizes them once per step (measured: 1 inter-GPU handoff per forward).

## Dependency graph per layer request

```
submit ──> SSD read ──> host pack copy ──> H2D ──> packed residency ──> dequant+compute
            (shared       (per-GPU host     (per-GPU   (VRAM, sufficient   (per-GPU GPU
             disk)         CPU)               PCIe)     at 3.5 GiB fp4)     engine)
                └───────────── overlaps next layer's compute ─────────────┘
```

Sequential constraints honored (measured structure, not assumptions):
- The official router is **sequential across steps**: step k+1's expert ids do
  not exist until step k's 43 layers finish → no cross-step read overlap.
- Layers are sequential within a step (transformer dependency).
- Within a step, record staging may lead the consumer by
  `staging_lead_layers` (fit: 0 → on-demand, matching v52's timeline shape).

## Resources and service laws

| Resource | Model | Justification |
|---|---|---|
| SSD | Work-conserving shared-bandwidth drain: a batch of B bytes finishes at `max(t, last)/ + B/rate`; per-read latency grows with concurrency, aggregate rate fixed | v63/v65 lane A/B: 3 vs 6 lanes neutral, lanes×p50 ≈ 368/372 MB/s invariant → saturated disk; per-read p50 doubles when lanes double |
| Host pack | FIFO per GPU, 2.855 ms/record (v52 mmap→pinned / colds); buffer-reuse scenario replaces copy with 0.05 ms pointer swap | Codex's A/B is exactly this change; both modeled |
| H2D | FIFO per GPU at 5.54 GB/s fitted (microbench ceiling 5.38 GB/s idle4); per-record 13.369 MB copies; grouped mode = one copy per layer | idle4 measured rate; fitted ≈ ceiling under dual-GPU contention |
| GPU compute | Sequential per GPU; 0.4537 ms/record (v52 CUDA events, fp4 dequant incl.); overlaps copies | idle4 overlap_fraction 0.22; v52 stage times |
| Host orchestration | 1.2 ms/layer serial slice (calibrated) | fills the non-disk intercept; v65 has 86 host syncs/step + dispatch |
| VRAM residency | Sufficient at 3.5 GiB fp4 (v65 ends fully resident); resident-hit counts come from the sealed per-step accounting | P2.3 analysis; engine_stats |

## Demand model (no invented counts)

Per step, the plan uses v65's sealed `per_token_accounting`:
- requests = 258 (43 × top-6),
- resident hits = measured per step,
- H2D copies = measured per step (records transferred),
- SSD reads = measured `storage_requests` per step
  (× 13,369,344 B = `storage_bytes` exactly — validated in tests),
- pack-sourced transfers = H2D copies − SSD reads.

Split across GPUs by the sealed per-GPU request totals (2613:2488) and across
layers proportional to the journal's per-layer unique-expert demand.

## What-if parameters (Phase C knobs)

`host_buffer_reuse`, `read_lanes` (inert by design — see below), `h2d_streams`,
`staging_lead_layers`, `grouped_h2d`, `grouped_dispatch`,
`host_orch_reduction`, `queue_depth`, `compute_overlaps_copies`,
`h2d_gbps`, `ssd_aggregate_mb_s`, `gpu_compute_per_record_ms`.

Lane count deliberately does not change the drain rate: the model encodes the
measured fact that the disk saturates before lanes do (v63 vs v65). Any
future SSD with different behavior would need this law re-fit — flagged in
NEXT_EXPERIMENTS as BLOCKED_BY_MISSING_MEASUREMENT for >6 lanes.

## Excluded (per brief)

Pruning/skipping, expert substitution, approximate codecs, KT BF16 CPU
execution, STQ/IQ2, router changes — none are representable, by design.

## Known simplifications (explicit)

1. Steps simulated independently; the ~1.2 ms/layer host slice absorbs
   inter-step handoff effects (no cross-step overlap exists to model).
2. Pack/Read/H2D pipelining is batched per layer rather than per record —
   equivalent under FIFO + saturated disk (validated against per-record
   ordering on step 0; difference < 0.1 %).
3. GPU compute serialized with itself only; multi-stream compute overlap is
   not representable in the current engine (d2d + gemm share one stream
   timeline in v52).
4. Inter-GPU handoff (1 d2h + 1 h2d of 32 KiB) folded into the sync tail.
5. No VRAM-slot contention model: v65 never evicts under the 3.5 GiB fp4
   budget during decode (resident hits come from the previous forward's
   resident set), so capacity effects are out of scope for this trace.
