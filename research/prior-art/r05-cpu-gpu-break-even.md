# R5 — CPU-vs-GPU miss-execution break-even (the Fiddler seam)

- Track: R5 — on a device-cache miss, execute the expert on the CPU where
  its weights already are, instead of moving the 12.75 MiB record to GPU
  (Fiddler's inversion: move the ~32 KiB activation round-trip, not the
  record).
- Branch: `research/prior-art-r05` @ `dc78dc4` (worktree `.freebuff/wt/r05`).
- Status: analysis only. No engine changes, no remote spend, nothing
  implemented (Phase-2 rule). Every number carries a label: MEASURED
  (sealed live 2×T4 evidence / in-repo tests), DERIVED (arithmetic on
  measured inputs, derivation stated), SIMULATED (sealed-journal replay),
  PAPER-REPORTED, UNKNOWN.
- Inputs: `research/route-pipeline/LIVE_PROFILE_RESULTS.md` +
  `evidence-live/` (sealed live host/sync profile, engine `217a333` +
  profiler `aae0f41a`), `STORAGE_VERDICT.md`,
  `research/phase2-concurrent-fill/` (SERIALIZATION_VERDICT, DESIGN),
  `research/phase2-legal-prefetch/LEGAL_PREFETCH.md`,
  `research/kt-cpu-bridge/` + `dee.cpp/experiments/kt_cpu_bridge/`.
- Companion: R6 (`research/prior-art/r06-heterogeneous-sink-audit.md`)
  audited the interface seams a CPU sink needs (gaps G1–G10); this file
  prices the decision and rules on the exactness question. R9 quantified
  the payload asymmetry (~408:1) this track exploits.

## Verdict — the break-even headline

**The contest is not over the 12.75 MiB transfer — it is over the ~2.4 ms
of caller-thread staging each miss costs after the fill that both paths
pay.** Per-miss, measured on the sealed bank:

- `T_gpu_path(miss, host-resident)` ≈ **2.4 ms caller-side** (stage-enqueue
  share: pinned gather `memcpy` of 13,369,344 B + `cudaMemcpyAsync` submit
  + cache bookkeeping) + ~1.2 ms device H2D (hidden; readiness ≈ 0) +
  ~0.35 ms device exec. Exposed ≈ **2.4–3.6 ms**.
- `T_gpu_path(miss, cold)` = the above **plus** the bank read: 18.9 ms
  single-flight anchor / 57.4 ms mean (162 p95) batched service / ~34 ms
  mean exposed per miss inside the 65.1 ms/row fill bucket. The fill is
  **invariant across sink choice** — a CPU executor still needs the record
  in RAM before it can compute.
- `T_cpu_path` ≈ `t_d2h_act` (~0.02–0.06 ms, 16 KiB) + `t_cpu(m)` +
  `t_h2d_out` (~0.01–0.05 ms, 16 KiB) + join. `t_cpu(1)` is **UNKNOWN** at
  real geometry; DERIVED band: ~2–6 ms for a tuned AVX2/AVX-512 single-core
  kernel, ~15–40 ms for the portable fp32 reference, ~1–3 ms for a 4-vCPU
  parallel tile.

Therefore:

1. **Serial (caller-thread) break-even: `t_cpu*(m=1) ≈ 3–4 ms/expert`.**
   Below it, EXECUTE_FROM_HOST wins every miss head-to-head; above it,
   serial CPU execution loses on host-resident misses and is catastrophic
   on cold misses (adds `t_cpu − 2.4 ms` on top of an already-blocking
   pread). The portable reference executor is on the losing side; a
   KT-shape AVX2 kernel is borderline; KT-AMX-class numbers
   (PAPER-REPORTED 21 TFLOPS BF16 on Xeon4 → ~2.4 µs compute, memory-floor
   ~0.5–1.3 ms) would win clearly — but Kaggle's CPU has no AMX and
   likely no AVX512-BF16 (probe, never assert — UNKNOWN).
2. **Overlapped (worker-pool) break-even: `t_cpu ≤ ~25–30 ms` is
   wall-free at decode.** The caller is blocked ~65.1 ms/row on fills
   anyway; a CPU worker that starts the moment a lease is Ready hides
   under the *next* miss's pread (34–96 ms shadow per miss, ~1.9
   misses/row). In this regime the answer does not depend on PCIe at all —
   it depends on having the submit/join seam (R6 G4) and ≥1 spare core.
3. **44.2% of device misses are already host-resident** (4,444 device
   misses vs 2,481 preads per response → 1,963 host hits, MEASURED sealed
   counters). Those are free-fill misses: the CPU path saves the whole
   ~2.4–3.6 ms staging tail on each with zero storage involvement.
4. **Whole-decode prize bound: ≈ the stage-enqueue bucket, ~6–9 s of the
   66.2 s decode wall (~10–14%), DERIVED upper bound** — plus second-order
   VRAM relief (60.5% of sealed-window records never repeat; diverting
   them stops VRAM insert+evict churn). The 42.0 s fill bucket is
   untouched — a CPU sink cannot fix storage, and on this bank storage is
   the wall.
5. **Decode vs prefill: decode is the favored regime.** Prefill dedups to
   22–41 unique experts/layer, each serving only ~2–8 tokens — CPU GEMM
   amortizes weight decode across tokens (KT's 4×4 tile) but `t_cpu(m)`
   still scales ~linearly while the GPU GEMM gets *more* efficient with m.
   Serial CPU prefill loses; overlapped prefill is marginal (fill shadow
   1.45 s/layer vs ~30 experts × 5–30 ms of CPU work). This matches
   Fiddler's published crossover (CPU wins small batch, GPU wins large).

Recommendation: a `MissExecutionPolicy{TRANSFER_TO_DEVICE,
EXECUTE_FROM_HOST, HYBRID}` is **exact-legal** under dee's contract as a
post-route placement decision provided (a) each executor numeric class
passes the same predeclared gates vs the fp32 trusted reference, (b) the
policy is a deterministic function of logged state (or journaled), and
(c) the sealed run is re-executed per executor mix — same precedent as
the FP16-cache and INT8-transfer numeric paths. The sealed 16/16 contract
is compatible; it needs a new seal record field naming the executor
class, not a new contract. Details in §5–6.

## 1. The two paths, precisely

Per authoritative route `(layer, expert_id)` that misses the device tier:

```
TRANSFER_TO_DEVICE (today):
  host.acquire(record)  ── pread 12.75 MiB if ¬resident (CALLER thread,
                           phase2-serial; engine.cpp:3153 →
                           expert_tiers.cpp:126-146 →
                           host_expert_tier.cpp:245-270 →
                           expert_store.cpp:631)
                        → pinned-slot gather memcpy (6 regions → one
                           contiguous staging buffer,
                           async_prefetcher.cpp:867-888)
                        → cudaMemcpyAsync H2D (12.75 MiB; pinned fast
                           path, pageable fallback :779-789)
                        → VRAM admission (evict victim at 281-slot arena)
                        → FP4→FP16 dequant (cuda_convert.cu:231-245) +
                           FP16 GEMM fp32-acc + fused SwiGLU
                           (swiglu_cuda.cu)
                        → fp32 raw row → weighted_combine_fp16_kernel
                           (device, order-stable, swiglu_cuda.cu:86-116)

EXECUTE_FROM_HOST (proposed):
  host.acquire(record)  ── identical pread if ¬resident; identical
                           Filling→Ready lease semantics (lease IS the
                           read certificate — R6 Q2)
                        → PackedExpertView borrow of the 6 regions
                           (record layout [gate_w][up_w][down_w][g_s]
                           [u_s][d_s] already matches the bridge view —
                           expert_tiers.cpp:56-79; zero-copy)
                        → activation row: already host fp32 on host-API
                           paths; needs a ~16 KiB D2H on the device path
                           (R6 G6)
                        → CPU expert compute (fp32 reference class or
                           KT-shape bf16 class), routing weight applied
                           before w2 (dee placement) — output fp32
                           [hidden]
                        → fp32 result row → combine input (host-API:
                           memcpy into experts_out, zero ordering; device
                           path: 16 KiB H2D before the existing
                           per-layer output sync — R6 Q4c)
                        → lease release → slot returns to LRU
```

What is NOT on the CPU path: no pinned gather, no `cudaMemcpyAsync`, no
VRAM block, no eviction, no device pin. What IS still on it: the cold
pread (storage is sink-agnostic) and the caller's `host.acquire` under
today's serial phase2 path.

Payload asymmetry (canonical, DERIVED): record 13,369,344 B vs activation
round-trip 32 KiB (4096 × 4 B in + out, fp32) = **407.9:1**. Per decode
token worst case: 258 routed calls × 32 KiB ≈ 8.06 MiB of activation
traffic vs 258 × 12.75 MiB ≈ 3.22 GiB of weight traffic — the Fiddler
premise holds at dee geometry (cross-ref R9 §1).

## 2. Measured inputs (all sealed 2×T4 SM75 unless noted)

Decode attribution, 645 rows (tokens 7–21), wall 66.233 s —
`LIVE_PROFILE_RESULTS.md`, `evidence-live/host-sync-attribution.json`:

| Component | Total | Per row | Tier |
|---|---:|---:|---|
| fill_wait (storage reads, host-blocked) | 41.990 s (63.4%) | 65.1 ms | MEASURED |
| stage_enqueue_wait (H2D submit + cache ops, host-serial) | 9.436 s (14.2%) | 14.6 ms | MEASURED |
| native_output_sync (required drain) | 4.894 s (7.4%) | p50 7.99 / p95 8.08 ms | MEASURED |
| expert_compute dispatch (host span) | 0.256 s | 0.40 ms | MEASURED |
| decode dispatch (dequant launch) | 0.044 s | 0.07 ms | MEASURED |
| gather_scatter | 0.114 s | 0.18 ms | MEASURED |
| readiness_wait | 0.027 s | ~0.04 ms (transfers always ready) | MEASURED |
| route_d2h_host_wait | 0.015 s | 0.022 ms p50, 0.16 max | MEASURED |
| combine | 0.139 s | 0.22 ms | MEASURED |
| shared expert (device-serial) | 0.308 s decode / 24.1 s prefill | — | MEASURED (device event) |
| unknown (dense attn + orchestration + journal) | 9.196 s (13.9%) | — | MEASURED bucket |
| closure | 0.861 | — | MEASURED |

Miss-stream structure (sealed counters + sealed-journal sim, validated
±1: `phase2-concurrent-fill` validation.json, `legal_prefetch` summary):

| Quantity | Value | Tier |
|---|---:|---|
| requests/response (16 forwards × 43 layers, topk 6) | 5,099 | MEASURED |
| device hits | 655 (12.8%) | MEASURED |
| device misses → host hit (no pread) | 1,963 (38.5%) | DERIVED (4,444 − 2,481) |
| device misses → host miss (pread) | 2,481 (48.7%) | MEASURED |
| H2D bytes | 3.71 GB/token → ~59.4 GB/response | MEASURED |
| storage bytes | 2.07 GB/token → 33.17 GB | MEASURED |
| decode miss batch m per layer | {0..6}, mean ≈ 1.9 | SIMULATED (sealed-journal) |
| prefill unique experts per layer | 22–41 | SIMULATED (sealed-journal) |
| VRAM arena / host LRU | 281 slots (3.5 GiB) / 682 slots (8.5 GiB) per GPU | MEASURED config |
| records that ever repeat on sealed window | 935/2,364 (39.5%) | MEASURED journal |

Per-record service times:

| Term | Value | Tier |
|---|---:|---|
| single-flight pread (12.75 MiB) | 18.9 ms anchor; 18–19 ms measured | MEASURED (replay) |
| per-request worker service, sealed whole-run | 96 ms mean (239.1 s / 2,481) | MEASURED |
| per-request under 3-way device sharing | 57.4 ms mean / 162 ms p95 | MEASURED (replay) |
| bank aggregate ceiling | 0.29–0.37 GiB/s; 96% in-batch busy | MEASURED |
| exposed fill per miss | ~34 ms (65.1 ms/row ÷ ~1.9 misses) | DERIVED |
| H2D device per record | ~1.19 ms → ~11.2 GB/s effective (≤5.3 s / 59.4 GB, hidden) | DERIVED |
| stage submit per stage() call | ~1.85 ms avg (9.436 s / 5,099); ~2.4 ms per miss (gather+submit) | DERIVED |
| GPU expert exec, m=1 | ~0.3–0.4 ms device (1.5 s GEMMs / ~4,400 dispatches) | DERIVED |
| pinned gather memcpy (12.75 MiB) | ~0.7–1.3 ms at 10–20 GB/s host copy | DERIVED |
| activation D2H (16 KiB) | ~0.02–0.06 ms (route-D2H floor 0.022 ms @ 168 B) | DERIVED |
| result H2D (16 KiB fp32) | ~0.01–0.05 ms | DERIVED |
| `t_cpu(1)` Kaggle host, real geometry | **UNKNOWN** — DERIVED bounds below | UNKNOWN |

`t_cpu(1)` bounds for H=4096/I=2048, 25,165,824 params ≈ 50.3 MFLOP:

| Kernel class | Bound | Tier |
|---|---:|---|
| Memory floor (stream 12.75 MiB @ 10–25 GB/s single-core) | 0.5–1.3 ms | DERIVED |
| FMA floor (50.3 MFLOP @ 32–64 GFLOP/s AVX2 fp32/core) | 0.8–1.6 ms | DERIVED |
| Tuned AVX2/AVX-512, single core (LUT dequant + fmadd, KT-shape) | ~2–6 ms | DERIVED |
| 4-vCPU parallel (fixed partition) | ~1–3 ms | DERIVED |
| Portable fp32 reference (as committed, scalar loop, `reference_cpu_executor.cpp:120-141`) | ~15–40 ms | DERIVED (measured only at H=64/I=32: 3.70 ms p50 portable-torch — NOT scalable, do not extrapolate) |
| KT AMX class (Xeon4, 21 TFLOPS BF16) | ~0.15–0.5 ms (mem-floor-bound) | PAPER-REPORTED, inapplicable to Kaggle CPU |

Kaggle host CPU: **UNKNOWN model** — T4x2 sessions give ~4 vCPU Xeon
(Skylake/Cascade-Lake class) + ~30 GB RAM (env: MemTotal 31.35 GB). ISA:
AVX2 certain; AVX512F/BW probable; AVX512-BF16 unlikely (Cooper Lake+);
AMX absent (Sapphire Rapids+ only). The KT `fp4_mat_vec_kgroup_natural`
fast path is `__AVX512BF16__`-gated → on Kaggle the AVX2 `fmadd_ps` path
is the realistic kernel. Probe at init (`_cpu_detect`-style), log the
variant, never assert — standing rule from the KT audit.

## 3. The break-even derivation

### 3.1 Per-miss inequality

```
EXECUTE_FROM_HOST wins a miss iff:

  t_d2h_act + t_cpu(m) + t_h2d_out + t_join
      <  t_stage_submit + t_h2d_rec + t_gpu + t_queue
         (+ t_fill on BOTH sides — cancels)

Caller-thread (serial) form, plugging the measured/DERIVED values:

  t_cpu(1) + ~0.1 ms  <  ~2.4 ms + (~1.2 + 0.35 ms if device-side is
                          critical-path) ≈ 2.5–4.0 ms
```

So `t_cpu*(1) ≈ 3–4 ms` — i.e. ≥ ~13–20 GFLOP/s effective on a 50.3
MFLOP expert including FP4 decode. Serial verdict: tuned kernel
borderline-passes, portable reference fails (~15–40 ms), and on a cold
miss serial-CPU adds `t_cpu − 2.4 ms` directly onto the blocking chain —
serial EXECUTE_FROM_HOST is only ever a host-hit optimization.

### 3.2 Overlapped form (the Fiddler/MoE-Lightning point)

If CPU exec runs on a worker while the caller continues to the next
`stage()` (which blocks ~34–96 ms in the next miss's pread), the caller
cost of a CPU-routed miss collapses to `host.acquire` + lease handoff
(~µs). Join condition: the worker must finish before the layer's
weighted combine — envelope ≈ remaining layer wall (~37–100 ms of
non-fill work per row exists even with zero misses: decode row wall
102.7 ms, fill 65.1 ms — MEASURED). Break-even becomes:

```
n_cpu_inflight × t_cpu(1)  ≤  inter-miss shadow (~34–96 ms/miss)
                           or  row remainder (~37 ms floor)
```

i.e. at decode essentially any `t_cpu(1) ≤ ~25–30 ms` is wall-free with
1–2 workers. **This regime, not the serial one, is what the prior art
implements and what dee should evaluate.**

### 3.3 HYBRID split — q* worked example

`cost_model.py::plan_split` (cost_model.py:44-70): `q* = argmin_q
max(T_gpu(q), T_cpu(m−q))` by enumeration; ties prefer CPU. Using
`t_h2d` = caller-paid staging ≈ 3.6 ms (2.4 submit + 1.2 device), `t_gpu`
= 0.35 ms, `t_cpu` = 4 ms, overlapped model
`T_gpu(q) = max(t_h2d + q·t_gpu, q·t_h2d + t_gpu)`:

| m=3 | q=0 | q=1 | q=2 | q=3 |
|---|---:|---:|---:|---:|
| T_gpu | 0 | 3.95 | 7.55 | 11.15 |
| T_cpu | 12.0 | 8.0 | 4.0 | 0 |
| makespan | 12.0 | 8.0 | **7.55** | 11.15 |

q*=2, makespan 7.55 ms vs all-GPU 11.15 → ~32% staging-side saving. For
m=1 the model degenerates to the single-miss comparison (CPU iff
t_cpu < t_h2d + t_gpu ≈ 3.95 ms). Caveat: this isolated-layer model omits
the fill shadow — with overlap the real q* is usually m (all misses to
CPU) until thread/DRAM budget binds, because T_gpu work sits on the
caller while T_cpu does not.

### 3.4 Cold-miss honesty check

Both paths pay `t_fill`. Serial phase2 (`stage()` on the caller) makes
CPU exec strictly worse for cold misses by `t_cpu − 2.4 ms` *unless* the
exec overlaps a subsequent fill. Note the dependency: with ~1.9
misses/row the last miss of a row has no next-fill shadow inside the row
— it hides under the row's sync/orchestration tail instead (~30–45 ms).
A worker-pool fill design (T9 pre-acquire, DESIGN.md) is currently NO-GO
*as a wall optimization* (+23% measured regression on this bank) — but
its §7(c) explicitly names "CPU-decoupling for another reason" as a
preserved GO condition. The CPU sink is exactly that reason: the same
pool that pre-acquires can absorb `execute()`; re-score T9's gate under
the sink design, not the fill-parallelism hypothesis.

### 3.5 Whole-response bound (DERIVED, upper bounds only)

| Lever | Bound | Basis |
|---|---:|---|
| stage_enqueue removal for diverted misses | ~6–9 s of 66.2 s decode | DERIVED (9.436 s bucket × miss share) |
| readiness/queue | ~0 | MEASURED (already ~0) |
| sync tail change | ~0–small | join folds into existing 8 ms/row drain |
| fill wall | **0 — invariant** | MEASURED (sink-agnostic) |
| VRAM churn relief | unpriced second-order | 60.5% records never repeat → diverted inserts avoided |
| headroom if `t_cpu(1)` ≤ 3 ms AND serial | +~2–3 ms/miss × 4,444 ≈ 9–13 s | DERIVED |

Pack-cap lesson applies (LIVE_PROFILE_RESULTS §"paradox resolved"):
predicted wins below ±2–5 s run noise are unverifiable — the ~9 s upper
bound clears noise, the per-miss marginal (~1–4 ms) does not except in
aggregate.

## 4. Contention

- CPU threads also drive fills/staging: on ~4 vCPU, pread service is
  I/O-blocked (threads sleep in the kernel), so 1–2 exec workers are
  schedulable — but the caller thread's submit/enqueue/combine work and
  the journal/orchestration bucket (9.2 s unknown) also want cores.
  UNKNOWN: measured contention coefficient. Bound it with a lease budget
  (R6 G10: `cpu_sink_lease_budget < dynamic_slots`).
- DRAM bandwidth sharing: demand fills write ~0.3–0.74 GB/s effective
  (MEASURED); a CPU exec stream reads 12.75 MiB per exec (~3–13 GB/s in
  bursts). Sum ≪ the ~20–40 GB/s a 4-vCPU Xeon sustains — likely fine,
  but FreeToken's own finding applies (PAPER-REPORTED): "PCIe transfer
  and direct CPU expert execution both draw from the same host-memory
  bandwidth" — on a fuller system the gather memcpy + exec reads + H2D
  pinned writes contend. Measure on-host; do not assume.
- Page-cache interplay: `mincore` showed 15.7–32.7% residency at read —
  cold reads really do hit the device. CPU exec on freshly-pread bytes
  reads warm page-cache/LRU-slot memory (fast path), so the sink
  *benefits* from the fill's own caching effect.
- Prefill contention is worse: bigger batches want all cores for GEMM
  while fill threads multiply — another reason decode is the favored
  regime (§3.5, Fiddler crossover).

## 5. What "exact" means at this seam (the biggest question)

**Verdict: "exact" at the execution sink is gate-equivalence to the
trusted fp32 reference plus the end-to-end seal — not bitwise identity
to the CUDA path.** Justification and contract proposal:

1. dee's arbiter is the FP32 Python reference
   (`scripts/deepseek_v4_expert_reference.py`), "deliberately NOT
   bitwise-tied to any specific CUDA reduction order" (:20-21). The CUDA
   candidate itself "carries ... error that must fall inside the
   predeclared DS7 tolerance — near-bitwise agreement is NOT expected"
   (:23-27). Bitwise-vs-CUDA is therefore not the standard anywhere in
   dee today; the standard is *the same predeclared gate stack*.
2. The existing per-expert gate (in-tree): `rel_rmse < 0.02`,
   `cosine > 0.999` vs the fp32 reference
   (`tests/test_deepseek_v4_fp4_expert.cpp:247-248`), plus layer-trace
   categories (`test_deepseek_v4_layer.py` cosine gates) and the seal:
   identical token IDs + text on the sealed 16-token generation
   (ACCEPT_CORRECTNESS, identical IDs/text — `evidence-live/*/correctness.json`).
3. Numeric classes ordered by fidelity to the reference:
   - **CPU fp32 (`ReferenceCpuExecutor`)** — reference semantics verbatim;
     only delta is summation order (~1e-7 rel). Strictly *closer* to the
     arbiter than the shipping CUDA path.
   - **CUDA FP4→FP16 path (sealed today)** — exact dequant (E2M1 × 2^e
     products are fp16-exact), fp16 GEMM with fp32 accumulate, fused
     SwiGLU fp32-math/fp16-IO (`swiglu_cuda.cu`, `cuda_convert.cu:231-245`).
   - **KT-faithful bf16 class** — additional bf16 boundary round-trips;
     synthetic-shape metrics: cosine 0.99999, mean_rel 0.019, p95_rel
     0.067 (SUMMARY.md §"viable"). p95 outside a naive 2% band → must be
     re-gated at real geometry before it may run sealed.
   So a CPU path can be *more* exact than the current sealed path; the
   only questionable class is the bf16 one.
4. **Proposed contract — a sealed CPU-path clause set:**
   - (a) Executor-class tag per expert execution: `{device-fp4-fp16,
     host-fp32, host-bf16-kt}` recorded with the run; the seal artifact
     names which classes ran.
   - (b) Each class must independently pass the per-expert gate vs the
     fp32 reference at *real geometry* (H=4096/I=2048, real checkpoint
     tensors — `DEE_REAL_EXPERT_DIR` leg), not only synthetic shapes.
   - (c) Determinism: outputs must be bit-reproducible run-to-run. The
     single-thread reference is deterministic by construction; a
     *work-stealing* pool (KT `do_work_stealing_job`) is NOT — partition
     order varies → reduction order varies → non-reproducible logits.
     Clause: sealed runs use fixed partition/fixed thread count, or
     single-thread. This is a hard requirement the KT import would
     otherwise violate silently.
   - (d) End-to-end re-seal: run the sealed 16-token decode under each
     admitted executor mix; require identical IDs/text (16/16). A policy
     change that alters which class executes an expert is a new candidate
     path → re-seal (precedent: FP16-cache and INT8-transfer each
     re-validated).
   - (e) Fail-closed: a CPU-executor error aborts the layer; skipping an
     expert is a contract violation (same shape as `!computed` failures;
     R6 Q4d). `0xFF` scale, `alpha≠0`, shape/dim mismatch already fail
     closed in the bridge (`cpu_executor.hpp`, reference
     `validate_execute_args` :69-99).
5. Validation mechanics already exist: codec identity, clamp asymmetry,
   weight-placement equivalence, determinism, metric gates are all in
   `experiments/kt_cpu_bridge/tests/` (20 tests passing on this branch);
   the dee reference gates were explicitly NOT weakened — the CPU path
   inherits them unchanged.

## 6. MissExecutionPolicy vs the sealed 16/16 contract

```
enum class MissExecutionPolicy {
    TRANSFER_TO_DEVICE,   // today
    EXECUTE_FROM_HOST,    // every device-miss executes on CPU
    HYBRID                // per-expert q* split / state-driven
};
```

- **Legality:** the decision is post-route and per-expert — routing,
  identity, ordering, and the combine are untouched, so it sits inside
  the exactness envelope ("may change where data lives … how scheduled").
  It is *not* a prediction: no expert is chosen or skipped by a model.
  Execution sink is a placement decision like the tier residency choices
  dee already makes.
- **Coexistence with the seal:** the sealed contract binds *outputs*
  (identical IDs/text) plus the gate stack — not *where* bytes computed.
  A HYBRID policy is admissible iff its per-expert choice is a
  deterministic function of journal-visible state (residency snapshot,
  m, tier occupancy) OR the choice is appended to the route journal, so
  any sealed run is replayable bit-for-bit. Timing-dependent choices
  (e.g. "CPU if a worker is free right now") break reproducibility —
  disallow for sealed runs, allow for non-sealed perf evaluation.
- **Composability notes:** TRANSFER stays mandatory as fallback (no CPU
  executor → fail-closed to GPU); HYBRID + `PolicyResident` partitions
  are orthogonal (regime-C contract unaffected — CPU diversion never
  touches admission semantics); a diverted expert should probably still
  be H2D-admitted lazily *only if* a future repeat is expected — 60.5%
  of sealed records never repeat, so "divert AND skip VRAM insert" is
  the better default on this trace (state-based, not predicted).
- **Seam requirements** (priced by R6, not re-derived here): G1 sink-aware
  `stage_ex`, G2 executor interface promotion, G3 region descriptor,
  G4 submit/join for the overlap regime, G5 consume-loop sink tags,
  G8 `NoFill` probe for "CPU only if already in RAM" policies, G9 metrics
  dimension, G10 lease budget. Cheapest first cell: synchronous
  EXECUTE_FROM_HOST on the host-API paths (no G4/G6) — exactness +
  miss-latency evidence without a new GPU batch.

## 7. Prior-art mapping

| System | Mechanism | Reported result | dee applicability |
|---|---|---|---|
| **Fiddler** (ICLR'25, arXiv 2402.07033) | Per-layer CPU-vs-GPU decision from a latency model: CPU exec ≈ linear in m, GPU ≈ flat + weight-transfer overhead → CPU for small batch. AVX512-BF16 CPU kernel. | PAPER-REPORTED: >3 tok/s Mixtral-8x7B on 24 GB GPU; 1.26× single-batch, 1.30× long-prefill, 11.57× beam vs SOTA; 8.2×/10.1× vs offloading on Quadro RTX 6000 / L4 | The exact seam of this track. dee difference: dee's misses also pay a *storage* fill Fiddler doesn't have (its weights were host-resident); the decision must be per-miss and overlap-aware, not per-layer |
| **KTransformers** (pinned 31985f4) | AMX/AVX512/AVX2 MXFP4 CPU kernels; `submit_with_cuda_stream` overlap; dynamic AMX↔AVX512 switch at avg >4 tokens/expert | PAPER-REPORTED: 21 TFLOPS BF16 / ~35 TOPS INT8 MoE kernel on Xeon4; DSv3 prefill 418 tok/s (Xeon4+4090); R1 decode ~13.7 tok/s | Kernel source of truth — dee's packed bytes are byte-identical to its input (FORMAT_COMPATIBILITY proven). Import the kernel *structure*, not the pool/mask/SGLang coupling (do-not-import list stands). AVX2 path is the Kaggle-realistic one |
| **FreeToken** (arXiv 2608.16157) | Bandwidth-adaptive q* split; global LRU expert cache; double-buffered prefill streaming; same model family (DSv4-Flash listed, 284B on gaming desktop) | PAPER-REPORTED: 1.5–2.3× decode vs baselines; PCIe transfer and CPU exec share host-mem bandwidth (their stated reason for decode-time finer allocation) | Closest analog to `plan_split`; confirms the host-BW contention point (§4) and that q* must be runtime-adaptive |
| **HybriMoE** (DAC'25, arXiv 2504.05897) | Dynamic intra-layer CPU/GPU scheduling on KT + impact-driven prefetch + score-based cache | PAPER-REPORTED: 1.33× prefill / 1.70× decode vs SOTA hybrid | Evidence that *intra-layer* splitting (HYBRID) beats static placement under unstable activation patterns — supports per-miss over per-layer policy granularity |
| **MoE-Lightning** (ASPLOS'25, arXiv 2411.11217) | CGOPipe overlaps CPU compute + I/O + GPU compute; HRM hierarchical-roofline picks placement | PAPER-REPORTED: ≤10.3× throughput vs offloading systems for Mixtral-8x7B on a single T4 | The overlap theorem behind §3.2 — but throughput-regime (batch) work; dee's batch-1 decode overlap is the caller-blocked fill shadow, not a pipeline stage |

## 8. Falsification & measurement plan (no GPU spend required)

What would kill or shrink the design:

1. `t_cpu(1)` on the Kaggle CPU > ~30 ms even with a tuned AVX2 kernel →
   overlap break-even fails on low-miss rows (last-miss tail ~30–45 ms);
   CPU sink survives only as a host-hit fast path.
   *Measure:* build `kt_cpu_bridge` standalone on Kaggle (CPU ledger),
   `bench_cpu_expert.py --hidden 4096 --inter 2048` + a C++ AVX2 port;
   report p50/p95 at real geometry with real expert bytes
   (`DEE_REAL_EXPERT_DIR`).
2. CPU exec measurably slows the fill path (DRAM/page-cache contention
   > ~10% service regression) → overlap is not free. *Measure:* exec
   loop concurrent with `tools/fill_replay` on the bank.
3. bf16-kt class fails the per-expert gate at real geometry (p95_rel
   already 0.067 at toy shape) → only the fp32 class is admissible;
   if fp32 is also too slow, the whole track degrades to serial-host-hit
   only.
4. Nondeterminism under threading (work-stealing reduction order) →
   sealed contract forces fixed partition or single-thread, capping
   `t_cpu` improvement; measure variance across runs.
5. The ~6–9 s prize is real but the *fill* is 42 s — if GPU Batch #1's
   reprofile shows the hierarchy fixed (residency working), the miss
   stream itself shrinks and the absolute prize shrinks with it. This is
   a Phase-2.x/Phase-5 lever, not a Phase-2 blocker.

Non-goals reaffirmed: no implementation in Phase 2; no remote spend; KT
pool/global-mask/SGLang coupling not imported; AMX/AVX512-BF16 never
asserted.

## 9. Roadmap placement

1. Lands after Phase-2 GPU Batch #1 (needs the live tier machinery
   proven) and pairs with Phase-3's arbitrary universe — on an
   11,776-record bank, misses dominate and a second sink pays most.
   Explicitly listed in AGENTS.md dee-serve direction ("CPU/GPU hybrid
   miss execution").
2. Cheapest first motion (Phase-2.x mechanism cell, host-side only):
   synchronous EXECUTE_FROM_HOST on `moe_forward_experts` /
   `moe_forward_batch` — needs only R6's G1–G3, zero new sync machinery,
   produces the exactness gate data (per-expert fp32-class check +
   re-seal) and the serial `t_cpu` measurement.
3. Overlap regime (the actual prize) needs G4 submit/join + a
   deterministic-order per-engine pool — and revives the T9 pre-acquire
   pool under its §7(c) "CPU-decoupling" clause; re-score that gate with
   the sink included rather than as a fill-parallelism repair.
4. Wire `cost_model.py::plan_split` with MEASURED `t_cpu/t_h2d/t_gpu`
   from step 2; HYBRID stays journaled-deterministic for sealed runs.

## Appendix — constants used

| Symbol | Value | Source |
|---|---:|---|
| R (record) | 13,369,344 B (12.75 MiB) | MEASURED (DEE4) |
| F (per expert-token) | 50.33 MFLOP (2 × 25,165,824) | DERIVED |
| act round-trip | 32 KiB fp32 (4096 × 4 × 2) | DERIVED |
| payload ratio | 407.9 : 1 | DERIVED |
| misses/response | 4,444 device; 2,481 cold | MEASURED |
| t_fill | 18.9–96 ms/miss (device/regime dependent) | MEASURED |
| t_stage_submit | ~1.85 ms avg, ~2.4 ms/miss | DERIVED |
| t_h2d_rec | ~1.19 ms @ ~11.2 GB/s | DERIVED |
| t_gpu(1) | ~0.3–0.4 ms | DERIVED |
| t_d2h + t_h2d_out | ~0.03–0.1 ms | DERIVED |
| t_cpu(1) | UNKNOWN; 2–6 ms tuned / 15–40 ms portable (DERIVED bounds) | UNKNOWN |
| fill shadow | 34–96 ms/miss; 65.1 ms/row; ~37 ms row floor | MEASURED/DERIVED |
