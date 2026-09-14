# KT CPU Bridge — final summary (research/kt-cpu-bridge)

Upstream pin: ktransformers `31985f40bcc40da08107efdb1f81bf88cb38c6b2` (2026-09-01), Apache-2.0.
Branch: `research/kt-cpu-bridge`. Untouched: `freebuff/deepseek-v4-flash-0731-t4`
+ all campaign kernels/configs/evidence. No merge. No full-model TPS claim.

## Deliverables (all on this branch)

1. `research/ktransformers/KT_CPU_AUDIT.md` (canonical) + copy at
   `research/kt-cpu-bridge/KT_CPU_AUDIT.md` — Phase A audit (15 areas).
2. `research/kt-cpu-bridge/FORMAT_COMPATIBILITY.md` — Phase B proof.
3. `research/kt-cpu-bridge/CPU_EXECUTOR_DESIGN.md` — Phase C design.
4. `dee.cpp/experiments/kt_cpu_bridge/` — isolated adapter prototype
   (C++ `ReferenceCpuExecutor` + `KTransformersCpuExecutor`, Python
   `codec/reference/cost_model`, bench driver; standalone CMake, NOT in root build).
5. `dee.cpp/experiments/kt_cpu_bridge/tests/` — codec, correctness,
   cost-model, C++ smoke tests (20 passed here; existing dee v4 gates still pass).
6. `python/kt_cpu_bridge/cost_model.py` + `bench/bench_cpu_expert.py` — Phase E
   microbench API + `q*` enumeration simulator (no live-scheduling change).
7. `research/kt-cpu-bridge/THIRD_PARTY_KTRANSFORMERS.md` — license/attribution
   (zero verbatim copies; semantic regions itemized).
8. This file — final summary + blocker list.

## What can be reused directly

- MXFP4 weight bytes (E2M1, low-nibble-first, block-32, row-major `[out,in]`,
  `w1=gate/w3=up/w2=down`): byte-identical, zero-copy `memcpy`.
- E8M0 law (`2^(e-127)`) + `(u8<<7).view(bf16)` loader step: value-exact for
  real checkpoints.
- Asymmetric SwiGLU clamp (`gate=max-only`, `up=±`, `limit=10.0`): already
  matches dee CUDA + Python reference — no change needed.
- Kernel structure (LUT decode → dpbf16 → fmadd(scale) → 4-wide reduce → 4×4
  prefill tile; natural-order permute; `expand_e8_scales` hoist; `fast_memcpy`)
  as a porting template (design only in Phase 1).
- `submit_with_cuda_stream` + `sync_with_cuda_stream` overlap pattern
  (interface reserved, not bound).

## What must be adapted

- Loader: full-pool walk → per-expert bounded fill from `HostPackCache`
  (weights zero-copy; scales via lossless `ue8m0→bf16→fp32→compact-e8`,
  ~1-2 ms convert + ~1-2 ms memcpy per expert, parallelizable).
- Lifetimes: borrowed global pointers → per-call `PackedExpertView` borrows +
  adapter-owned flat blob (`required_size = n*k/2 + n*k/gs`, memcpy-evictable;
  cache compacted `e8` to skip torch conversion on hit).
- Threading: KT global singleton pool → caller-thread sync now, per-engine
  pool later (never share KT's singleton; 2-slot layer ring assumption removed).
- Placement: static top-K mask + full CPU backing → dee bounded
  admission/eviction + `q* = argmin max(T_gpu(q), T_cpu(m-q))` enumeration
  (simulator ready, live scheduler untouched).
- Router: SGLang-side topk/weights → dee `route_topk` + scalar
  `routing_weight` per call (adapter never renormalizes).
- ISA: AMX/AVX512-BF16 fast path → portable + AVX2-verified baseline first;
  AMX port later behind the same interface (probe at init, never assert).

## Is on-demand expert execution viable? YES

`SSD packed expert → bounded RAM entry → optional scale re-encode →
CPUInfer-shape compute` works with steady-state residency EQUAL to source size
(~12.6 MB/expert at H=4096/I=2048). Measured here (portable torch,
H=64/I=32): ~3.5 ms/expert ref/emulated; H2D-memcpy proxy ~1.5 µs at that
shape (not a transfer claim — measure `t_h2d/t_gpu` on the campaign host with
dee's own T4 kernels). Correctness: KT-emulated vs fp32 on synthetic fixtures
gives `cosine 0.99999, mean_rel ~0.019, p95_rel ~0.067` (deterministic,
finite); C++ MSVC smoke passes (`ref=851.986 kt=856.000` on the tiny case,
delta = bf16-boundary rounding, as designed).

## Does KTransformers require too much RAM? UPSTREAM YES / BRIDGE NO

- Upstream KT: YES — full-pool C++ NUMA copies + `chunked_prefill_size`-sized
  scratch + pinned per-B buffers, no eviction (fail-loud on oversize `qlen`).
  Importing `TP_MOE`/loader wholesale would violate dee's bounded-RAM design.
- This bridge: NO — per-expert flat blob + caller-owned I/O, nothing retained
  across calls in Phase 1; converted bytes live ONLY inside dee's bounded host
  cache. Full-model second copy explicitly NOT required and NOT built.

## Exact blocker list

1. ~~Weight layout mismatch~~ — CLEARED (byte-identical, proven §2-§5 of
   FORMAT_COMPATIBILITY.md).
2. Scale re-encode required — NOT blocking (lossless, per-expert, cacheable;
   implemented + tested). Hard rule: fail closed on `0xFF`; never pass raw
   `u8*` where KT expects `bf16*` (wrong-kernel ABI = silent corruption).
3. `group_size != 32` (e.g. NVFP4 group-16) — OUT OF SCOPE, fail closed
   (AMX rejects; adapter rejects).
4. `cpu_backend/` headers not in sparse checkout — `WorkerPool`
   work-stealing/exception semantics UNVERIFIED. No production pool binding
   until full-tree audit of that directory. (Phase-1 adapter owns no threads,
   so not blocking for prototype.)
5. `e=0/255` edge encodings diverge (KT `+inf` vs dee clamp) — absent from
   real checkpoints; adapter fails closed. Not blocking.
6. T4 (`SM_75`) outside KT's validated GPU matrix (`SM_86/89/120`, CUDA≥12.8,
   flashinfer≥0.6.9) — IRRELEVANT to CPU execution (CUDA-arch-independent by
   construction) but GPU-side numbers must come from dee's own T4 kernels.
   No KT Triton path on the campaign host.
7. Singleton pool / global mask / SGLang router coupling / GGUF-or-prequant
   assumptions — DO NOT IMPORT (documented conflicts; adapter avoids all).
8. Real-expert tensors not present on this host — component proof uses
   synthetic fixtures + dee trusted-reference bitwise check
   (`test_matches_dee_trusted_reference_when_available` passes where the
   reference imports); set `DEE_REAL_EXPERT_DIR` on a fixture host to run the
   real-tensor leg. No synthetic weight is ever presented as real.

## For Codex (cherry-pick order)

1. `kt_bridge/{packed_expert_view,cpu_executor}.hpp` (stable interface).
2. `reference_cpu_executor.{hpp,cpp}` (arbiter — keep).
3. `kt_cpu_executor.{hpp,cpp}` scale-compaction + bf16-boundary order (port
   inner dots to AMX when ready; keep `0xFF`/`alpha` guards).
4. `cost_model.py::plan_split` (wire with MEASURED `t_cpu/t_h2d/t_gpu`).
5. Bench driver JSON → campaign host measurements.
6. Do NOT pick up: singleton pool, global mask, SGLang coupling, full-pool
   loader, GGUF path, any `freebuff/*` adjacency.

## Verification log (this host)

- `pytest dee.cpp/experiments/kt_cpu_bridge/tests/`: **20 passed**.
- `pytest dee.cpp/tests/test_deepseek_v4_expert_reference.py dee.cpp/tests/test_deepseek_v4_moe_reference.py`: **14 passed** (existing gates unweakened).
- `cmake --build dee.cpp/experiments/kt_cpu_bridge/build`: **OK** (MSVC `kt_bridge.lib` + `kt_bridge_smoke.exe`); smoke prints `smoke OK`.
- `bench_cpu_expert.py --hidden 64 --inter 32 --repeats 20`: ref p50 ~3.70 ms,
  KT-emulated p50 ~3.75 ms (portable-torch, single row — NOT a campaign claim).
- `git status` scope: only `research/ktransformers/`, `research/kt-cpu-bridge/`,
  `dee.cpp/experiments/kt_cpu_bridge/` added. `freebuff/*`, campaign configs,
  evidence ledger untouched (verify in review diff before push).
