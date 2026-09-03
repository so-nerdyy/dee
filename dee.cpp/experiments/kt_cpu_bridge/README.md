# kt_cpu_bridge — isolated KTransformers CPU-expert bridge prototype

Parallel research track (`research/kt-cpu-bridge`). NOT production integration.
NOT in the root build. Campaign benchmark paths untouched.

Docs (authoritative, on this branch):

- `research/ktransformers/KT_CPU_AUDIT.md` — full source audit (Phase A)
- `research/kt-cpu-bridge/FORMAT_COMPATIBILITY.md` — MXFP4 proof (Phase B)
- `research/kt-cpu-bridge/CPU_EXECUTOR_DESIGN.md` — adapter design (Phase C)
- `research/kt-cpu-bridge/THIRD_PARTY_KTRANSFORMERS.md` — license/attribution
- `research/kt-cpu-bridge/SUMMARY.md` — final summary + blocker list

Build (standalone):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Test (python, from `dee.cpp/`):

```
python -m pytest experiments/kt_cpu_bridge/tests/test_kt_bridge_codec.py -q
python -m pytest experiments/kt_cpu_bridge/tests/test_kt_bridge_correctness.py -q
python -m pytest experiments/kt_cpu_bridge/tests/test_kt_bridge_cost_model.py -q
python -m pytest experiments/kt_cpu_bridge/tests/test_kt_bridge_executor_cpp.py -q
```

Bench:

```
python experiments/kt_cpu_bridge/bench/bench_cpu_expert.py --hidden 64 --inter 32 --repeats 50
```

Rules: bounded per-expert records only; no KT types past the adapter; no
threads/NUMA in Phase 1; `0xFF` scale fails closed; `swiglu_alpha != 0` fails
closed; no full-model TPS claim.

## Offline real-weight correctness leg

Build `kt_bridge_reference_probe` with the standalone CMake project. It links
only `ReferenceCpuExecutor`, not the KT emulator, CPUInfer, or a worker pool.
Run its `--self-test` for shape-overflow and NaN-clamp rejection.

```
python bench/verify_real_expert.py --shard LOCAL_PARTIAL_SAFETENSORS --bundle SEALED_BUNDLE_DIR --seal TERMINAL_SEAL_JSON --executor REFERENCE_PROBE_EXE --out REPORT_JSON
```

The bundle must contain `dee4-metadata.json`, `dee4-integrity.jsonl`, and
`routed_experts.jsonl`; the seal must list their SHA-256s under `raw_sha256`.
The CLI defaults to layer 0, expert 155; both are selectable. It verifies six
official tensor hashes and the reconstructed canonical DEE4 record hash before
execution. It never downloads or stages any weights. The local inputs and
output report are not added to the production build or campaign.

The journal supplies expert selection only. Full activation rows and routing
weights are not in that journal, so this leg clearly labels both as synthetic
probes. It is **not** native KT execution, a sealed-activation replay, or model
parity. The existing dee DS8 gates are reported unchanged; a separate tighter
FP32 allclose check is also reported. Neither gate is relaxed after results.
An optional pytest integration reads `real-expert-config.json` inside
`DEE_REAL_EXPERT_DIR`, with absolute `shard`, `bundle`, `seal`, and `executor`
paths and optional `layer`/`expert`. Without it, pytest reports a real skip.

Safe integration remains a synchronous borrow for one expert: dee pins its
host-cache entry through `execute`, supplies routing weight/configuration,
and combines output. No cache admission, eviction, routing, prediction,
scheduling, or SSD ownership crosses this interface. Async integration is
not authorized by this probe; the pinned WorkerPool lifecycle and exception
hazards require a separate design and concurrency tests.
