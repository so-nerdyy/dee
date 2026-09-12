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
