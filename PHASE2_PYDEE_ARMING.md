# Phase 2 arming via pydee (W1-T3)

Status: implemented and locally verified on the CPU (DEE_CUDA=OFF) build.
Phase 2 remains default-OFF; nothing here arms it implicitly. This document
is the Python-side contract for the switches defined in
`include/dee/expert_tiers.h` / `host_expert_tier.h` and validated in
`src/engine.cpp` (`Engine::init`, ~lines 3424-3455). Metric semantics live in
PHASE2_METRICS.md; the matched-run protocol lives in PHASE2_AB_RUNBOOK.md.

## Python config surface

```python
import pydee
cfg = pydee.EngineConfig()
cfg.phase2.enabled = True                    # master switch
cfg.phase2.host_enabled = True               # host-tier arm
cfg.phase2.vram_priority_fix_enabled = True  # actual-recency VRAM arm
cfg.phase2.model_identity = "deepseek-v4-flash-0731@<pinned-rev>"
cfg.phase2.host.slot_bytes = 0               # 0 => packed expert record size
cfg.phase2.host.alignment = 4096             # power of two, >= sizeof(void*)
cfg.phase2.host.policy_slots = 0             # must stay 0 (plain LRU policy)
cfg.phase2.host.dynamic_slots = 640          # count must fit the budget
cfg.phase2.host.budget_bytes = 8556380160    # hard cap incl. slot padding
cfg.phase2.host.try_pin = True               # attempt slot registration
```

`host_policy` / `device_policy` are deliberately NOT exposed: a null pointer
makes the engine install `PlainLruHostPlacementPolicy`, which is the accepted
Phase-2 policy. `Engine::init` copies the config, so every field must be set
before `pydee.new_engine(cfg)` / `engine.init(cfg)`.

## Metrics surface

`engine.phase2_metrics(completed_tokens=0) -> dict` returns every
`TierMetrics`/`HostTierStats` field documented in PHASE2_METRICS.md: host
counters under the `host` sub-dict (without the `host.` prefix), device
counters and wait-time boundaries at top level, `tokens` /
`bytes_per_token_valid` / `*_bytes_per_token`, and `device_gpu_wait_ms`
encoded as `None` when unmeasured (UNKNOWN, never zero). A disabled engine
returns an all-zero, invalid-denominator snapshot; the VRAM-only arm reports
`device_bytes`/`device_peak_bytes`/`device_budget`/`tokens` with an empty
host section. `engine.runtime_config()["phase2"]` echoes the effective armed
switches and host geometry for evidence binding.

## Harness plumbing

`build_native_engine(...)` in `scripts/deepseek_v4_model.py` accepts:

| kwarg | run_config.json key | env var |
|---|---|---|
| `phase2_mode` (`"off"`/`"vram"`/`"host"`/`"both"`) | `phase2_mode` | `NATIVE_PHASE2` |
| `phase2_host_budget_bytes` | `phase2_host_budget_bytes` | `NATIVE_PHASE2_HOST_BYTES` |
| `phase2_host_dynamic_slots` | `phase2_host_dynamic_slots` | `NATIVE_PHASE2_HOST_SLOTS` |
| `phase2_host_policy_slots` | `phase2_host_policy_slots` | `NATIVE_PHASE2_HOST_POLICY_SLOTS` |
| `phase2_host_slot_bytes` | `phase2_host_slot_bytes` | `NATIVE_PHASE2_HOST_SLOT_BYTES` |
| `phase2_host_alignment` | `phase2_host_alignment` | `NATIVE_PHASE2_HOST_ALIGNMENT` |
| `phase2_host_try_pin` | `phase2_host_try_pin` | `NATIVE_PHASE2_TRY_PIN` |
| `phase2_model_identity` | `phase2_model_identity` | `NATIVE_PHASE2_MODEL_IDENTITY` |

Env vars override run_config.json keys, matching every existing knob.
Host budgets/slots are per-engine; each of the two dual-GPU engines owns its
own slot arena. When a host arm is selected without an explicit identity the
harness pins `deepseek-v4-flash-0731@<REV>` (the same immutable revision the
integrity evidence already seals). The host arm requires `cache_dtype=fp4`;
anything else fails before engine construction.

## Evidence fields

Every `generated_checkpoint.jsonl` record gains `phase2` — a per-engine
`phase2_metrics(step+1)` snapshot with the completed-token denominator. The
final `result.json` gains a `phase2` block:

```json
"phase2": {
  "requested_mode": "both",
  "model_identity": "deepseek-v4-flash-0731@9e165c30...",
  "metrics": {"cuda0": {...TierMetrics...}, "cuda1": {...}},
  "config":  {"cuda0": {...effective switches + geometry...}, "cuda1": {...}}
}
```

`profile.json` carries the same block. `run_config.json` evidence records all
resolved phase2 keys next to the existing knobs.

## Arming the four A/B cells (Kaggle)

Each cell is one `run_config.json` change (or the env equivalents) on the
same pinned commit; DEE4 trace store + `cache_dtype=fp4` stay constant. The
host-tier numbers below are a starting geometry: 640 slots x 12.75 MiB
records ≈ 8.0 GiB per engine; keep `phase2_host_budget_bytes` =
`dynamic_slots` x `slot_bytes` (13,369,344 after 4096 alignment) so no slot
is implicitly denied. Budget is per engine — the 17 GiB host-pack cap math
in the harness does not include it, so pick a value that fits measured free
RAM alongside `host_pack_cache_bytes`.

Baseline (`phase2_mode` omitted or `"off"`):

```json
{"phase2_mode": "off"}
```

VRAM-only (actual-recency eviction, no host tier):

```json
{"phase2_mode": "vram"}
```

Host-only:

```json
{"phase2_mode": "host",
 "phase2_host_budget_bytes": 8556380160,
 "phase2_host_dynamic_slots": 640,
 "phase2_host_try_pin": true}
```

Combined:

```json
{"phase2_mode": "both",
 "phase2_host_budget_bytes": 8556380160,
 "phase2_host_dynamic_slots": 640,
 "phase2_host_try_pin": true}
```

Direct pydee cell equivalent (any engine, e.g. inside a notebook):

```python
cfg.phase2.enabled = True
cfg.phase2.vram_priority_fix_enabled = True   # vram / both
cfg.phase2.host_enabled = True                # host / both
cfg.phase2.model_identity = "deepseek-v4-flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
cfg.phase2.host.dynamic_slots = 640
cfg.phase2.host.budget_bytes = 640 * 13369344
engine = pydee.new_engine(cfg)                # raises on invalid combos
```

## Fail-closed matrix (verified)

`Engine::init` returns false / `pydee.new_engine` raises for: sub-switch on
with master off (`src/engine.cpp` ~3426), master on with neither arm
(~3431), host arm without packed-FP4 + CUDA + Fp4E2m1 transfer +
`model_identity` (~3435), zero/oversized host geometry or slots exceeding
the budget (`HostExpertTier` ctor, `src/host_expert_tier.cpp` ~160), and
`policy_slots != 0` under the default plain-LRU policy (~3703). The CPU-only
build rejects every host arm at the FP4/CUDA gate; the VRAM-only arm is
backend-neutral and initializes on the host-mock path.

## Local verification recipe (this machine)

```bash
# dee_core (llvm-mingw; the MSYS2 mingw64 gcc is broken here)
cd dee.cpp
CC=/c/Users/carth/scoop/apps/mingw-mstorsjo-llvm-msvcrt/current/bin/gcc.exe \
CXX=/c/Users/carth/scoop/apps/mingw-mstorsjo-llvm-msvcrt/current/bin/g++.exe \
cmake -S . -B build -G "MinGW Makefiles" -DDEE_CUDA=OFF \
  -DZLIB_LIBRARY=<shim>/libz.a -DZLIB_INCLUDE_DIR=<shim>/include
cmake --build build --target dee_core -j8
# pydee (MinGW driver; setup.py strips pybind11's MSVC-only flags)
cp <shim>/libz.a <shim>/libstdc++.a build/   # satisfy -lz -lstdc++
python pydee/setup.py build_ext --inplace --compiler=mingw32
# libc++.dll/libunwind.dll must sit beside the .pyd or on PATH
python -m pytest tests/test_phase2_binding.py -v   # 7 passed
```

On Kaggle/Linux the canonical path is unchanged: `cmake --build build` then
`python pydee/setup.py build_ext --inplace` (no `--compiler` flag).
