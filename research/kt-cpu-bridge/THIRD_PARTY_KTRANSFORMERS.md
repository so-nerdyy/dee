# Third-party: KTransformers

Upstream: https://github.com/kvcache-ai/ktransformers
Pinned commit audited: `31985f40bcc40da08107efdb1f81bf88cb38c6b2` (2026-09-01).
Upstream license: **Apache License 2.0** (repo-root `LICENSE`).

This branch (dee `research/kt-cpu-bridge`) preserves all legally required
notices and attribution for reused concepts. Preference was given to
linking/integration over copying: **no large upstream sections are vendored
here.** All prototype code under `dee.cpp/experiments/kt_cpu_bridge/` is
original to dee (interface + portable reimplementation of publicly documented
math), with the upstream-derived semantic details itemized below so a future
porter credits the right regions.

## Copy status

- Files copied verbatim from KTransformers: NONE.
- Files adapted (line-for-line port): NONE.
- Regions reimplemented from documented semantics (values verified against
  dee's own trusted references, not pasted): listed in §2.
- Regions referenced (read for design, not reproduced): listed in §3.

If a future change vendors any upstream region, append it to §4 with the exact
commit, file range, and modification note BEFORE committing.

## 1. License text (upstream)

KTransformers root `LICENSE` is Apache-2.0 (January 2004). Full text lives
upstream; the operative obligations for this track:

- Retain copyright + license notices on any redistributed derived work.
- State significant changes when modifying vendored files.
- `NOTICE` file attribution if one exists upstream (none observed at pin).
- This file + per-file header notes satisfy the attribution practice; they
  are not legal advice.

Python files in `kt-kernel/python/` carry `SPDX-License-Identifier: Apache-2.0`.
`kt-kernel/ext_bindings.cpp:1-9` and `kt-kernel/operators/amx/fp4-moe.hpp:1-14`
carry `Copyright (c) 2024 by KVCache.AI, All Rights Reserved` headers — any
future vendoring of those files MUST keep those headers intact.
`kt-kernel/operators/common.hpp` carries no header; root `LICENSE` still applies.

## 2. Semantic regions reimplemented (values, not code)

| # | Semantics | Upstream source (exact) | dee counterpart (this branch) | Notes |
|---|---|---|---|---|
| R1 | E2M1 codepoints `{0,±0.5,±1,±1.5,±2,±3,±4,±6}` + `-0.0` quirk, low-nibble-first packing | `kt-kernel/operators/amx/fp4-moe.hpp:49-98` (`fp4_bf16_lo/hi`, `mxfp4_to_bf16_32`); `kt-kernel/operators/avx2/mxfp4-moe.hpp:53-83` | `experiments/kt_cpu_bridge/{include/kt_bridge/cpu_executor.hpp (table decl), src/*_executor.cpp (kFp4Table*), python/kt_cpu_bridge/codec.py (FP4_TABLE, unpack_fp4)}` | Values cross-checked against dee's own `FP4_TABLE` (`scripts/deepseek_v4_expert_reference.py:35-41`, `src/weight_mmap.cpp:76-82`) — identical. Table is factual quantization data, restated, not pasted. |
| R2 | E8M0 law `value = 2^(bits-127)`; loader bit-cast `(u8<<7).view(bf16)`; `finalize_scale_e8` positive-pow2 validation + in-place compact; `expand (e<<23)` | `kt-kernel/python/utils/loader.py:1222-1231`; `kt-kernel/operators/amx/fp4-moe.hpp:243-303,390-401` | `cpu_executor.hpp (kt_bridge_e8m0_to_f32)`, `kt_cpu_executor.{hpp,cpp} (compact/expand_scales)`, `codec.py (decode_e8m0, compact_scales_e8, expand_e8_scales, ue8m0_to_bf16_bits)` | Verified against dee `e8m0_to_f32` (`src/weight_mmap.cpp:88-99`) + torch `float8_e8m0fnu` round-trips. |
| R3 | Asymmetric SwiGLU clamp (`gate=min(g,limit)`, `up=clamp(u,±limit)`, `silu(g)*u`; `alpha>0` MiniMax variant rejected) | `kt-kernel/operators/amx/la/amx.hpp:47-101`; `kt-kernel/operators/avx2/avx2_bf16_utils.hpp:118-169`; `kt-kernel/operators/common.hpp:311-320` | `src/*_executor.cpp`, `python/kt_cpu_bridge/reference.py` | Already identical to dee `src/swiglu_cuda.cu:22-24` + `scripts/deepseek_v4_expert_reference.py:124-126`. |
| R4 | Late routing-weight placement (`x += w*down_out` in fp32, skipped experts = 0, no renormalization) + `merge_results` fp32→bf16 | `kt-kernel/operators/amx/moe_base.hpp:413-436,620-638,760-804` | Documented + proven equivalent in `tests/test_kt_bridge_correctness.py::test_routing_weight_placement_equivalence`; adapter implements dee placement (before-`w2`) | Algebra identity `(w·h)W^T = w·(hW^T)`; no code copied. |
| R5 | Kernel structure (decode-via-LUT → dpbf16 → fmadd(scale) → 4-wide reduce → 4×4 prefill tile; natural-order permute + `expand_e8_scales` hoist + `fast_memcpy`) | `kt-kernel/operators/amx/fp4-moe.hpp:165-173,350-700,771-799,873+` | Design notes only (`CPU_EXECUTOR_DESIGN.md` §4/§9, `KT_CPU_AUDIT.md` §5/§13); Phase-1 executors are portable scalar fp32, NOT a port | Future AMX porter: cite this row + keep upstream copyright headers on any pasted hunks. |

## 3. Regions referenced (design only, nothing reproduced)

- CPUInfer lifecycle / async: `kt-kernel/ext_bindings.cpp:467-539,641-652`; `kt-kernel/python/experts_base.py:216-343,542-589,600-879`.
- MOEConfig: `kt-kernel/operators/common.hpp:230-336`; `kt-kernel/ext_bindings.cpp:833-918`.
- MXFP4 loading: `kt-kernel/python/utils/loader.py` V4 branch; `kt-kernel/python/utils/amx.py` MXFP4 dispatch; `kt-kernel/examples/test_fp4_moe_v4.py`; `kt-kernel/bench/bench_fp4_moe.py` (via PR #1970 description).
- Execution/merge: `kt-kernel/operators/moe-tp.hpp:136-258`; `kt-kernel/operators/amx/moe_base.hpp`; `kt-kernel/operators/avx2/moe_base.hpp`; `kt-kernel/operators/llamafile/moe.hpp`; `kt-kernel/operators/moe_kernel/moe.hpp`.
- Placement: `kt-kernel/operators/common.hpp:241-258`; `kt-kernel/python/experts_base.py:162-213,427-441`; `kt-kernel/python/utils/amx.py:1008-1081`; `kt-kernel/fp8_layerwise_transport.{hpp,cpp}`.
- Portability: `kt-kernel/python/_cpu_detect.py`; `kt-kernel/CMakeLists.txt:281-350`; `doc/en/DeepSeek-V4-Flash.md:21-36,152-180`; PR #1970; release v0.6.2 notes.

## 4. Vendored-code log (append-only; currently empty)

| Date | Upstream commit | Upstream file:lines | Local file:lines | Change summary |
|---|---|---|---|---|
| — | — | — | — | No vendored hunks in this prototype. |

## 5. Attribution

KTransformers is developed by MADSys Lab @ Tsinghua University, Approaching.AI,
9#AISoft, and community contributors (see upstream `MAINTAINERS.md`), under
Apache-2.0. The MXFP4 MoE operator audited here credits `oql, Codex and Claude`
(per `fp4-moe.hpp` header, 2026-04-20) and PR #1970 (`yyj6666667`, merged
2026-05-03). dee thanks those authors; this track reuses their published
semantics with attribution and without misrepresenting provenance.
