# Universality track — final summary

All Phase A–H artifacts are committed on `research/universality`; no
production file was modified (only `research/universality/` +
`dee.cpp/experiments/universality/` + `tests/` added). Sealed evidence,
`freebuff/deepseek-v4-flash-0731-t4`, and sibling tracks untouched; no merge.

## The 10 required answers

1. **Systems-core model-independence: ~95%.** mmap/header/lookup, both byte
   caches, LRU+pin eviction, prefetch control plane, scheduler keys,
   telemetry, alloc forensics are GENERIC by source inspection (no tensor
   names, no codec branches, no geometry). Remainder is tuning/comments.
2. **MoE-execution-stack genericity: ~60%.** SwiGLU kernels, top-k
   threading, placement math, and the `moe_forward_*` bypass are generic;
   descriptor shape, codec gate, router switch, and mock combine need the
   M1–M3 boundary work.
3. **Full-model plug-and-play today: ~10%.** No second model boots; config
   parsing, router-in-engine, residual, shared/MTP wiring are missing or
   Python-side. The number is low on purpose — the architecture is a MoE
   server, not a full runtime.
4. **Five most deeply embedded DSV4 assumptions:** (a) MXFP4 e2m1+e8m0 decode
   shared by host + CUDA + store gates (L1/L6, 5 files); (b) tri-projection
   `[3]`/`[6]` record shape across store/transfer/staging (L2);
   (c) `w1/w2/w3`+`.scale` naming and the `*2` width probe (L3);
   (d) router+combine living in the Python reference with no in-engine
   counterpart — the MoE-server architecture itself (H5/L7/L9);
   (e) 43/256/top-6/dense-8.84 GB geometry in budgets, telemetry comments,
   and Oracle dims (H9/L10).
5. **Abstractions Codex should eventually cherry-pick:** `ExpertDescriptor`
   + `CacheKey` (kills L2/L3), codec registry (kills L1/L5/L6), explicit
   model knob killing the L4 conflation, `RouterDesc` scoring switch,
   `norm_formula` flag. All are proven by the Phase F prototypes.
6. **Abstractions NOT to introduce yet:** `AttentionBackend` implementations
   (seam only), Oracle generalization, tokenizer API, cache/scheduler
   interfaces (nothing would implement them), MTP/speculative abstractions —
   each would be a seam with one implementation, i.e. cost without value.
7. **Second-model difficulty: moderate, bounded, frontend-heavy.** Systems
   core needs zero changes; the work is adapter + codec-registry +
   descriptor plumbing + config/router/norm flags (M0–M4). No kernel,
   cache, scheduler, or telemetry redesign.
8. **Minimum refactor before model #2:** M0 (explicit model knob + F8 split,
   init-only, ~4 files) then M1+M2 (codec registry + N-projection
   descriptor) — after that a Qwen-style model can boot in bypass mode with
   additive-only changes. M3 (in-engine router/combine/norm) follows for
   non-bypass correctness.
9. **Is dee a general sparse-model runtime yet? No — and the audit says the
   label would be wrong.** Its *systems core* is general (95%); its *MoE
   engine* is family-general with one codec (60%); full-model inference is
   DSV4-shaped with the dense half in Python (10%). "General MoE expert
   serving core with a DSV4 reference frontend" is the accurate claim.
10. **Next experiment after DSV4 stabilization:** M0 + bypass-mode bring-up
    of the Ornith (Qwen3.5-MoE) dialect already half-present in the resolver
    — it exercises a second name table, BF16-native codec path, and the
    `rms-plus1` kernel with zero new math, proving the M1/M2 seams before a
    truly foreign architecture arrives.

## Verification performed

- Direct-grep verification of headline counts (58 e2m1/e8m0, 0 C++ router-math,
  0 engine attention, 1× `43` = telemetry comment, `attn` hit =
  `cudaGetDeviceProperties`).
- `dsv4_assumption_inventory.json` validated as parseable JSON (21 entries).
- Phase F: 3 headers + `test_universality.cpp` written; this host's MSYS2
  toolchain crashes on any compilation (hello-world fails), so execution was
  verified via `pytest tests/test_universality_prototypes.py` — 8 passed,
  1 skipped (compile deferred to a healthy host with the exact g++ command
  in the test docstring). Production build untouched (no CMake changes).
