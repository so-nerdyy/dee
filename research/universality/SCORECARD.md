# Universality scorecard (Phase G)

Engineering estimates, not measured facts. Basis for each number is the cited
`UNIVERSALITY_AUDIT.md` item (§) and inventory entry (H/L). Confidence =
strength of the source evidence. "Blocker" = the one thing to fix first.

| Category | Generic now | Confidence | Evidence | Main blocker |
|---|---|---|---|---|
| Storage / runtime (mmap, lookup, materialize) | 95% | high | §§1,6; L5 only | `F8` conflation + codec gate in `open()` (L1/L5) |
| RAM cache | 100% | high | §2 | none (re-tune only) |
| VRAM cache | 100% | high | §3 | none (re-tune only) |
| Scheduler (admission/eviction) | 95% | high | §§4,9 | nothing structural; `avail_layer` fallback is convenience |
| Prefetch | 70% | high | §5; L2 | `[3]`/`[6]` Transfer shape + FP4 entry points |
| CPU/GPU planner (placement) | 80% | med-high | §§6,10 | packed-residency branch; next codec needs enum + stage branch |
| Expert execution (SwiGLU) | 75% | med-high | §13 | mock combine (L9); kernels themselves already generic |
| Codec layer | 30% | high | §12; L1/L5/L6 | single-codec enum, shared e2m1 decode, no registry |
| Router | 50% | med | §14; L7 | hardcoded name + softmax-only; bypass exists so non-blocking |
| Attention / state (in-engine) | 0%* | high | §15 | *by design: 100% correctly external in HF; no in-engine work wanted |
| Full-model frontend | 20% | med | §§16–20; L8/L10/L11 | no `config.json` parsing; defaults; norm flag; shared/MTP unwired |
| Complete plug-and-play model support | 5% | med | all above | no second adapter exists yet (expected — this track forbids rewrites) |

## Overall rollups

- **Systems core: ~95% model-independent.** Items §§2,3,4,6,7,8,9 are
  GENERIC by source inspection (opaque `(key,nbytes)`, no tensor names, no
  codec branches). The remaining 5% is tuning defaults and one comment.
- **MoE engine: ~60% generic.** Kernels, scheduling keys, placement math,
  and the bypass seams are generic; descriptor shape (L2), codec (L1/L6),
  router (L7), and combine (L9) need the boundary work in MIGRATION_PLAN
  M1–M3.
- **Full-model plug-and-play: ~10%.** dee.cpp is a routed-expert server +
  norm helper behind a Python dense frontend, not a full-model runtime.
  Calling it a general sparse-model runtime today would be wrong (see
  FINAL_SUMMARY Q9); calling its systems core general is source-grounded.
