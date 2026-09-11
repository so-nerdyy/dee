# PHASE2_NEXT_BOTTLENECK.md — what binds after Phase-2D (prediction, not measurement)

Assumes: Phase-2 memory hierarchy installed (FxR warmup-pin host tier per
Flash's PHASE2_RECOMMENDATION.md + VRAM priority-artifact repair) AND
FIXED_SLOT_STAGING built (this track's recommendation). All forward numbers
are DERIVED projections / SIMULATED carries, labeled as such.

## Projected wall stack (decode, 16-token response scale, @0.29 GiB/s bank)

| Stage | Today (MEASURED) | After FxR host tier (DERIVED) | After + fixed slots (DERIVED) |
|---|---|---|---|
| Critical fill (cold bytes) | 42.0 s | ~26–33 s (−22–40% cold bytes at 16–17 GiB; warmup-pin 60% vs LRU 51% hits) | ~26–33 s (slots don't cut bytes) |
| Stage-enqueue + reservation + memset/memcpy (host serial) | ~14 s (9.4 + 4.6 + 0.8–2.8, overlapping) | ~14 s | ~5–8 s (−5–9 s prize) |
| Output sync (required drain) | 4.9 s | 4.9 s | 4.9 s |
| Unknown (dense attn + orchestration) | 9.2 s | 9.2 s | 9.2 s |
| Combine / D2H / readiness / compute-dispatch | ~0.5 s | ~0.5 s | ~0.5 s |
| **Decode total** | **66.2 s** | **~55–61 s** | **~46–56 s** |

## The next bottleneck (in order)

1. **Cold-fill bytes again (still #1).** Even after both mechanisms, ~26–33 s
   of ~46–56 s (≈55–60%) remains bank-bound cold fills at 0.29 GiB/s. The
   hierarchy moved the floor; it did not remove the device. Next levers in
   Flash's domain (more host RAM to 24 GiB pooled → 72% hits; bank placement
   on faster tier — at 3 GiB/s the whole fill wall is <7 s and policy gaps
   collapse). Class-1 work is NOT done; it just got cheaper per byte.
2. **Dense attention + orchestration (~9 s, UNKNOWN bucket).** Becomes the
   #2 exposed term (~17–20% of the post-2D wall) and is currently
   uninstrumented torch-path + Python-gap time. It was invisible behind fills;
   fixed slots + FxR will expose it. Next: break the 9.2 s down (attn kernels
   vs journal/checkpoint vs Python orchestration) before any kernel project.
3. **Required output sync (4.9 s, uniform).** Grows in share (→~10%) as fills
   shrink, but remains a required drain with no legal host work (R5). Only a
   cross-token (speculative) redesign changes this; exact-mode has no lever.
4. **Expert compute (1.5 s device).** Still noise. Component-split and kernel
   work stay unjustified until fills + attention are an order of magnitude
   smaller. The C1/C2 gate stands.

## What would invalidate this ordering

- Bank faster than ~1 GiB/s (fills collapse; attention becomes #1 immediately).
- Host RAM ≥ 29.4 GiB working set (static pin → 100% hits; fills → ~0;
  wall → attention + sync ≈ 15 s; serving becomes compute/attention-bound).
- A causal ≥60%-useful predictor + router-ahead scheduler (rewrites both the
  fill row and the sync row; research track only).

## Standing instruction for the post-2D campaign

Measure in this order: (i) cold-byte floor at the new hierarchy (ledger
re-analysis with terminal `result`), (ii) attention/orchestration breakdown
of the ~9 s bucket, (iii) fixed-slot A/B against run noise with abort
criteria. Do not start kernel, codec, GDS, or predictor implementation on the
basis of this projection — every number above the MEASURED column is a
hypothesis awaiting the live A/Bs.
