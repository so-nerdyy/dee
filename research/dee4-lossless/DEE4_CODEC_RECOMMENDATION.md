# DEE4 codec recommendation

## Decision: NO_CODEC_WORTH_BUILDING

One line: the best real lossless codec (zstd-10, whole record) measures
**0.9178 mean / 0.9196 worst** over all 2364 bank records — SIDE_OPTIMIZATION
(>= 0.90) — worth ~3.5 s off a 42 s fill bucket at the cost of a production
decompressor, a second format, and CPU per fill. The theoretical ceiling
(order-1 bound, worst-case 0.9004) barely touches USEFUL and no real codec
attains it.

## Evidence behind the decision

1. **Packed FP4 weights (94.1% of bytes) are maximum-entropy noise.**
   lz4 1.0000 on all 2364 records; zstd 0.965; order-0 0.9597; real rANS
   0.9597 (== bound, as theory demands); nibble streams 3.86 bits/nibble;
   bit-planes 0.95–1.00 (sign bit 0.99999); zeros 0.4%, longest run 3.
   No byte, nibble, bit-plane, or context model has anything to grip.
2. **Scales (5.9%) compress well (0.12–0.19) but are too small to matter.**
   A scales-only packing captures ~5 points of the ~8 available — and is
   the only piece worth revisiting if the format is ever relaid.
3. **Global state buys nothing.** zstd-dict (train 256 -> test 64):
   0.9201 vs independent 0.9210. Exact region dedup: 2364/2364 unique.
4. **No lane/pattern dependence.** Worst-case ~= mean everywhere; the
   bank is homogeneous. There is no "easy subset" to skim.
5. **The lever stays FEED-side.** Phase 1 closed the device at 0.3 GB/s
   cold; this scan closes the byte axis at ~0.92. Remaining fill-wall
   levers, in order: fewer misses (residency/policy), smaller records
   (requires format change, not a codec), earlier submission (overlap).

## Revisit conditions (re-run this exact harness if)

- The bank moves to a slower/cheaper store (the 3.5 s scales with
  device time per byte).
- Record contents change (different quantization, added metadata/
  padding — relayout first, this scan second).
- A host tier makes fills bigger or more frequent (recompute
  `DEE4_STORAGE_PROJECTION.md` with the new miss profile first).

## Explicitly not recommended

- Custom order-1 / nibble-rANS decoder: ~2 points over zstd-10 best
  case, large complexity, still SIDE/USEFUL-borderline.
- Nibble-split + LZ: measured 1.04 (worse than raw).
- RLE/zero packing: 0.4% zeros, max run 3.
- Any lossy quantization (out of scope; byte-identicalness held for
  2364/2364 + 8/8 pilot + 64/64 dict-study records).
- Production implementation of any of the above at this time.
