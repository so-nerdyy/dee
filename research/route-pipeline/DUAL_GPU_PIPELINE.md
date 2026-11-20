# Dual-GPU pipelining (Phase D; 2× T4, split at layer 22)

Geometry: cuda0 holds layers 0–21, cuda1 holds 22–42 with independent caches
(P2.3 corrected-trace replay). Hidden states cross A→host→B (no P2P
assumed); each side's transfer stream is independent.

## Can known-for-A work stage while B computes?

- **Decode, single token: no official overlap.** Execution is layer-serial:
  while A computes layers ≤21, no B-side route is known (R5 chain crosses
  the handoff); while B computes, A has no further work for this token.
  The handoff (`handoff_gpu` in `sim.py`) is itself on the critical path.
- **Hash prefetch crosses the boundary usefully in one direction only:**
  layers 0–2 live on A and are known at token start — but A computes first
  anyway, so the win is intra-A head start, not A↔B overlap. No hash layers
  live on B.
- **Host-side staging is GPU-independent:** SSD→pinned fills need no GPU.
  But fills need official ids, and ids follow the same R5 chain — so the
  transfer streams' independence buys nothing without earlier routes.

## Barriers audited (removed NONE — analysis only)

1. **Per-layer `cudaStreamSynchronize`** (`engine.cpp:1037-1054`, forced by
   the public overload `:748-754`): biggest avoidable candidate. The
   consumer (torch combine on another stream) needs only an event wait, not
   a host round-trip; the tree already proves the pattern —
   `moe_forward_batch_device_impl(..., synchronize_output=false, ...)` is
   used by the combined path (`:1927-1935`). Safe removal requires showing
   every reader waits on the exported event; that proof is listed as
   required work, not done here.
2. **Route-ID D2H + full torch-stream sync per layer**
   (`layer_candidate.py:448`): the bytes are 24 B at decode; the sync is the
   cost. Could become event-ordered host-readable staging (double-buffered
   ids + event poll) — needs the same reader-proof as (1).
3. **Host-driven per-rank `addcmul_` combine loop** (6 sequential launches):
   dependency-legal but launch-serial; a single indexed kernel preserves
   order (numerics review required — NOT proposed here, kernel work is out
   of scope for this track).
4. **Sequential prepare→stage host loop** (`engine.cpp:880-886`): batch
   submission (LEGAL_OVERLAP.md C.2) needs no barrier change at all.

No synchronization was removed or weakened by this track.
