# Route-D2H sync audit

## Current path

Per layer: compact int32 route IDs (24 B at batch-1 decode) copied D2H,
then `torch.cuda.current_stream(ids.device).synchronize()`
(`layer_candidate.py:448` + `route_id_host_synchronizations`). The bytes are
trivial; the full-stream sync is the cost — it waits for ALL previously
queued stream work, not just the ID copy.

## Is the scope wider than necessary? Yes, in principle.

The CPU consumer (native scheduler grouping, `engine.cpp:775-797`) needs
only the ID bytes. Narrower legal pattern: order the copy after the router
kernel via an event, copy to pinned memory, and have the host wait on the
copy's completion only. Unrelated stream work (prior layer's tail, shared
compute, prefetches) need not be waited on.

## Do NOT assume yes — the residual wait is real

The CPU still must wait for the ID copy itself before grouping; if the
router kernel just launched, that wait covers the router too. The prize is
BOUNDED by unrelated stream work actually present at that point — which the
sealed profile does not break out (UNKNOWN). Model:
`event_handoff.narrow_d2h()` returns `avoidable_ms = unrelated_stream_ms`
(exactly) vs `unavoidable_ms = copy_ms`. Verdict with no unrelated work:
`LOW_EXPECTED_VALUE`; otherwise `WORTH_PROTOTYPING`. Tests:
`test_narrow_d2h_bounds_prize`.

## Prototype or reject

Prototyped as ordering model + runner case C (event vs sync cost bounds the
prize from the other side: if events cost ~sync, narrowing buys nothing).
Formal rejection deferred until case C measures both sides live. No
production sync touched either way.
