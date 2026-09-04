#!/usr/bin/env python3
"""Prototype B: shared-expert overlap with exact join + contention model.

Layer-math audit (DSV4, source: deepseek_v4_moe_reference.moe_layer_forward,
pydee adapter combine): per layer,

    routed  = sum_rank weights[rank] * expert_{ids[rank]}(h)   (rank order kept)
    out     = routed + shared(h)                                (then residual)

True dependencies of shared(h): the layer input hidden state ONLY — available
before the router even runs. Its current placement after routed combine
(layer_candidate.py _run_experts tail) is code order, not dependence.

Exact join: EITHER branch may be computed first if the other is buffered;
the join `routed + shared` in that order is preserved (join() adds in the
documented order regardless of completion order). Combine-term order inside
`routed` is untouched (rank loop preserved by the caller, not modeled here).

Contention honesty: concurrency != speedup on one T4. Routed staging is
host/SSD+H2D (no SMs); routed GEMM and shared GEMMs share DRAM bandwidth and
SMs. The model reports overlap_window plus a contention factor c in [0,1]
(0 = fully serialized by shared bandwidth, 1 = perfect overlap) and the
resulting wall interval [max, sum]. c must be MEASURED (runner case B);
this module never assumes it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BranchResult:
    name: str
    complete_t_ms: float
    payload: object  # buffered branch output (opaque here)


def exact_join(routed: BranchResult, shared: BranchResult) -> tuple[object, dict]:
    """Join in documented order (routed + shared) regardless of completion
    order. Both branches must be complete; the join itself is an add."""
    if routed.payload is None or shared.payload is None:
        raise ValueError("join with incomplete branch (missing wait)")
    first = (routed.name if routed.complete_t_ms <= shared.complete_t_ms
             else shared.name)
    return (("add", routed.payload, shared.payload),
            {"first_done": first, "join_order": ["routed", "shared"]})


@dataclass
class OverlapWindow:
    routed_start_ms: float
    routed_end_ms: float
    shared_start_ms: float
    shared_end_ms: float

    def overlap_ms(self) -> float:
        return max(0.0, min(self.routed_end_ms, self.shared_end_ms)
                   - max(self.routed_start_ms, self.shared_start_ms))

    def wall_interval(self, contention: float) -> tuple[float, float]:
        """(best, worst) wall for the overlapped region. contention in [0,1]
        must come from measurement (runner case B)."""
        if not 0.0 <= contention <= 1.0:
            raise ValueError("contention in [0,1], measured")
        routed = self.routed_end_ms - self.routed_start_ms
        shared = self.shared_end_ms - self.shared_start_ms
        overlap = self.overlap_ms()
        # best: overlap fully hides the shorter branch; worst: serialization.
        best = max(routed, shared, routed + shared - overlap)
        worst = routed + shared
        # Interpolated by measured contention (0 -> worst, 1 -> best).
        expected = worst + contention * (best - worst)
        return (round(min(best, expected), 6), round(worst, 6))

    def extra_workspace_bytes(self, hidden: int, tokens: int,
                              dtype_bytes: int = 2) -> int:
        """Buffering one branch for the join: input + output rows."""
        return 2 * hidden * tokens * dtype_bytes

    def needs_join_sync(self) -> bool:
        return True  # exactly one event join; counted, not removed
