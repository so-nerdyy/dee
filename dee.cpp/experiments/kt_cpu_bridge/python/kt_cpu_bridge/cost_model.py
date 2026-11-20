"""Scheduler cost-model simulator (Phase E).

Answers: for m missing experts, split q to GPU (transfer+exec) and m-q to CPU
(exec) so makespan min max(T_gpu(q), T_cpu(m-q)) is minimized — by ENUMERATION
(dee top-k tiny: top-6/8). No hardcoded FreeToken equations, no live-scheduling
change; pure data in, plan out. Codex wires measured t_cpu/t_h2d/t_gpu later.

Conventions (ms/expert unless noted):
  t_cpu  = CPU execution ms/expert (decode, m==1 regime; prefill uses t_cpu_mat)
  t_h2d  = H2D transfer ms/expert (packed e2m1+e8m0 bytes -> device)
  t_gpu  = GPU execution ms/expert (post-resident SwiGLU slice)
If transfer_overlaps_forecast/compute, H2D for q experts overlaps CPU work.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitPlan:
    m: int
    q_star: int
    t_gpu_makespan_ms: float
    t_cpu_makespan_ms: float
    makespan_ms: float
    transfer_overlaps: bool


def _t_gpu(q: int, t_h2d: float, t_gpu: float, overlap: bool) -> float:
    if q <= 0:
        return 0.0
    if overlap:
        # First expert pays transfer, rest pipeline: max(chain) approx =
        # t_h2d + q*t_gpu when compute-bound, q*t_h2d + t_gpu when xfer-bound.
        # Model as max of the two bounds (roofline-style, no hardcode).
        return max(t_h2d + q * t_gpu, q * t_h2d + t_gpu)
    return q * (t_h2d + t_gpu)


def _t_cpu(n_cpu: int, t_cpu: float) -> float:
    return max(0, n_cpu) * t_cpu


def plan_split(
    m: int,
    t_cpu: float,
    t_h2d: float,
    t_gpu: float,
    transfer_overlaps: bool = True,
) -> SplitPlan:
    """Enumerate q in [0, m]; q* = argmin max(T_gpu(q), T_cpu(m-q)).

    Ties -> smallest q (prefer CPU, preserve VRAM bandwidth for residents).
    """
    if m < 0:
        raise ValueError("m must be >= 0")
    for v in (t_cpu, t_h2d, t_gpu):
        if not (v >= 0):
            raise ValueError("times must be >= 0 and finite")
    best_q, best_span = 0, float("inf")
    best_g = best_c = 0.0
    for q in range(m + 1):
        g = _t_gpu(q, t_h2d, t_gpu, transfer_overlaps)
        c = _t_cpu(m - q, t_cpu)
        span = max(g, c)
        if span < best_span - 1e-12:
            best_span, best_q, best_g, best_c = span, q, g, c
    return SplitPlan(m=m, q_star=best_q, t_gpu_makespan_ms=best_g,
                     t_cpu_makespan_ms=best_c, makespan_ms=best_span,
                     transfer_overlaps=transfer_overlaps)


def plan_table(m: int, t_cpu: float, t_h2d: float, t_gpu: float,
               transfer_overlaps: bool = True) -> list:
    rows = []
    for q in range(m + 1):
        g = _t_gpu(q, t_h2d, t_gpu, transfer_overlaps)
        c = _t_cpu(m - q, t_cpu)
        rows.append({"q": q, "t_gpu_ms": g, "t_cpu_ms": c, "makespan_ms": max(g, c)})
    return rows


@dataclass(frozen=True)
class MicrobenchSample:
    kind: str  # "cpu" | "h2d" | "gpu"
    ms_per_expert: float
    bytes_per_expert: int = 0
    hidden: int = 0
    inter: int = 0
    replicas: int = 0

    @property
    def bandwidth_gbps(self) -> float:
        if self.ms_per_expert <= 0 or self.bytes_per_expert <= 0:
            return 0.0
        return (self.bytes_per_expert / 1e9) / (self.ms_per_expert / 1e3)
