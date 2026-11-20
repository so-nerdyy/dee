"""Cost-model simulator tests: q* enumeration, no hardcoded equations."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kt_cpu_bridge.cost_model import plan_split, plan_table  # noqa: E402


def test_all_cpu_when_transfer_dominates():
    p = plan_split(m=6, t_cpu=2.0, t_h2d=50.0, t_gpu=1.0)
    assert p.q_star == 0
    assert p.makespan_ms == pytest.approx(12.0)


def test_all_gpu_when_cpu_dominates():
    p = plan_split(m=6, t_cpu=50.0, t_h2d=1.0, t_gpu=1.0)
    assert p.q_star == 6


def test_split_interior_balances():
    # t_cpu=10, t_h2d=1, t_gpu=1 overlap: brute-force check q* minimizes max.
    p = plan_split(m=8, t_cpu=10.0, t_h2d=1.0, t_gpu=1.0)
    rows = plan_table(8, 10.0, 1.0, 1.0)
    best = min(rows, key=lambda r: (r["makespan_ms"], r["q"]))
    assert p.q_star == best["q"]
    assert p.makespan_ms == pytest.approx(best["makespan_ms"])


def test_tie_prefers_smallest_q():
    p = plan_split(m=4, t_cpu=0.0, t_h2d=0.0, t_gpu=0.0)
    assert p.q_star == 0 and p.makespan_ms == 0.0


def test_zero_missing():
    p = plan_split(m=0, t_cpu=5.0, t_h2d=5.0, t_gpu=5.0)
    assert (p.q_star, p.makespan_ms) == (0, 0.0)


def test_no_overlap_never_better_than_overlap():
    a = plan_split(m=6, t_cpu=4.0, t_h2d=3.0, t_gpu=2.0, transfer_overlaps=True)
    b = plan_split(m=6, t_cpu=4.0, t_h2d=3.0, t_gpu=2.0, transfer_overlaps=False)
    assert a.makespan_ms <= b.makespan_ms + 1e-9


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        plan_split(m=-1, t_cpu=1.0, t_h2d=1.0, t_gpu=1.0)
    with pytest.raises(ValueError):
        plan_split(m=2, t_cpu=-1.0, t_h2d=1.0, t_gpu=1.0)
