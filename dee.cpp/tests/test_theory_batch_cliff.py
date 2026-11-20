"""Tests for theory.batch_cliff (Phase-7 host-hit correction for batched decode)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

from theory import batch_cliff as bc  # noqa: E402


def _uniform(n=1000, p=0.05):
    return np.full(n, p)


def test_union_is_monotone_and_saturates():
    u = bc.step_inclusion(_uniform(), 4)
    vals = [bc.union_over_steps(u, k) for k in (1, 2, 4, 16, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(1000.0, rel=1e-6)


def test_cliff_is_one_step_set():
    pi = _uniform()
    u1 = bc.cliff_records(pi, 8)
    assert bc.step_lru_hit(pi, 8, u1 - 1) == 0.0
    assert bc.step_lru_hit(pi, 8, u1 + 1) > 0.0
    assert bc.reuse_horizon(bc.step_inclusion(pi, 8), u1 - 1) == 0


def test_cliff_moves_up_with_batch():
    pi = _uniform()
    cliffs = [bc.cliff_records(pi, b) for b in (1, 2, 4, 8, 16)]
    assert all(a < b for a, b in zip(cliffs, cliffs[1:]))


def test_hit_rises_with_capacity_and_reaches_one():
    pi = _uniform()
    hits = [bc.step_lru_hit(pi, 4, c) for c in (100, 300, 600, 900, 1000)]
    assert all(a <= b for a, b in zip(hits, hits[1:]))
    assert hits[-1] == pytest.approx(1.0)


def test_cap_never_raises_registered():
    pi = _uniform()
    for c in (50, 400, 2000):
        assert bc.corrected_host_hit(0.51, pi, 8, c) <= 0.51


def test_r5_rows():
    """Corrected nominal: b8-h16 collapses to 0 (measured 0); the three passing rows keep
    their registered nominal."""
    summary = bc.run(write=False)
    rows = {r["run"]: r for r in summary["r5_check"]}
    assert rows["p6m-l4-r5-b8-h16"]["corrected_nominal"] == 0.0
    for name in ("p6m-l4-r5-b1-h64", "p6m-l4-r5-b1-h16", "p6m-l4-r5-b8-h64"):
        assert rows[name]["corrected_nominal"] == rows[name]["registered_nominal"]
