"""Tests for theory.batch_cliff_replay (b=8 host-hit cliff from the real 8-prompt journals)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

from theory import batch_cliff_replay as r  # noqa: E402

REC = 13369344


def test_lru_replay_basics():
    steps = [[(0, 1), (0, 2)], [(0, 1), (0, 2)], [(0, 3)], [(0, 1)]]
    # capacity 2: step 1 hits both, step 2 misses and evicts (0,1), step 3 misses
    out = r.replay_lru(steps, 2, window=(1, 3))
    assert (out["hits"], out["lookups"]) == (2, 4)
    # a loop larger than the cache never hits
    loop = [[(0, i) for i in range(5)]] * 6
    assert r.replay_lru(loop, 4, window=(1, 5))["hit"] == 0.0
    assert r.replay_lru(loop, 5, window=(1, 5))["hit"] == 1.0


@pytest.fixture(scope="module")
def steps():
    return r.stream(r.load_cohort())


def test_rebuilt_stream_matches_r5(steps):
    decode = [len(x) for x in steps[1:65]]
    assert sum(decode) / len(decode) == pytest.approx(r.R5_MEASURED["step_set_mean"], rel=0.01)
    lo, hi = r.R5_WINDOW
    looks = sum(len(x) for x in steps[lo:hi + 1])
    assert looks == pytest.approx(r.R5_MEASURED["window_lookups"], rel=0.03)


def test_replay_reproduces_both_r5_b8_rows(steps):
    for row in r.R5_MEASURED["rows"]:
        hit = r.replay_lru(steps, r.capacity_records(row["host_gib"], REC))["hit"]
        assert hit == pytest.approx(row["measured_hit"], abs=0.02)


def test_q7_repeats_q0_so_seven_streams_suffice(steps):
    assert r.stream(r.load_cohort()[:7]) == steps


def test_add5_rows_order(steps):
    rows = {x["host_gib"]: x for x in r.add5_rows(steps, REC)}
    assert rows[18]["host_hit_rate"]["nominal"] == 0.0
    assert rows[24]["host_hit_rate"]["nominal"] == pytest.approx(0.4604, abs=1e-4)
    assert rows[32]["host_hit_rate"]["nominal"] > rows[24]["host_hit_rate"]["nominal"]
