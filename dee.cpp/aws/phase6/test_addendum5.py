"""Tests for addendum5.py (grades p6m-l4-r6 against Addendum 5).

Run from this directory:  python -m pytest -q test_addendum5.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import addendum5

B, W = 8, 32


def _summary(run_id):
    return {"run_id": run_id, "model": "dsv4-flash", "gpu": "L4", "n_gpus": 1, "commit": "abc",
            "host_gib_requested": 16.0, "budget_mib": 3584, "n_tokens": 65, "cohort": "0-7"}


def _row(tmp: Path, role: str, *, step_ms: float, reads_per_step: float, hits_per_step: float,
         misses_per_step: float, hint_hits=0.0, hint_reads=0.0, resident=0.0, loads=100.0,
         tokens=("a", "b"), journal='{"forward_step": 0, "layer": 0, "expert_ids_rank_order": [[1, 2]]}\n',
         lookahead=None) -> Path:
    d = tmp / f"p6m-l4-r6-b8-h16-{role}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(_summary(d.name)))
    result = {"status": "COMPLETE", "mode": "cohort", "cohort_k": B,
              "gpu_environment": {"single_gpu_mode": True, "gpu_count": 1},
              "decode_timings_ms": [step_ms] * 64,
              "rows": [{"prompt_index": i, "token_ids_sha256": t} for i, t in enumerate(tokens)],
              "placement": {"host_eviction_policy": role},
              "prefetch_lookahead": lookahead}
    (d / "result-c0.json").write_text(json.dumps(result))
    (d / "result.json").write_text(json.dumps({"status": "RUNNING"}))
    cp = []
    for s in range(65):
        cp.append({"step": s,
                   "expert_store": {"cuda0": {"source_reads": reads_per_step * s}},
                   "host_pack": {"cuda0": {"hits": hits_per_step * s, "misses": misses_per_step * s,
                                           "hint_hits": hint_hits * s, "hint_reads": hint_reads * s}},
                   "engine_stats": {"cuda0": {"resident_hits": resident * s,
                                              "cache_hits": resident * s,
                                              "cache_loads": loads * s}}})
    (d / "generated_checkpoint-c0.jsonl").write_text("\n".join(json.dumps(r) for r in cp) + "\n")
    (d / "routed_experts-c0.jsonl").write_text(journal)
    return d


def _four(tmp, **over):
    rows = {
        "lru": dict(step_ms=13500, reads_per_step=1520, hits_per_step=0, misses_per_step=1520),
        "nu": dict(step_ms=10600, reads_per_step=920, hits_per_step=640, misses_per_step=880,
                   resident=12, loads=88),
        "nu-prev": dict(step_ms=8600, reads_per_step=930, hits_per_step=1320, misses_per_step=200,
                        hint_hits=700, hint_reads=900, resident=12, loads=88),
        "nu-router": dict(step_ms=9000, reads_per_step=950, hits_per_step=1300, misses_per_step=220,
                          lookahead={"recall": 0.71, "precision": 0.52}),
    }
    for role, kw in over.items():
        rows[role].update(kw)
    return [_row(tmp, role, **kw) for role, kw in rows.items()]


def test_grades_a_run_that_matches_the_predictions(tmp_path):
    dirs = _four(tmp_path)
    evs = {addendum5.role_of(d): addendum5.score.load_evidence(d) for d in dirs}
    card = addendum5.grade_all(evs)
    assert card["A0"]["gate"] == "PASS"
    b = card["bands"]
    assert b["A1"]["measured"] == 0.0 and b["A1"]["grade"] == "PASS"
    assert b["A2"]["measured"] == pytest.approx(640 / 1520)
    assert b["A3"]["measured"] == pytest.approx(920 / 8)
    assert b["A4"]["measured"] == pytest.approx(0.12)
    assert b["A7"]["measured"] == pytest.approx(700 / 900)
    assert b["A8"]["measured"] == pytest.approx(13500 / 10600)
    assert all(v["grade"] == "PASS" for v in b.values()), b
    assert card["nu_router"]["recall"] == 0.71
    assert card["nu_router"]["grade"] == "PASS"
    assert addendum5.main([str(d) for d in dirs]) == 0


def test_a8_kills_when_next_use_is_not_faster(tmp_path):
    dirs = _four(tmp_path, nu={"step_ms": 13500})
    evs = {addendum5.role_of(d): addendum5.score.load_evidence(d) for d in dirs}
    card = addendum5.grade_all(evs)
    assert card["bands"]["A8"]["grade"] == "KILLED"
    assert card["overall"] == "KILLED"


def test_a0_failure_voids_that_row(tmp_path):
    dirs = _four(tmp_path, **{"nu-prev": {"tokens": ("a", "X")}})
    evs = {addendum5.role_of(d): addendum5.score.load_evidence(d) for d in dirs}
    card = addendum5.grade_all(evs)
    assert card["A0"]["gate"].startswith("FAIL")
    for bid in ("A5", "A6", "A7", "A9", "A10"):
        assert card["bands"][bid]["grade"].startswith("VOID")
    assert card["bands"]["A2"]["grade"] == "PASS"
    assert addendum5.main([str(d) for d in dirs]) == 1


def test_vram_hit_counts_inflight_hits_as_requests(tmp_path):
    dirs = _four(tmp_path)
    nu = next(d for d in dirs if d.name.endswith("-nu"))
    lines = (nu / "generated_checkpoint-c0.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines]
    for r in recs:
        r["engine_stats"]["cuda0"]["inflight_hits"] = 20 * r["step"]
    (nu / "generated_checkpoint-c0.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    evs = {addendum5.role_of(d): addendum5.score.load_evidence(d) for d in dirs}
    card = addendum5.grade_all(evs)
    assert card["bands"]["A4"]["measured"] == pytest.approx(12 / (12 + 20 + 88))


def test_every_row_void_is_not_a_pass(tmp_path):
    dirs = _four(tmp_path, nu={"tokens": ("X", "b")}, **{"nu-prev": {"tokens": ("X", "b")}})
    evs = {addendum5.role_of(d): addendum5.score.load_evidence(d) for d in dirs}
    evs["lru"].result = None
    card = addendum5.grade_all(evs)
    assert all(b["grade"].startswith("VOID") for b in card["bands"].values())
    assert card["overall"].startswith("VOID")


def test_role_comes_from_the_run_id_suffix():
    assert addendum5.role_of(Path("runs/p6m-l4-r6-b8-h16-nu")) == "nu"
    assert addendum5.role_of(Path("runs/p6m-l4-r6-b8-h16-nu-prev")) == "nu-prev"
    assert addendum5.role_of(Path("runs/p6m-l4-r6-b8-h16-nu-router")) == "nu-router"
    with pytest.raises(SystemExit):
        addendum5.role_of(Path("runs/p6m-l4-r6-b8-h16-other"))
