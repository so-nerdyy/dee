#!/usr/bin/env python3
"""Tests for the exact-critical-path simulator and A/B ingestion path.

Runs against the sealed v65 route journal (read-only) when present; skips
with a clear message otherwise. No sealed evidence is modified.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
REPO = HERE.parents[2]                      # ../dee-critical-path worktree root
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

from exact_critical_path_sim import (  # noqa: E402
    default_params, simulate, gpu_layer_uniques, load_route_bundles,
    find_trace, demand_plan, V65,
)

REPO_CANDIDATES = [
    REPO,
    Path("C:/Users/carth/Downloads/dynamic_expert_eviction"),   # main checkout
]


def _journal() -> Path | None:
    for root in REPO_CANDIDATES:
        cand = root / "dee.cpp/tmp/v65-terminal-fetch-20260903T0202Z/routed_experts.jsonl"
        if cand.exists():
            return cand
    return None


@pytest.fixture(scope="module")
def uniques():
    jp = _journal()
    if jp is None:
        pytest.skip("sealed v65 route journal not present on this machine")
    return gpu_layer_uniques(load_route_bundles(jp))


@pytest.fixture(scope="module")
def params():
    cal = RESEARCH / "results" / "calibration.json"
    if cal.exists():
        return json.loads(cal.read_text())["params"]
    return default_params()


# ---------------------------------------------------------------------------
# Route journal loading
# ---------------------------------------------------------------------------

def test_journal_loads_16_forwards_43_layers(uniques):
    assert len(uniques) == 16            # 16 forwards (1 prefill + 15 decode)
    for fwd in uniques:
        assert len(fwd[0]) == 22         # cuda0 layers
        assert len(fwd[1]) == 21         # cuda1 layers


def test_journal_matches_sealed_selection_count(uniques):
    """Journal top-k selections across 16 forwards = 5676 (sealed provenance)."""
    jp = _journal()
    n = 0
    with jp.open() as fh:
        for line in fh:
            rec = json.loads(line)
            n += sum(len(row) for row in rec["expert_ids_rank_order"])
    assert n == 5676


def test_demand_plan_uses_sealed_per_step_counts(uniques):
    plan = demand_plan(uniques)
    assert len(plan) == 15
    s0 = plan[0]
    for g in (0, 1):
        tot = sum(l["uniques"] for l in s0[g])
        # unique experts per GPU per step must not exceed journal uniques
        assert tot <= sum(len(u) for u in uniques[1][g])
    # sealed step-0 ssd reads = 96 split across GPUs
    reads = sum(l["read"] for g in (0, 1) for l in s0[g])
    assert reads == pytest.approx(96, rel=0.01)


# ---------------------------------------------------------------------------
# Simulator invariants
# ---------------------------------------------------------------------------

def test_baseline_reproduces_sealed_wall(params, uniques):
    res = simulate(params, uniques)
    assert res["rel_error"] == pytest.approx(0, abs=0.03)   # within 3 %
    assert 64.0 < res["predicted_wall_s"] < 80.0


def test_more_staging_never_hurts(params, uniques):
    w0 = simulate({**params, "staging_lead_layers": 0}, uniques)["predicted_wall_s"]
    w8 = simulate({**params, "staging_lead_layers": 8}, uniques)["predicted_wall_s"]
    w42 = simulate({**params, "staging_lead_layers": 42}, uniques)["predicted_wall_s"]
    assert w8 <= w0 + 0.5
    assert w42 <= w8 + 0.5


def test_buffer_reuse_never_hurts(params, uniques):
    w0 = simulate({**params, "host_buffer_reuse": False}, uniques)["predicted_wall_s"]
    w1 = simulate({**params, "host_buffer_reuse": True}, uniques)["predicted_wall_s"]
    assert w1 <= w0 + 0.5


def test_faster_h2d_never_hurts(params, uniques):
    w1 = simulate({**params, "h2d_gbps": params["h2d_gbps"]}, uniques)["predicted_wall_s"]
    w2 = simulate({**params, "h2d_gbps": params["h2d_gbps"] * 2}, uniques)["predicted_wall_s"]
    assert w2 <= w1 + 0.5


def test_nonoverlap_is_never_faster(params, uniques):
    w_ov = simulate({**params, "compute_overlaps_copies": True}, uniques)["predicted_wall_s"]
    w_no = simulate({**params, "compute_overlaps_copies": False}, uniques)["predicted_wall_s"]
    assert w_no >= w_ov - 0.5


def test_faster_disk_never_hurts(params, uniques):
    w1 = simulate({**params, "ssd_aggregate_mb_s": params["ssd_aggregate_mb_s"]},
                  uniques)["predicted_wall_s"]
    w2 = simulate({**params, "ssd_aggregate_mb_s": params["ssd_aggregate_mb_s"] * 1.5},
                  uniques)["predicted_wall_s"]
    assert w2 <= w1 + 0.5


def test_step_walls_are_positive_and_ordered_fields(params, uniques):
    res = simulate(params, uniques)
    for s in res["steps"]:
        assert s["wall_ms"] > 0
        assert s["h2d_total_ms"] >= 0
        assert s["compute_only_ms"] >= 0
        assert s["reads"] >= 0 and s["pack_hits"] >= 0 and s["resident_hits"] >= 0
    assert len(res["steps"]) == 15


def test_sealed_totals_reproduced(params, uniques):
    """Over 15 decode steps the plan must consume exactly the sealed totals."""
    res = simulate(params, uniques)
    reads = sum(s["reads"] for s in res["steps"])
    assert reads == pytest.approx(sum(V65["per_step"]["ssd_reads"]), rel=0.01)
    hits = sum(s["resident_hits"] for s in res["steps"])
    assert hits == pytest.approx(sum(V65["per_step"]["resident_hits"]), rel=0.01)


# ---------------------------------------------------------------------------
# CLI + ingestion path
# ---------------------------------------------------------------------------

def test_cli_baseline_runs():
    r = subprocess.run([sys.executable, str(TOOLS / "exact_critical_path_sim.py"),
                        "--baseline"], capture_output=True, text=True,
                       cwd=str(REPO), timeout=180)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert "predicted_wall_s" in out and "rel_error" in out


def _write_ab(tmp_path, wb, wc):
    art = {"baseline": {"run_id": "T", "commit": "t", "variant": "baseline",
                        "decode_wall_s": wb, "decode_tokens": 15,
                        "reuse_telemetry": {"buffers_reused": 0}},
           "candidate": {"run_id": "T", "commit": "t", "variant": "candidate",
                         "decode_wall_s": wc, "decode_tokens": 15,
                         "reuse_telemetry": {"buffers_reused": 708}}}
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(art))
    return p


def test_ingest_agrees_on_improvement(tmp_path):
    p = _write_ab(tmp_path, 72.0, 60.0)
    r = subprocess.run([sys.executable, str(TOOLS / "ingest_ab.py"), "--ab", str(p),
                        "--out", str(tmp_path / "ing.json")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "ing.json").read_text())
    assert rep["direction_agrees"] is True


def test_ingest_flags_disagreement_on_regression(tmp_path):
    """Measured regression while the model predicts improvement => the tool
    must report direction disagreement (False), not paper over it."""
    p = _write_ab(tmp_path, 72.0, 80.0)
    r = subprocess.run([sys.executable, str(TOOLS / "ingest_ab.py"), "--ab", str(p),
                        "--out", str(tmp_path / "ing.json")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "ing.json").read_text())
    assert rep["direction_agrees"] is False


def test_ingest_single_arm_rejected(tmp_path):
    art = {"run_id": "T", "commit": "t", "variant": "baseline",
           "decode_wall_s": 72.0, "decode_tokens": 15}
    p = tmp_path / "arm.json"
    p.write_text(json.dumps(art))
    r = subprocess.run([sys.executable, str(TOOLS / "ingest_ab.py"), "--ab", str(p),
                        "--out", str(tmp_path / "ing.json")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode != 0
    assert "both arms" in r.stderr or "need both arms" in r.stderr


def test_ingest_missing_wall_reported_unknown(tmp_path):
    art = {"baseline": {"run_id": "T", "commit": "t", "variant": "baseline",
                        "decode_tokens": 15},
           "candidate": {"run_id": "T", "commit": "t", "variant": "candidate",
                         "decode_wall_s": 70.0, "decode_tokens": 15}}
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(art))
    r = subprocess.run([sys.executable, str(TOOLS / "ingest_ab.py"), "--ab", str(p),
                        "--out", str(tmp_path / "ing.json")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "ing.json").read_text())
    assert rep["measured"]["baseline"] == "UNKNOWN (decode_wall_s missing)"
    assert rep["direction_agrees"] is None
