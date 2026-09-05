#!/usr/bin/env python3
"""Tests for run_evidence.py (one-command evidence entry point).

All host-runnable: environment probing (real + mocked), BLOCKED bundle
schema with a zero-invented-timings invariant, live-path argument gating,
and ranking hold-by-default on fixture data. No GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

import run_evidence  # noqa: E402
from run_evidence import (  # noqa: E402
    CLASSIFICATION_BLOCKED,
    EXIT_BLOCKED,
    blocked_bundle,
    main,
    probe_environment,
)


def _no_numbers(obj) -> bool:
    """True if no int/float timing-like values hide anywhere (bools ok)."""
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return False
    if isinstance(obj, dict):
        return all(_no_numbers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_no_numbers(v) for v in obj)
    return True


def test_probe_reports_no_gpu_here():
    env = probe_environment()
    assert env["live_dual_t4"] is False
    assert env["cuda"] is False
    assert env["gpus"] == []


def test_blocked_bundle_invents_zero_timings(tmp_path):
    env = {"nvidia_smi": False, "gpus": [], "cuda": False,
           "torch": "x", "sm75_only": False, "live_dual_t4": False}
    bundle = blocked_bundle(tmp_path, env, "unit-test")
    assert bundle["classification"] == CLASSIFICATION_BLOCKED
    assert bundle["timings_invented"] == 0
    for name in ("environment.json", "host-sync-attribution.json",
                 "per-layer.csv", "per-token.json", "closure.json",
                 "abc-mechanics.json", "candidate-ranking.json",
                 "next-ab.json", "bundle.json"):
        assert (tmp_path / name).is_file(), name
    for name in ("host-sync-attribution.json", "closure.json",
                 "abc-mechanics.json", "candidate-ranking.json"):
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert _no_numbers(payload), name
        text = json.dumps(payload)
        assert "BLOCKED_LIVE_GPU" in text
    header = (tmp_path / "per-layer.csv").read_text(encoding="utf-8").splitlines()
    assert len(header) == 1 and "provenance" in header[0]
    for mode in ("profile-off", "profile-on"):
        status = json.loads((tmp_path / mode / "status.json").read_text(encoding="utf-8"))
        assert status["classification"] == CLASSIFICATION_BLOCKED


def test_main_returns_blocked_exit(tmp_path, capsys):
    code = main(["--out", str(tmp_path)])
    assert code == EXIT_BLOCKED
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == CLASSIFICATION_BLOCKED


def test_live_path_requires_args(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_evidence, "probe_environment",
                        lambda: {"nvidia_smi": True, "gpus": [], "cuda": True,
                                 "torch": "x", "sm75_only": True,
                                 "live_dual_t4": True})
    code = main(["--out", str(tmp_path)])
    assert code == 2
    out = json.loads(capsys.readouterr().out)
    assert out["classification"] == "LIVE_HOST_MISSING_ARGS"


def test_ranking_defaults_unmeasured(tmp_path):
    ranking = json.loads((tmp_path / "candidate-ranking.json").read_text(encoding="utf-8")) \
        if (tmp_path / "candidate-ranking.json").is_file() else None
    if ranking is None:
        env = {"nvidia_smi": False, "gpus": [], "cuda": False,
               "torch": "x", "sm75_only": False, "live_dual_t4": False}
        blocked_bundle(tmp_path, env, "unit-test")
        ranking = json.loads((tmp_path / "candidate-ranking.json").read_text(encoding="utf-8"))
    assert all(v["rank"] == "UNMEASURED" for v in ranking["candidates"].values())
    nxt = json.loads((tmp_path / "next-ab.json").read_text(encoding="utf-8"))
    assert "no A/B is specified" in nxt["next"]
