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


def test_probe_reports_current_host_without_assumptions():
    # No invariant about THIS host: only structural shape is asserted.
    # Behavior per environment is covered by mocked tests below, so the
    # suite passes identically on CPU-only and dual-T4 hosts.
    env = probe_environment()
    assert set(env) >= {"nvidia_smi", "gpus", "cuda", "torch", "sm75_only",
                        "live_dual_t4"}
    assert env["live_dual_t4"] == bool(env["cuda"] and env["sm75_only"])
    assert isinstance(env["gpus"], list)


def _mock_env(monkeypatch, nvidia_out=None, torch_cuda=True,
              torch_version="2.7.1", no_torch=False):
    import run_evidence as re_mod
    monkeypatch.setattr(
        re_mod.shutil, "which",
        lambda name: "/usr/bin/nvidia-smi" if nvidia_out is not None else None)

    class _Proc:
        def __init__(self, stdout, returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return _Proc(nvidia_out or "")

    monkeypatch.setattr(re_mod.subprocess, "run", fake_run)
    if no_torch:
        import builtins
        real_import = builtins.__import__

        def hooked(name, *a, **k):
            if name == "torch":
                raise ImportError("mocked missing torch")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", hooked)
    else:
        import types
        fake = types.ModuleType("torch")
        fake.__version__ = torch_version
        fake.cuda = types.SimpleNamespace(is_available=lambda: torch_cuda)
        monkeypatch.setitem(sys.modules, "torch", fake)
    return re_mod.probe_environment()


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


def test_env_matrix_no_gpu(monkeypatch):
    env = _mock_env(monkeypatch, nvidia_out=None, torch_cuda=False)
    assert env == {"nvidia_smi": False, "gpus": [], "cuda": False,
                   "torch": "2.7.1", "sm75_only": False, "live_dual_t4": False}


def test_env_matrix_one_t4(monkeypatch):
    env = _mock_env(monkeypatch,
                    nvidia_out="0, Tesla T4, 7.5\n", torch_cuda=True)
    assert env["sm75_only"] is False  # need exactly two
    assert env["live_dual_t4"] is False


def test_env_matrix_dual_t4(monkeypatch):
    env = _mock_env(monkeypatch,
                    nvidia_out="0, Tesla T4, 7.5\n1, Tesla T4, 7.5\n",
                    torch_cuda=True)
    assert env["sm75_only"] is True
    assert env["live_dual_t4"] is True
    assert [g["name"] for g in env["gpus"]] == ["Tesla T4", "Tesla T4"]


def test_env_matrix_wrong_cc(monkeypatch):
    env = _mock_env(monkeypatch,
                    nvidia_out="0, NVIDIA A100, 8.0\n1, NVIDIA A100, 8.0\n",
                    torch_cuda=True)
    assert env["sm75_only"] is False
    assert env["live_dual_t4"] is False


def test_env_matrix_cuda_unavailable_despite_smi(monkeypatch):
    env = _mock_env(monkeypatch,
                    nvidia_out="0, Tesla T4, 7.5\n1, Tesla T4, 7.5\n",
                    torch_cuda=False)
    assert env["nvidia_smi"] is True and env["cuda"] is False
    assert env["live_dual_t4"] is False


def test_env_matrix_no_torch(monkeypatch):
    env = _mock_env(monkeypatch, nvidia_out=None, no_torch=True)
    assert env["torch"] is None and env["cuda"] is False
    assert env["live_dual_t4"] is False
def test_main_returns_blocked_exit(monkeypatch, tmp_path, capsys):
    # Deterministic on ANY host: force the no-GPU probe (the live path is
    # covered by the mocked matrix above, not by this machine's hardware).
    import run_evidence as re_mod
    monkeypatch.setattr(re_mod, "probe_environment",
                        lambda: {"nvidia_smi": False, "gpus": [], "cuda": False,
                                 "torch": "x", "sm75_only": False,
                                 "live_dual_t4": False})
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


def test_dry_run_validates_without_executing(tmp_path, capsys):
    from run_evidence import dry_run_live
    env = probe_environment()
    report = dry_run_live(
        tmp_path,
        type("Args", (), {"command": "python mock_campaign.py",
                          "fixture_shard": None})(),
        env)
    assert set(report["checks"]) >= {"dual_t4", "command_exists",
                                     "out_writable", "source_sha",
                                     "profiler_source", "python_modules",
                                     "abc_runner", "schemas"}
    assert report["ready"] == all(c["pass"] for c in report["checks"].values())
    assert (tmp_path / "dry-run.json").is_file()
    # On this CPU-only host the GPU/command checks honestly fail.
    assert report["checks"]["profiler_source"]["pass"] is True
    assert report["checks"]["schemas"]["pass"] is True
