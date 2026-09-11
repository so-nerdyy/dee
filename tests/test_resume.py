#!/usr/bin/env python3
"""Resume + manifest tests (Phase K).

Verifies: terminal stages with matching hashes are reused, tampered or
missing artifacts force re-execution, partial output is never reused, and
the manifest records terminal state per stage. No CUDA.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dee.cpp" / "experiments" / "route_pipeline"
sys.path.insert(0, str(EXP))

from run_evidence import (  # noqa: E402
    mark_stage,
    sha256_file,
    stage_verified,
)

MOCK = EXP / "mock_campaign.py"
RUN = EXP / "run_evidence.py"
TMP = Path("C:/Users/carth/AppData/Local/Temp/opencode/rpresume")


def _base_cmd(workname: str, scenario: int) -> list[str]:
    work = TMP / workname
    shutil.rmtree(work, ignore_errors=True)
    abc = work / "abc.json"
    subprocess.run([sys.executable, str(MOCK), "--out", str(work / "abcf"),
                    "--scenario", str(scenario), "--emit-abc", str(abc)],
                   capture_output=True, check=True)
    return [sys.executable, str(RUN), "--out", str(work / "ev"),
            "--command", f"{sys.executable} {MOCK}",
            "--prompt-hash", "mock-prompt-hash", "--mock-abc", str(abc),
            "--no-gpu-check", "--extra-cmd-args", f"--scenario {scenario}"]


def _ev(workname: str) -> Path:
    return TMP / workname / "ev"


def test_resume_reuses_verified_stages():
    cmd = _base_cmd("r4", 4)
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr[-2000:]
    work = _ev("r4")
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"profile", "abc", "rank"}
    assert all(v["status"] == "terminal" for v in manifest["stages"].values())
    bundle_before = (work / "bundle.json").read_text(encoding="utf-8")
    second = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr[-2000:]
    bundle = json.loads((work / "bundle.json").read_text(encoding="utf-8"))
    assert set(bundle["reused_stages"]) == {"profile", "abc"}
    assert json.loads(bundle_before)["next"] == bundle["next"]


def test_tampered_artifact_forces_rerun():
    cmd = _base_cmd("r4t", 4)
    subprocess.run(cmd, capture_output=True, check=True)
    ev = _ev("r4t")
    target = ev / "abc-mechanics.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["tampered"] = True
    target.write_text(json.dumps(payload), encoding="utf-8")
    again = subprocess.run(cmd + ["--resume"], capture_output=True, text=True)
    assert again.returncode == 0, again.stderr[-2000:]
    bundle = json.loads((ev / "bundle.json").read_text(encoding="utf-8"))
    assert "abc" not in bundle["reused_stages"]
    assert "tampered" not in (ev / "abc-mechanics.json").read_text(encoding="utf-8")


def test_missing_artifact_forces_rerun(tmp_path):
    target = tmp_path / "abc-mechanics.json"
    target.write_text("{}", encoding="utf-8")
    manifest: dict = {}
    mark_stage(tmp_path, manifest, "abc", ["abc-mechanics.json"])
    assert stage_verified(tmp_path, manifest, "abc", ["abc-mechanics.json"]) is True
    target.unlink()  # artifact lost after manifest: must re-run, never reuse
    assert stage_verified(tmp_path, manifest, "abc", ["abc-mechanics.json"]) is False


def test_partial_output_never_reused(tmp_path):
    (tmp_path / "abc-mechanics.json").write_text("{}", encoding="utf-8")
    manifest = {"stages": {"abc": {"status": "running", "artifacts": {}}}}
    assert stage_verified(tmp_path, manifest, "abc", ["abc-mechanics.json"]) is False
    manifest2: dict = {}
    mark_stage(tmp_path, manifest2, "abc", ["abc-mechanics.json"])
    assert stage_verified(tmp_path, manifest2, "abc", ["abc-mechanics.json"]) is True
    assert sha256_file(tmp_path / "abc-mechanics.json") == \
        manifest2["stages"]["abc"]["artifacts"]["abc-mechanics.json"]
