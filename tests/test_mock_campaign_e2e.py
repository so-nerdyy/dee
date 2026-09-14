#!/usr/bin/env python3
"""End-to-end mock campaign tests (Phase G/H).

Every scenario executes run_evidence.py against mock_campaign.py through the
EXACT live ingestion path (pack -> pair validation -> merge -> closure ->
ABC -> ranking -> next-ab). The only thing mocked is the data source, never
the aggregation logic. No CUDA required; no timings are real.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dee.cpp" / "experiments" / "route_pipeline"
MOCK = EXP / "mock_campaign.py"
RUN = EXP / "run_evidence.py"
TMP = Path("C:/Users/carth/AppData/Local/Temp/opencode/rpe2e")


def _run_scenario(number: int, extra: str = "") -> tuple[int, dict, Path]:
    work = TMP / f"s{number}"
    shutil.rmtree(work, ignore_errors=True)
    abc = work / "abc.json"
    proc = subprocess.run(
        [sys.executable, str(MOCK), "--out", str(work / "abcf"),
         "--scenario", str(number), "--emit-abc", str(abc)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    cmd = [sys.executable, str(RUN), "--out", str(work / "ev"),
           "--command", f"{sys.executable} {MOCK}",
           "--prompt-hash", "mock-prompt-hash",
           "--mock-abc", str(abc), "--no-gpu-check",
           "--extra-cmd-args", f"--scenario {number}{extra}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    bundle_path = work / "ev" / "bundle.json"
    assert bundle_path.is_file(), proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.returncode, json.loads(bundle_path.read_text(encoding="utf-8")), work


def _artifacts(work: Path) -> Path:
    return work / "ev"


def test_scenario_1_sync_promotes_event_handoff():
    code, bundle, work = _run_scenario(1)
    assert code == 0
    assert bundle["classification"] == "LIVE_RUN_OK"
    assert bundle["next"] == "EVENT_HANDOFF_PROMOTED"
    assert bundle["closure_status"] == "OK"
    ev = _artifacts(work)
    for name in ("host-sync-attribution.json", "per-layer.csv",
                 "per-token.json", "closure.json", "abc-mechanics.json",
                 "candidate-ranking.json", "next-ab.json", "bundle.json"):
        assert (ev / name).is_file(), name
    attribution = json.loads((ev / "host-sync-attribution.json").read_text(encoding="utf-8"))
    assert len(attribution["records"]) == 2 * 4  # tokens x layers
    rows = (ev / "per-layer.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 8  # header + records
    tokens = json.loads((ev / "per-token.json").read_text(encoding="utf-8"))["tokens"]
    assert len(tokens) == 2 and all(t["layers"] == 4 for t in tokens)
    closure = json.loads((ev / "closure.json").read_text(encoding="utf-8"))
    assert closure["closure_fraction"] >= 0.85
    assert closure["forced"] is False
    ranking = json.loads((ev / "candidate-ranking.json").read_text(encoding="utf-8"))
    assert ranking["candidates"]["EVENT_HANDOFF"]["rank"] == "PROMOTE"
    nxt = json.loads((ev / "next-ab.json").read_text(encoding="utf-8"))
    assert nxt["candidate"] == "EVENT_HANDOFF"
    assert "ONLY" in nxt["ab_shape"]["candidate"]


def test_scenario_2_shared_promoted():
    code, bundle, work = _run_scenario(2)
    assert code == 0 and bundle["next"] == "SHARED_OVERLAP_PROMOTED"


def test_scenario_3_d2h_promoted():
    code, bundle, work = _run_scenario(3)
    assert code == 0 and bundle["next"] == "ROUTE_D2H_NARROW_PROMOTED"


def test_scenario_4_no_optimization_justified():
    code, bundle, work = _run_scenario(4)
    assert code == 0
    assert bundle["next"] == "NO_OVERLAP_OPTIMIZATION_JUSTIFIED"
    assert bundle["closure_status"] == "OK"


def test_scenario_5_profile_incomplete():
    code, bundle, work = _run_scenario(5, extra=" --wall-scale 3.0")
    assert code == 0
    assert bundle["closure_status"] == "PROFILE_INCOMPLETE"
    assert bundle["next"] == "NO_OVERLAP_OPTIMIZATION_JUSTIFIED"
    closure = json.loads((work / "ev" / "closure.json").read_text(encoding="utf-8"))
    assert closure["closure_fraction"] < 0.85


def test_scenario_6_invalid_pair():
    code, bundle, work = _run_scenario(6)
    assert code == 1
    assert bundle["classification"] == "EVIDENCE_FAILURE"
    assert "INVALID_PROFILE_PAIR" in bundle["error"]


def test_scenario_7_invalid_evidence():
    code, bundle, work = _run_scenario(7)
    assert code == 1
    assert bundle["classification"] == "EVIDENCE_FAILURE"
    assert "INVALID_PROFILE_EVIDENCE" in bundle["error"]
