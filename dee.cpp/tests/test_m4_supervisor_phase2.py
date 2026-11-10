from __future__ import annotations

import importlib.util
from pathlib import Path


SUPERVISOR_PATH = (
    Path(__file__).resolve().parents[2] / "tmp" / "m4_supervisor.py"
)
SPEC = importlib.util.spec_from_file_location("m4_supervisor_phase2", SUPERVISOR_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


def test_no_seal_mode_succeeds_after_terminal_download(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state = {}

    def save_state(payload, merge=True):
        if not merge:
            state.clear()
        state.update(payload)

    monkeypatch.setattr(SUPERVISOR, "save_state", save_state)
    monkeypatch.setattr(SUPERVISOR, "load_state", lambda: dict(state))
    monkeypatch.setattr(
        SUPERVISOR,
        "resolve_integration_sha",
        lambda *_: "1" * 40,
    )
    monkeypatch.setattr(
        SUPERVISOR,
        "get_status",
        lambda *_: ("COMPLETE", "KernelWorkerStatus.COMPLETE"),
    )
    monkeypatch.setattr(SUPERVISOR, "download_artifacts", lambda *_: True)
    monkeypatch.setattr(SUPERVISOR, "find_evidence_root", lambda *_: tmp_path)
    monkeypatch.setattr(SUPERVISOR, "log", lambda *_: None)

    result = SUPERVISOR._main_unlocked([
        "--model-dir",
        str(tmp_path),
        "--evidence-dir",
        str(tmp_path / "download"),
        "--integration-sha",
        "1" * 40,
        "--integration-branch",
        "codex/phase2-cap32-matrix",
        "--run-id",
        "phase2-fixture",
        "--no-push",
        "--no-seal",
    ])

    assert result == 0
    assert state["complete"] is True
    assert state["download_ok"] is True
    assert state["seal_rc"] is None
