from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_milestone25_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("run_milestone25_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_default_capacity_preserves_explicit_diagnostic_override() -> None:
    assert MATRIX.default_capacity_flags(["--profile"], 32) == [
        "--cache-experts", "32",
    ]
    assert MATRIX.default_capacity_flags(
        ["--cache-experts", "4", "--allow-sub-topk-cache"], 32
    ) == []
    with pytest.raises(ValueError, match="top-k 8"):
        MATRIX.default_capacity_flags([], 4)


def test_run_tee_keeps_complete_trace_off_notebook_stdout(
    tmp_path: Path, capsys,
) -> None:
    log_path = tmp_path / "run.log"
    trace_path = tmp_path / "raw-allocation-trace.log"
    command = [
        sys.executable,
        "-c",
        (
            "print('ordinary output');"
            "print('[ta_alloc] id=1 ptr=0x1');"
            "print('[ta_free] id=1 ptr=0x1');"
            "print('[ta_free_DOUBLE_FREE_ABORT] ptr=0x1');"
            "print('Fatal Python error: Aborted')"
        ),
    ]

    MATRIX.run_tee(command, log_path, trace_path=trace_path)

    stdout = capsys.readouterr().out
    assert "ordinary output" in stdout
    assert "[ta_alloc]" not in stdout
    assert "[ta_free] id=1" not in stdout
    assert "[ta_free_DOUBLE_FREE_ABORT]" in stdout
    assert "Fatal Python error: Aborted" in stdout

    log = log_path.read_text(encoding="utf-8")
    assert "ordinary output" in log
    assert "[ta_alloc] id=1" in log
    assert "[ta_free] id=1" in log
    assert "[ta_free_DOUBLE_FREE_ABORT]" in log
    assert "Fatal Python error: Aborted" in log

    trace = trace_path.read_text(encoding="utf-8")
    assert "ordinary output" not in trace
    assert "[ta_alloc] id=1" in trace
    assert "[ta_free] id=1" in trace
    assert "[ta_free_DOUBLE_FREE_ABORT]" in trace
    assert "Fatal Python error: Aborted" not in trace
