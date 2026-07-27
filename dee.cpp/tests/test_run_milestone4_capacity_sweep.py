import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_milestone4_capacity_sweep import (
    CAPACITIES,
    _validate_timing_completeness,
    capacity_experiments,
)


def test_capacity_sweep_changes_only_capacity_and_profiling():
    experiments = capacity_experiments(require_dual_gpu=True)

    assert [item["capacity"] for item in experiments] == [
        capacity for capacity in CAPACITIES for _ in range(2)
    ]
    assert [item["profiled"] for item in experiments] == [
        value for _ in CAPACITIES for value in (False, True)
    ]
    for experiment in experiments:
        flags = experiment["flags"]
        assert "--warmup-generation" in flags
        assert "--require-dual-gpu" in flags
        capacity_index = flags.index("--cache-experts")
        assert int(flags[capacity_index + 1]) == experiment["capacity"]
        assert ("--profile" in flags) is experiment["profiled"]
        assert ("--trace-requests" in flags) is experiment["profiled"]
        assert ("--profile-timeline" in flags) is experiment["profiled"]


def test_timing_completeness_fails_closed(tmp_path):
    timing = tmp_path / "timing-raw.json"
    timing.write_text(json.dumps({
        "profile_snapshots": [{
            "phase": "decode",
            "step": 1,
            "layers": [{
                "layer": 0,
                "profile": {
                    "operations": {
                        "timing_events_allocated": 1025,
                        "timing_events_dropped": 1,
                    }
                },
            }],
        }],
    }))

    with pytest.raises(RuntimeError, match="instrumentation is incomplete"):
        _validate_timing_completeness(timing, "capacity-8-profiled")
