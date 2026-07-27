import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_milestone4_capacity_sweep import (
    CAPACITIES,
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
