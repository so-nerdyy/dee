from __future__ import annotations

import importlib.util
from pathlib import Path


RUNTIME_PATH = (Path(__file__).parents[1] / "kaggle" /
                "deepseek-v4-flash-0731" / "deepseek_v4_model_runtime.py")
SPEC = importlib.util.spec_from_file_location("deepseek_v4_model_runtime", RUNTIME_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def _host(current: int = 4_000_000_000,
          peak: int = 7_000_000_000) -> dict[str, int]:
    return {"current_rss_bytes": current, "peak_rss_bytes": peak,
            "ceiling_bytes": runtime.HOST_RSS_CEILING_BYTES}


def test_generation_memory_helpers_fail_closed_on_host_and_gpu_overage() -> None:
    memory = {"cuda0_reserved_gib": 6.0, "cuda1_reserved_gib": 6.0,
              "host_memory": _host()}
    assert runtime._build_memory_within_ceiling(memory)
    memory["host_memory"] = _host(current=13_977_284_608,
                                   peak=17_153_708_032)
    assert not runtime._build_memory_within_ceiling(memory)
    assert not runtime._gpu_peaks_within_ceiling(
        {"cuda0": 8.0, "cuda1": 14.0})


def test_generation_memory_helpers_require_complete_snapshots() -> None:
    assert not runtime._host_memory_within_ceiling({})
    assert not runtime._build_memory_within_ceiling(None)
    assert not runtime._gpu_peaks_within_ceiling({"cuda0": 8.0})
