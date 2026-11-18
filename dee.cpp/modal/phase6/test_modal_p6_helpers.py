"""Local tests for the modal_p6 helpers that run before any GPU bills."""
import hashlib
import json
import os

import pytest

m = pytest.importorskip("modal_p6")

PROMPTS = "dee.cpp/aws/phase6/prompts_phase4_8.json"


def test_parse_configs_accepts_the_l4_slice():
    cfgs = json.dumps([
        {"tag": "b1-h64", "host_gib": 64, "n_tokens": 16},
        {"tag": "b1-h16", "host_gib": 16, "n_tokens": 16},
        {"tag": "b8-h64", "host_gib": 64, "n_tokens": 65, "ignore_eos": True,
         "cohort": "0-7", "prompts_file": PROMPTS},
    ])
    assert [c["tag"] for c in m._parse_configs(cfgs, 17.0, 96.0)] == ["b1-h64", "b1-h16", "b8-h64"]
    assert m._parse_configs("", 17.0, 96.0) == [None]


@pytest.mark.parametrize("bad", [
    [{"tag": "a", "hostgib": 16}],                                   # typo'd key
    [{"host_gib": 16}],                                              # no tag
    [{"tag": "a"}, {"tag": "a"}],                                    # duplicate tag
    [{"tag": "a", "host_gib": 90}],                                  # 90 + 12 > 96
    [{"tag": "a", "cohort": "0-7", "n_tokens": 65, "ignore_eos": True}],  # no prompts
    [{"tag": "a", "cohort": "0-7", "prompts_file": PROMPTS, "n_tokens": 16,
      "ignore_eos": True}],                                          # b=8 too short
    [{"tag": "a", "cohort": "0-7", "prompts_file": PROMPTS, "n_tokens": 65}],  # EOS can stop
])
def test_parse_configs_fails_closed(bad):
    with pytest.raises((ValueError, RuntimeError)):
        m._parse_configs(json.dumps(bad), 17.0, 96.0)


def _store(tmp_path):
    src = tmp_path / "src"
    (src / "seg").mkdir(parents=True)
    segs = []
    for i in range(3):
        b = os.urandom(3_000_000 + i)
        (src / "seg" / f"s{i}.bin").write_bytes(b)
        segs.append({"file": f"seg/s{i}.bin", "bytes": len(b),
                     "sha256": hashlib.sha256(b).hexdigest()})
    (src / "metadata.json").write_text(json.dumps({"segments": segs}))
    return src, {"segments": segs}


def test_stage_and_seal_copies_and_verifies(tmp_path):
    src, meta = _store(tmp_path)
    seal = m._stage_and_seal(src, tmp_path / "dst", meta, lambda *_: None)
    assert seal["verified"] and seal["segments"] == 3
    for sg in meta["segments"]:
        assert (tmp_path / "dst" / sg["file"]).read_bytes() == (src / sg["file"]).read_bytes()
    assert (tmp_path / "dst" / "metadata.json").is_file()


def test_stage_and_seal_refuses_a_bad_segment(tmp_path):
    src, meta = _store(tmp_path)
    meta["segments"][1]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m._stage_and_seal(src, tmp_path / "dst", meta, lambda *_: None)


def test_stage_and_seal_refuses_a_missing_segment(tmp_path):
    src, meta = _store(tmp_path)
    (src / "seg" / "s2.bin").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        m._stage_and_seal(src, tmp_path / "dst", meta, lambda *_: None)


def test_drop_page_cache_reports_meminfo(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 4096)
    out = m._drop_page_cache([f, tmp_path / "absent"], lambda *_: None)
    assert set(out) == {"cached_gib_before", "cached_gib_after"}


def test_vram_fit_refuses_dsv4_on_one_t4():
    with pytest.raises(RuntimeError, match="usable"):
        m._vram_fit("T4", "dsv4-flash")          # p6m-t4-r1 OOM


@pytest.mark.parametrize("gpu", ["2xT4", "L4", "2xL4", "A10", "L40S", "RTXPRO6000"])
def test_vram_fit_allows_the_cells_that_ran_or_fit(gpu):
    assert "GiB per GPU" in m._vram_fit(gpu, "dsv4-flash")


def test_vram_fit_refuses_an_oversized_budget_override():
    with pytest.raises(RuntimeError):
        m._vram_fit("L40S", "dsv4-flash", budget_mib=28672)


def test_placement_knobs_map_to_runner_env():
    cfg = {"tag": "b8-h16-nu", "host_gib": 16, "n_tokens": 65, "ignore_eos": True,
           "cohort": "0-7", "prompts_file": "p.json",
           "eviction_policy": "next_use", "host_eviction_policy": "next_use",
           "prefetch_hint_lanes": 2, "prefetch_hint_source": "prev_step",
           "prefetch_lookahead": "router"}
    assert m._parse_configs(json.dumps([cfg]), 17.0, 96.0)[0]["tag"] == "b8-h16-nu"
    assert m._placement_env(cfg) == {
        "NATIVE_EVICTION_POLICY": "next_use",
        "NATIVE_HOST_EVICTION_POLICY": "next_use",
        "NATIVE_PREFETCH_HINT_LANES": "2",
        "NATIVE_PREFETCH_HINT_SOURCE": "prev_step",
        "NATIVE_PREFETCH_LOOKAHEAD": "router",
    }
    assert m._placement_env(None) == {}
    assert m._placement_env({"tag": "plain"}) == {}


@pytest.mark.parametrize("bad", [
    {"eviction_policy": "belady"},
    {"host_eviction_policy": "rank_priority"},
    {"prefetch_hint_lanes": 9},
    {"prefetch_hint_lanes": "2"},
    {"prefetch_hint_source": "prev_step"},                 # hints with no lanes
    {"prefetch_lookahead": "router", "prefetch_hint_lanes": 0},
    {"prefetch_queue_order": "lifo", "prefetch_hint_lanes": 1},
])
def test_placement_knobs_fail_closed(bad):
    with pytest.raises(ValueError):
        m._parse_configs(json.dumps([{"tag": "x", **bad}]), 17.0, 96.0)


def test_parse_verdict_single_prompt_and_cohort():
    single = ["[01:02:03] === VERDICT: ACCEPT_CORRECTNESS; performance_eligible=True ==="]
    assert m.parse_verdict(single) == "ACCEPT_CORRECTNESS; performance_eligible=True"
    # r6 logged this shape and the old parser reported UNKNOWN for every row
    cohort = ["[02:00:00] === VERDICT cohort 0: ACCEPT_CORRECTNESS ==="]
    assert m.parse_verdict(cohort) == "cohort 0: ACCEPT_CORRECTNESS"
    two = ["=== VERDICT cohort 1: REJECT_NUMERICAL ===", "x", "=== VERDICT cohort 0: ACCEPT_CORRECTNESS ==="]
    assert m.parse_verdict(two) == "cohort 0: ACCEPT_CORRECTNESS; cohort 1: REJECT_NUMERICAL"
    assert m.parse_verdict(["no verdict here"]) == "UNKNOWN"
