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
