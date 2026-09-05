#!/usr/bin/env python3
"""Tests for experiment/pack-cap-ab tooling.

Coverage:
  - contract lint: pre-registered fields, thresholds, frozen provenance
  - make_harness: base sha pin, arm B single-line diff, driver embedding
  - session_driver (synthetic): /tmp cleanup, WORK reset between arms,
    memory gates on the REAL memory.json schema, cap-check parsing,
    harness-always-exits-0 handled via classification not exit codes
  - classify_experiment: decision rule on synthetic evidence (supported,
    null, order-confounded, missing-arm invalid)
No sealed evidence is modified; no network access is required.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TOOLS = PKG / "tools"
REPO_ROOT = PKG.parent.parent          # .../dee-pack-ab
BASE = PKG / "base" / "dee-cpp-dsv4-host-reuse-217-ab-20260904.py"
CONTRACT = PKG / "experiment-contract.json"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver_mod():
    return load_module(TOOLS / "session_driver.py", "session_driver")


@pytest.fixture(scope="module")
def classify_mod():
    return load_module(TOOLS / "classify_experiment.py", "classify_experiment")


# ------------------------------------------------------------- contract ----

def test_contract_frozen_fields():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["preregistered"] is True
    assert c["sessions"]["session1"]["order"] == ["A", "B"]
    assert c["sessions"]["session2"]["order"] == ["B", "A"]
    dr = c["decision_rule"]
    assert dr["memory"]["abort_thresholds"][
        "min_checkpoint_mem_available_gib_below"] == 1.5
    assert dr["memory"]["abort_thresholds"]["process_vmhwm_gib_above"] == 30.0
    assert dr["performance"]["no_early_accept_after_pair_1"] is True
    assert dr["performance"]["meaningful_noise_threshold_s"] == 0.905
    assert c["arms"]["A"]["LRU_TOTAL_CAP_GIB"] == 17.0
    assert c["arms"]["B"]["LRU_TOTAL_CAP_GIB"] == 20.0
    assert set(c["decision_rule"]["final_classification_enum"]) == {
        "PACK_CAP_SUPPORTED", "PACK_CAP_NULL", "PACK_CAP_ORDER_CONFOUNDED",
        "PACK_CAP_MEMORY_REJECTED", "INVALID_EXPERIMENT", "BLOCKED_LIVE_GPU"}


def test_contract_base_harness_present_and_pinned():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    want = c["canonical_baseline"]["base_harness_sha256"]
    got = hashlib.sha256(BASE.read_bytes()).hexdigest()
    assert got == want, "base harness drifted from the pinned sha256"


# ---------------------------------------------------------- make_harness ---

def test_make_harness_builds_packages(tmp_path):
    r = subprocess.run(
        [sys.executable, str(TOOLS / "make_harness.py"),
         "--s1-dir", tmp_path / "s1", "--s2-dir", tmp_path / "s2"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for s in ("s1", "s2"):
        assert (tmp_path / s / "kernel-metadata.json").is_file()
        assert (tmp_path / s / "session-driver.py").is_file()
    meta = json.loads((tmp_path / "s1" / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == "true"
    assert meta["dataset_sources"] == ["nivind/deepseek-v4-flash-0731-shards"]
    prov = json.loads((PKG / "harness-provenance.json").read_text())
    assert prov["github_force_push_repair"]["checkout_pin_unchanged"] is True
    assert len(prov["arm_A"]["modifications"]) == 1
    assert "repair bundle" in prov["arm_A"]["modifications"][0]
    assert len(prov["arm_B"]["modifications"]) == 2
    assert any("17.0 -> 20.0" in m for m in prov["arm_B"]["modifications"])


def test_arm_b_diff_is_exactly_one_line():
    """Arm B vs arm A: the ONLY difference is the cap constant (both share
    the force-push repair bundle)."""
    r = subprocess.run(
        [sys.executable, str(TOOLS / "make_harness.py"),
         "--s1-dir", tmp_dir(), "--s2-dir", tmp_dir()],
        capture_output=True, text=True)
    assert r.returncode == 0
    a = (PKG / "harness_arm_a_repaired.py").read_text(encoding="utf-8")
    b = (PKG / "harness_arm_b_cap20.py").read_text(encoding="utf-8")
    diffs = [(x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x != y]
    assert len(diffs) == 1
    old, new = diffs[0]
    assert "LRU_TOTAL_CAP_GIB = 17.0" in old
    assert "LRU_TOTAL_CAP_GIB = 20.0" in new


def test_both_arms_carry_repair_bundle():
    """Both arms must embed the repair bundle sha (GitHub force-push repair)
    and keep the checkout pin + fetch refspec untouched."""
    r = subprocess.run(
        [sys.executable, str(TOOLS / "make_harness.py"),
         "--s1-dir", tmp_dir(), "--s2-dir", tmp_dir()],
        capture_output=True, text=True)
    assert r.returncode == 0
    for f in ("harness_arm_a_repaired.py", "harness_arm_b_cap20.py"):
        t = (PKG / f).read_text(encoding="utf-8")
        assert 'SOURCE_BUNDLE_SHA256 = "76e1b437' in t
        assert 'COMMIT = "217a33359b06a0453444a698ec52e4078b77e388"' in t
        assert 'refs/heads/codex/dee4-bounded-fill-storage' in t
        assert 'SOURCE_BUNDLE_SHA256 = "ac2bac46' not in t
    a = (PKG / "harness_arm_a_repaired.py").read_text(encoding="utf-8")
    assert "LRU_TOTAL_CAP_GIB = 17.0" in a


def tmp_dir():
    import tempfile
    return Path(tempfile.mkdtemp())


def test_driver_placeholder_substitution_complete(tmp_path):
    """Session 1: PREAD_ENABLE must be 0; session 2: 1 (rider after all arms)."""
    subprocess.run([sys.executable, str(TOOLS / "make_harness.py"),
                    "--s1-dir", tmp_path / "s1", "--s2-dir", tmp_path / "s2"],
                   capture_output=True, text=True, check=True)
    s1 = (tmp_path / "s1" / "session-driver.py").read_text(encoding="utf-8")
    s2 = (tmp_path / "s2" / "session-driver.py").read_text(encoding="utf-8")
    # The template placeholders must be gone. (A literal '@@' survives by
    # design inside the rider's sanity check against an unembedded rider.)
    for name, text in (("s1", s1), ("s2", s2)):
        for ph in ("@@ARM_A_B64@@", "@@ARM_A_SHA256@@", "@@ARM_B_B64@@",
                   "@@ARM_B_SHA256@@", "@@SESSION_ID@@", "@@ARM_ORDER@@",
                   "@@PREAD_B64@@", "@@PREAD_SHA256@@", "@@PREAD_ENABLE@@"):
            assert ph not in text, f"{name}: {ph} unsubstituted"
        assert len(text) > 200_000, f"{name}: driver suspiciously small " \
            f"({len(text)} bytes) - harnesses not embedded?"
    assert 'PREAD_ENABLE = "0"' in s1
    assert 'PREAD_ENABLE = "1"' in s2
    assert 'SESSION_ID = "session1"' in s1
    assert 'SESSION_ID = "session2"' in s2
    assert 'ARM_ORDER = "A,B"' in s1
    assert 'ARM_ORDER = "B,A"' in s2


# ------------------------------------------------------- session_driver ----

def test_verify_embedded_rejects_bad_sha(driver_mod):
    m = driver_mod
    m.ARM_A_B64 = base64.b64encode(b"fake harness").decode()
    m.ARM_A_SHA256 = "0" * 64          # valid b64, wrong sha -> must refuse
    with pytest.raises(SystemExit):
        m.verify_embedded()


def test_between_arms_cleanup_keeps_trace_bank(driver_mod, tmp_path, monkeypatch):
    m = driver_mod
    # fake harness tmp paths + trace bank under a temp root
    fake = {}
    for i, p in enumerate(m.HARNESS_TMP_PATHS):
        pp = tmp_path / f"t{i}"
        pp.mkdir()
        (pp / "junk").write_text("x")
        fake[p] = pp
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "metadata.json").write_text("{}")
    monkeypatch.setattr(m, "HARNESS_TMP_PATHS", [str(p) for p in fake.values()])
    monkeypatch.setattr(m, "TRACE_BANK", str(bank))
    info = m.clean_between_arms()
    assert info["removed"] == [str(p) for p in fake.values()]
    for p in fake.values():
        assert not p.exists()
    assert (bank / "metadata.json").is_file()


def test_memory_gate_real_schema(driver_mod):
    m = driver_mod
    # REAL schema from seal-era memory.json
    evidence = {"memory_gate_inputs": {
        "vmhwm_gib": 22.75, "vmrss_gib": 22.64,
        "min_checkpoint_mem_available_gib": 7.81}}
    g = m.memory_gate(evidence)
    assert g["triggered"] is False
    # trigger: min avail below 1.5
    evidence["memory_gate_inputs"]["min_checkpoint_mem_available_gib"] = 1.2
    g = m.memory_gate(evidence)
    assert g["triggered"] is True and "MemAvailable" in g["reason"]
    # trigger: HWM above 30
    evidence["memory_gate_inputs"] = {"vmhwm_gib": 30.5,
                                      "min_checkpoint_mem_available_gib": 7.8}
    g = m.memory_gate(evidence)
    assert g["triggered"] is True and "VmHWM" in g["reason"]
    # missing memory.json -> gate trips (fail closed)
    g = m.memory_gate({})
    assert g["triggered"] is True


def test_parse_arm_evidence_real_schema(driver_mod, tmp_path):
    m = driver_mod
    (tmp_path / "native-generate-result.json").write_text(json.dumps({
        "classification": "ACCEPT_CORRECTNESS", "run_id": "r1",
        "decode_wall_s": 70.691, "decode_tok_s": 0.212,
        "host_pack_budget_gib": [10.0, 10.0],
        "host_pack": {"cuda0": {"hits": 1290, "misses": 1338, "evictions": 656},
                      "cuda1": {"hits": 1441, "misses": 1046, "evictions": 363}},
        "engine_stats": {"cuda0": {"h2d_bytes": 100}, "cuda1": {"h2d_bytes": 200}},
        "byte_accounting": {"storage_bytes_total": 12345},
    }), encoding="utf-8")
    (tmp_path / "memory.json").write_text(json.dumps({
        "process_final_and_peak_gib": {"VmHWM": 23.4, "VmRSS": 23.1},
        "system_final_gib": {"MemTotal": 31.35},
        "minimum_checkpoint_host_mem_available_gib": 6.5,
        "host_pack_budget_bytes": [10737418240, 10737418240],
        "checkpoint_records": 16,
    }), encoding="utf-8")
    ev = m.parse_arm_evidence(tmp_path)
    assert ev["classification"] == "ACCEPT_CORRECTNESS"
    assert ev["decode_wall_s"] == 70.691
    assert ev["host_pack"]["cuda0"]["misses"] == 1338
    assert ev["memory_gate_inputs"]["vmhwm_gib"] == 23.4
    assert ev["memory_gate_inputs"]["min_checkpoint_mem_available_gib"] == 6.5


def test_cap_check_parses_budget_line(driver_mod, tmp_path):
    m = driver_mod
    log_line = ("budget=7.00GiB/GPU host_pack=10.00/10.00GiB "
                "batched=False profile=False diagnostics=False "
                "mem_avail=26.3GiB mem_total=31.3GiB lru_cap=20.0GiB "
                "cache_dtype=fp4 source_read_lanes=3")
    vals = __import__("re").findall(r"lru_cap=([0-9.]+)GiB", log_line)
    assert [float(v) for v in vals] == [20.0]
    packs = __import__("re").findall(r"host_pack=([0-9.]+)/([0-9.]+)GiB", log_line)
    assert [float(a) for a, _ in packs] == [10.0]


# ------------------------------------------------- synthetic end-to-end ----

def _make_harness_with() -> bytes:
    """Tiny fake harness mimicking the real contract: prints the budget line
    (cap per arm, parsed from its own script name), writes real-schema evidence
    to the driver's WORK dir (passed via FAKE_WORK env, like /kaggle/working),
    and ALWAYS exits 0 (like the real one)."""
    return (b"import json, os, sys\n"
            b"arm = 'A' if sys.argv[0].endswith('_A.py') else 'B'\n"
            b"tot = '17.0' if arm == 'A' else '20.0'\n"
            b"pg = '8.5' if arm == 'A' else '10.0'\n"
            b"print('budget=7.00GiB/GPU host_pack=' + pg + '/' + pg + 'GiB "
            b"lru_cap=' + tot + 'GiB')\n"
            b"work = os.environ.get('FAKE_WORK', '.')\n"
            b"open(os.path.join(work,'native-generate-result.json'),'w')"
            b".write(json.dumps({\n"
            b"  'classification': 'ACCEPT_CORRECTNESS', 'run_id': 'syn-'+arm,\n"
            b"  'decode_wall_s': 69.5,\n"
            b"  'host_pack': {'cuda0': {'hits': 1290, 'misses': 1338},\n"
            b"               'cuda1': {'hits': 1441, 'misses': 1046}},\n"
            b"}))\n"
            b"open(os.path.join(work,'memory.json'),'w').write(json.dumps({\n"
            b"  'process_final_and_peak_gib': {'VmHWM': 23.0},\n"
            b"  'minimum_checkpoint_host_mem_available_gib': 6.0,\n"
            b"  'host_pack_budget_bytes': [10737418240]*2,\n"
            b"}))\n"
            b"print('done'); sys.exit(0)\n")


def test_synthetic_session_end_to_end(tmp_path, monkeypatch):
    """Run the actual driver main() against a fake WORK + fake harness."""
    m = load_module(TOOLS / "session_driver.py", "session_driver_syn")
    harness = _make_harness_with()
    m.ARM_A_B64 = base64.b64encode(harness).decode()
    m.ARM_A_SHA256 = hashlib.sha256(harness).hexdigest()
    m.ARM_B_B64 = m.ARM_A_B64
    m.ARM_B_SHA256 = m.ARM_A_SHA256
    m.SESSION_ID = "session1"
    m.ARM_ORDER = "A,B"
    m.PREAD_B64 = ""
    m.PREAD_ENABLE = "0"
    m.PREAD_SHA256 = ""
    work = tmp_path / "working"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setenv("FAKE_WORK", str(work))
    # keep the cleanup away from real /tmp
    monkeypatch.setattr(m, "HARNESS_TMP_PATHS", [])
    rc = m.main()
    assert rc == 0
    for arm, tot, pg in (("A", 17.0, 8.5), ("B", 20.0, 10.0)):
        d = work / f"session1-{arm}"
        assert (d / "arm-summary.json").is_file()
        s = json.loads((d / "arm-summary.json").read_text())
        assert s["evidence"]["classification"] == "ACCEPT_CORRECTNESS"
        assert s["memory_gate"]["triggered"] is False
        assert s["cap_check"]["expected"] == {
            "lru_cap_total": tot, "per_gpu": pg}
        assert s["cap_check_ok"] is True, s["cap_check"]
    assert (work / "session1-summary.json").is_file()


def test_memory_abort_stops_session(tmp_path, monkeypatch):
    m = load_module(TOOLS / "session_driver.py", "session_driver_abort")
    harness = (b"import json\n"
               b"open('memory.json','w').write(json.dumps({\n"
               b"  'process_final_and_peak_gib': {'VmHWM': 31.0},\n"
               b"  'minimum_checkpoint_host_mem_available_gib': 0.8,\n"
               b"}))\n")
    m.ARM_A_B64 = base64.b64encode(harness).decode()
    m.ARM_A_SHA256 = hashlib.sha256(harness).hexdigest()
    m.ARM_B_B64 = m.ARM_A_B64
    m.ARM_B_SHA256 = m.ARM_A_SHA256
    m.SESSION_ID = "session1"
    m.ARM_ORDER = "A,B"
    m.PREAD_B64 = ""
    m.PREAD_ENABLE = "0"
    m.PREAD_SHA256 = ""
    work = tmp_path / "working"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "HARNESS_TMP_PATHS", [])
    rc = m.main()
    summary = json.loads((work / "session1-summary.json").read_text())
    assert summary["results"]["A"]["memory_gate"]["triggered"] is True
    assert "B" not in summary["results"], "arm B must not run after abort"
    # driver returns non-zero (no valid arms)
    assert rc != 0


# ---------------------------------------------------- classify_experiment --

def _seed_live_dir(root: Path, walls: dict) -> Path:
    """walls: {session: (wallA, wallB, missesA[2], missesB[2])}."""
    for sess, (aw, bw, am, bm) in walls.items():
        for arm, wall, miss in (("A", aw, am), ("B", bw, bm)):
            d = root / f"{sess}-{arm}"
            d.mkdir(parents=True)
            (d / "native-generate-result.json").write_text(json.dumps({
                "classification": "ACCEPT_CORRECTNESS",
                "decode_wall_s": wall,
                "host_pack": {"cuda0": {"misses": miss[0]},
                              "cuda1": {"misses": miss[1]}},
            }), encoding="utf-8")
            (d / "memory.json").write_text(json.dumps({
                "process_final_and_peak_gib": {"VmHWM": 23.0},
                "system_final_gib": {"MemTotal": 31.35},
                "minimum_checkpoint_host_mem_available_gib": 5.0,
            }), encoding="utf-8")
    return root


def test_classify_supported(tmp_path, classify_mod):
    live = _seed_live_dir(tmp_path / "live",
                          {"session1": (71.0, 68.5, [1390, 1091], [1338, 1046]),
                           "session2": (70.8, 68.2, [1390, 1091], [1337, 1046])},
                          )
    r = subprocess.run(
        [sys.executable, str(TOOLS / "classify_experiment.py"),
         "--live-dir", str(live), "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads((tmp_path / "out" / "combined-comparison.json").read_text())
    d = out["decision"]
    assert d["informative_positive"] is True
    assert d["order"]["confounded"] is False
    assert d["case"] == "CASE_3_supported"
    assert d["classification"] == "PACK_CAP_SUPPORTED"
    mv = json.loads((tmp_path / "out" / "miss-validation.json").read_text())
    # mean measured delta = mean(-97, -98) = -97.5 vs predicted -51
    assert mv["deltas"]["session1_measured_minus_predicted_misses"] == -46
    assert mv["deltas"]["session2_measured_minus_predicted_misses"] == -47


def test_classify_order_confounded(tmp_path, classify_mod):
    # session1 A->B: B wins. session2 B->A: A wins (the SECOND arm wins again).
    live = _seed_live_dir(tmp_path / "live",
                          {"session1": (71.0, 68.5, [1390, 1091], [1338, 1046]),
                           "session2": (68.5, 70.8, [1338, 1046], [1390, 1091])},
                          )
    subprocess.run([sys.executable, str(TOOLS / "classify_experiment.py"),
                    "--live-dir", str(live), "--out-dir", str(tmp_path / "out")],
                   capture_output=True, text=True, check=True)
    out = json.loads((tmp_path / "out" / "combined-comparison.json").read_text())
    d = out["decision"]
    assert d["order"]["confounded"] is True
    assert d["classification"] == "PACK_CAP_ORDER_CONFOUNDED"


def test_classify_null(tmp_path, classify_mod):
    live = _seed_live_dir(tmp_path / "live",
                          {"session1": (71.0, 70.9, [1390, 1091], [1388, 1090]),
                           "session2": (71.2, 71.1, [1390, 1091], [1389, 1090])},
                          )
    subprocess.run([sys.executable, str(TOOLS / "classify_experiment.py"),
                    "--live-dir", str(live), "--out-dir", str(tmp_path / "out")],
                   capture_output=True, text=True, check=True)
    out = json.loads((tmp_path / "out" / "combined-comparison.json").read_text())
    d = out["decision"]
    assert d["informative_positive"] is False
    assert d["classification"] == "PACK_CAP_NULL"


def test_classify_missing_arm_invalid(tmp_path, classify_mod):
    # only session1 present -> cannot decide
    live = _seed_live_dir(tmp_path / "live",
                          {"session1": (71.0, 68.5, [1390, 1091], [1338, 1046])},
                          )
    subprocess.run([sys.executable, str(TOOLS / "classify_experiment.py"),
                    "--live-dir", str(live), "--out-dir", str(tmp_path / "out")],
                   capture_output=True, text=True, check=True)
    out = json.loads((tmp_path / "out" / "combined-comparison.json").read_text())
    assert out["decision"]["classification"] == "INVALID_EXPERIMENT"


def test_classify_case1_replay_failed(tmp_path, classify_mod):
    # misses do NOT fall -> CASE_1 even though walls improve
    live = _seed_live_dir(tmp_path / "live",
                          {"session1": (71.0, 68.5, [1390, 1091], [1390, 1091]),
                           "session2": (70.8, 68.2, [1390, 1091], [1392, 1093])},
                          )
    subprocess.run([sys.executable, str(TOOLS / "classify_experiment.py"),
                    "--live-dir", str(live), "--out-dir", str(tmp_path / "out")],
                   capture_output=True, text=True, check=True)
    out = json.loads((tmp_path / "out" / "combined-comparison.json").read_text())
    d = out["decision"]
    assert d["case"] == "CASE_1_replay_failed"
    assert d["classification"] == "PACK_CAP_NULL"
