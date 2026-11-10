"""Tests for score.py (Phase-6 scorer) and the b=8 harness additions.

Run from this directory:  python -m pytest -q test_score.py
The fake registry keeps most tests free of numpy/scipy; the last tests use the
real frozen predictor and skip when its dependencies are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import score
from bench_runner import build_runner_env

LABEL = "aws-L4-g6.2xlarge"
CELLS = {LABEL: ("1xL4", "g6.2xlarge", 0.9776, 3584, (16,)),
         "aws-L4-g6.8xlarge": ("1xL4", "g6.8xlarge", 2.0144, 3584, (64,))}


def fake_predict(label, b, host, bssd):
    # (nominal, lo, hi, (kill_lo, kill_hi)) per metric, shaped like predecl_aws.predict
    return {
        "decode_tps": (1.0, 0.7, 1.4, (0.5, 2.0)),
        "host_hit_rate": (0.51, 0.44, 0.75, (0.37, 0.80)),
        "cold_records_per_tok": (185.0, 82.0, 370.0, (61.0, 462.0)),
        "usd_per_1k_tok": (0.27, 0.20, 0.35, (0.10, 0.60)),
        "limiter": "storage",
    }


REG = score.Registry(cells=CELLS, predict=fake_predict, source="fake")


def _summary(**over):
    s = {"run_id": "r1", "model": "dsv4-flash", "gpu": "L4", "instance_type": "g6.2xlarge",
         "spot": False, "commit": "abc", "host_gib_requested": 16.0, "host_gib_effective": 16.0,
         "budget_mib": 3584, "host_budget_honored": True, "measured_b_ssd_gib_s": 1.9,
         "exit_code": 0, "est_cost_usd": 0.5}
    s.update(over)
    return s


def _b1_result(**over):
    r = {"status": "COMPLETE", "gpu_environment": {"single_gpu_mode": True, "gpu_count": 1},
         "generated_token_ids": list(range(16)), "decode_tokens": 15, "decode_wall_s": 15.0,
         "decode_tok_s": 1.0, "stage_profile": {"cuda0": {"fills": 1}},
         # cuda1 aliases cuda0 on one GPU: must not be double counted
         "host_pack": {"cuda0": {"hits": 51, "misses": 49}, "cuda1": {"hits": 51, "misses": 49}},
         "byte_accounting": {"storage_requests_per_generated_token": 150.0},
         "per_token_accounting": [{"phase": "prefill", "storage_requests": 900},
                                  {"phase": "decode", "storage_requests": 80},
                                  {"phase": "decode", "storage_requests": 100}]}
    r.update(over)
    return r


def _write(tmp: Path, summary: dict, results: dict[str, dict], checkpoint=None) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "summary.json").write_text(json.dumps(summary))
    for name, body in results.items():
        (tmp / name).write_text(json.dumps(body))
    if checkpoint is not None:
        (tmp / "generated_checkpoint-c0.jsonl").write_text(
            "\n".join(json.dumps(r) for r in checkpoint) + "\n")
    return tmp


def test_b1_pass_and_single_gpu_dedup(tmp_path):
    d = _write(tmp_path, _summary(), {"result.json": _b1_result()})
    card = score.score_run(d, REG)
    assert card["prerequisite_failures"] == []
    assert card["b"] == 1
    m = card["metrics"]
    assert m["host_hit_rate"]["measured"] == pytest.approx(0.51)      # cuda0 only
    assert m["cold_records_per_tok"]["measured"] == 150.0
    assert card["window"]["decode_only_cold_records_per_tok"] == 90.0
    assert m["usd_per_1k_tok"]["measured"] == pytest.approx(1000 * 0.9776 / 3600 / 1.0)
    assert all(v["grade"] == "PASS" for v in m.values())
    assert card["verdict"] == "PASS"


def test_p2_tightening_kills_high_hit_at_16gib(tmp_path):
    r = _b1_result(host_pack={"cuda0": {"hits": 65, "misses": 35}})
    card = score.score_run(_write(tmp_path, _summary(), {"result.json": r}), REG)
    hit = card["metrics"]["host_hit_rate"]
    assert hit["kill_hi"] == 0.57 and hit["kill_lo"] == 0.45
    assert hit["grade"] == "KILLED"          # inside the row range, outside P2
    assert card["verdict"] == "KILLED"


def test_weakened_between_range_and_kill(tmp_path):
    r = _b1_result(decode_tok_s=1.7)
    card = score.score_run(_write(tmp_path, _summary(), {"result.json": r}), REG)
    assert card["metrics"]["decode_tps"]["grade"] == "WEAKENED"
    assert card["verdict"] in ("WEAKENED", "KILLED")


@pytest.mark.parametrize("over, needle", [
    ({"host_budget_honored": False}, "host budget not honored"),
    ({"measured_b_ssd_gib_s": 0}, "no measured B_SSD"),
    ({"budget_mib": 4096}, "VRAM budget"),
    ({"host_gib_requested": 64.0, "host_gib_effective": 64.0}, "registered for host"),
    ({"instance_type": "g6.4xlarge"}, "not a registered AWS cell"),
    ({"exit_code": 1}, "did not complete"),
])
def test_prerequisites_make_run_untestable(tmp_path, over, needle):
    card = score.score_run(_write(tmp_path, _summary(**over), {"result.json": _b1_result()}), REG)
    assert card["verdict"] == "UNTESTABLE"
    assert any(needle in f for f in card["prerequisite_failures"])


def test_missing_stage_profile_and_short_run(tmp_path):
    r = _b1_result(stage_profile={}, generated_token_ids=list(range(4)))
    card = score.score_run(_write(tmp_path, _summary(), {"result.json": r}), REG)
    fails = " ".join(card["prerequisite_failures"])
    assert "stage profile" in fails and "generated 4 tokens" in fails


def test_ambiguous_results_untestable(tmp_path):
    d = _write(tmp_path, _summary(), {"result-q0.json": _b1_result(),
                                      "result-q1.json": _b1_result()})
    card = score.score_run(d, REG)
    assert card["verdict"] == "UNTESTABLE"
    assert "more than one runner result" in card["prerequisite_failures"][0]


def _cohort(steps=64, k=8, emitted=None, step_ms=1000.0):
    return {"status": "COMPLETE", "mode": "cohort", "cohort_k": k,
            "gpu_environment": {"single_gpu_mode": True, "gpu_count": 1},
            "n_forward_steps": steps + 1, "emitted_tokens": emitted if emitted is not None
            else k * (steps + 1), "decode_timings_ms": [5000.0] * (steps - 32) + [step_ms] * 32,
            "stage_profile": {"cuda0": {}} | {"x": 1},
            "host_pack": {"cuda0": {"hits": 10, "misses": 90}}}


def _checkpoint(steps=65, reads_per_step=800, hits_per_step=50, misses_per_step=50):
    return [{"step": i,
             "expert_store": {"cuda0": {"source_reads": 1000 + i * reads_per_step}},
             "host_pack": {"cuda0": {"hits": i * hits_per_step, "misses": i * misses_per_step}}}
            for i in range(steps)]


def test_b8_window_aggregate_tps_and_cold(tmp_path):
    d = _write(tmp_path, _summary(), {"result-c0.json": _cohort(),
                                      "result-c0-r0.json": {"row": 0}},   # row file ignored
               checkpoint=_checkpoint())
    card = score.score_run(d, REG)
    assert card["prerequisite_failures"] == []
    m = card["metrics"]
    assert m["decode_tps"]["measured"] == pytest.approx(8.0)      # 8 x 32 / 32 s
    assert m["cold_records_per_tok"]["measured"] == pytest.approx(800 * 32 / (8 * 32))
    assert m["host_hit_rate"]["measured"] == pytest.approx(0.5)   # window, not whole run
    assert card["window"]["from_step"] == 32 and card["window"]["to_step"] == 64
    assert card["b8_steady_gate"] == "OPEN"


def test_b8_ignores_running_placeholder_result(tmp_path):
    placeholder = {"status": "RUNNING", "run_id": "r1"}
    d = _write(tmp_path, _summary(), {"result.json": placeholder, "result-c0.json": _cohort()},
               checkpoint=_checkpoint())
    card = score.score_run(d, REG)
    assert card["result_file"] == "result-c0.json"
    assert card["prerequisite_failures"] == []


def test_b8_gate_closed_makes_tps_untestable(tmp_path):
    d = _write(tmp_path, _summary(), {"result-c0.json": _cohort()},
               checkpoint=_checkpoint(reads_per_step=8 * 500))      # 500 rec/tok > kill 462
    card = score.score_run(d, REG)
    assert card["b8_steady_gate"] == "CLOSED"
    assert card["metrics"]["decode_tps"]["grade"] == "UNTESTABLE"
    assert card["metrics"]["usd_per_1k_tok"]["grade"] == "UNTESTABLE"
    assert card["metrics"]["cold_records_per_tok"]["grade"] == "KILLED"


def test_b8_eos_and_short_runs_untestable(tmp_path):
    d = _write(tmp_path, _summary(), {"result-c0.json": _cohort(emitted=300)},
               checkpoint=_checkpoint())
    card = score.score_run(d, REG)
    assert any("--ignore-eos" in f for f in card["prerequisite_failures"])
    d2 = _write(tmp_path / "short", _summary(), {"result-c0.json": _cohort(steps=40)},
                checkpoint=_checkpoint(steps=41))
    assert any("decode steps" in f for f in score.score_run(d2, REG)["prerequisite_failures"])


def test_b8_without_checkpoint_scores_partial(tmp_path):
    card = score.score_run(_write(tmp_path, _summary(), {"result-c0.json": _cohort()}), REG)
    m = card["metrics"]
    assert m["cold_records_per_tok"]["grade"] == "UNTESTABLE"
    assert m["host_hit_rate"]["basis"].endswith("(no window counters)")
    assert card["b8_steady_gate"] == "CLOSED"


def test_render_md_and_ledger(tmp_path, monkeypatch):
    d = _write(tmp_path / "run", _summary(), {"result.json": _b1_result()})
    md = score.render_md(score.score_run(d, REG))
    assert "**Verdict: PASS**" in md and "| decode_tps |" in md
    monkeypatch.setattr(score, "load_registry", lambda: REG)
    ledger = tmp_path / "ledger.jsonl"
    assert score.main([str(d), "--write", "--ledger", str(ledger)]) == 0
    assert (d / "scorecard.md").is_file() and json.loads((d / "scorecard.json").read_text())
    row = json.loads(ledger.read_text().splitlines()[0])
    assert row["verdict"] == "PASS" and row["grades"]["decode_tps"] == "PASS"


# --- harness additions ------------------------------------------------------

def test_ignore_eos_reaches_runner_env():
    job = {"run_id": "t", "gpu": "L4", "visible_gpus": 1, "budget_mib": 3584,
           "n_tokens": 65, "cache_reset": "cold", "ignore_eos": True}
    env = build_runner_env(job, base_env={}, store_path="s", ckpt_dir="c", host_gib=16.0,
                           headers_dir="h", build_dir="b", src_root="r")
    assert env["NATIVE_IGNORE_EOS"] == "1"
    job["ignore_eos"] = False
    env = build_runner_env(job, base_env={}, store_path="s", ckpt_dir="c", host_gib=16.0,
                           headers_dir="h", build_dir="b", src_root="r")
    assert "NATIVE_IGNORE_EOS" not in env


def test_prompts_file_is_the_phase4_workload():
    prompts = json.loads((Path(__file__).parent / "prompts_phase4_8.json").read_text("utf-8"))
    assert len(prompts) == 8 and prompts[7] == prompts[0]
    assert all(p.startswith("<｜begin▁of▁sentence｜>")
               and p.endswith("<｜Assistant｜>") for p in prompts)


# --- real frozen predictor ---------------------------------------------------

def _real_registry():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    return score.load_registry()


def test_real_registry_matches_addendum2_sample():
    reg = _real_registry()
    p = reg.predict(LABEL, 1, 16, 2.0e9)
    assert p["decode_tps"][0] == pytest.approx(0.912, abs=1e-3)       # Addendum 2 sample
    assert p["cold_records_per_tok"][0] == pytest.approx(185.0, abs=0.1)
    p8 = reg.predict(LABEL, 8, 16, 2.0e9)
    assert p8["decode_tps"][0] == pytest.approx(2.522, abs=1e-3)
    assert set(reg.cells) >= {LABEL, "aws-L40S-g6e.16xlarge"}


def test_real_registry_end_to_end(tmp_path):
    reg = _real_registry()
    d = _write(tmp_path, _summary(measured_b_ssd_gib_s=2.0e9 / (1 << 30)),
               {"result.json": _b1_result(decode_tok_s=0.9)})
    card = score.score_run(d, reg)
    assert card["metrics"]["decode_tps"]["nominal"] == pytest.approx(0.912, abs=1e-3)
    assert card["metrics"]["decode_tps"]["grade"] == "PASS"
    json.dumps(card, default=float)   # numpy scalars must serialize


# --- Modal runs vs the frozen section-1 rows (stdlib only) ---------------------

def _modal_summary(**over):
    s = {"run_id": "m1", "platform": "modal", "model": "dsv4-flash", "gpu": "L4",
         "gpu_price_s": 0.000222, "n_gpus": 1, "commit": "abc",
         "host_gib_requested": 64.0, "host_gib_effective": 64.0, "host_basis": "TOTAL",
         "budget_mib": 3584, "host_budget_honored": True,
         "measured_b_ssd_gib_s": 2.5e9 / (1 << 30),          # exactly the 1xL4 spec
         "ssd_probe": {"direct": True, "on_volume": False},
         "exit_code": 0, "est_gpu_cost_usd": 0.4}
    s.update(over)
    return s


@pytest.fixture(scope="module")
def frozen():
    return score.load_frozen()


def test_modal_b1_frozen_row_with_p1_and_receipt(tmp_path, frozen):
    # 16 tokens billed at 0.0012 USD -> 0.075 $/1k, inside the frozen [0.0185, 0.0844]
    d = _write(tmp_path, _modal_summary(billed_usd=0.0012),
               {"result.json": _b1_result(decode_tok_s=1.5,
                                          host_pack={"cuda0": {"hits": 54, "misses": 46}})})
    card = score.score_run(d, frozen=frozen)
    assert card["platform"] == "modal" and card["cell"] == "1xL4"
    assert card["prerequisite_failures"] == []
    m = card["metrics"]
    assert (m["decode_tps"]["kill_lo"], m["decode_tps"]["kill_hi"]) == (0.9, 2.1)   # P1
    assert m["host_hit_rate"]["kill_hi"] == 1.0          # P2 band only at host 16
    assert m["usd_per_1k_tok"]["measured"] == pytest.approx(0.075)
    assert all(v["grade"] == "PASS" for v in m.values()), m
    assert card["b_ssd_vs_spec"] == pytest.approx(1.0)
    assert any("VRAM expert budget 3584 MiB" in w for w in card["warnings"])
    assert card["verdict"] == "PASS"


def test_modal_bssd_outside_band_untests_tps_and_cost(tmp_path, frozen):
    d = _write(tmp_path, _modal_summary(measured_b_ssd_gib_s=1.0),
               {"result.json": _b1_result(decode_tok_s=1.5)})
    card = score.score_run(d, frozen=frozen)
    m = card["metrics"]
    assert m["decode_tps"]["grade"] == "UNTESTABLE"
    assert m["usd_per_1k_tok"]["grade"] == "UNTESTABLE"
    assert m["cold_records_per_tok"]["grade"] == "PASS"
    assert any("no billing receipt" in w for w in card["warnings"])
    assert card["verdict"] == "PARTIAL"


def test_modal_p1_kill_inside_row_range(tmp_path, frozen):
    d = _write(tmp_path, _modal_summary(), {"result.json": _b1_result(decode_tok_s=2.15)})
    card = score.score_run(d, frozen=frozen)
    assert card["metrics"]["decode_tps"]["grade"] == "KILLED"   # row hi 2.22, P1 kill 2.1


def test_modal_p14_and_p15b_bands_on_l40s_b8(tmp_path, frozen):
    s = _modal_summary(gpu="L40S", host_gib_requested=256.0, host_gib_effective=256.0,
                       budget_mib=28672, measured_b_ssd_gib_s=6.0e9 / (1 << 30))
    d = _write(tmp_path, s, {"result-c0.json": _cohort(step_ms=800.0)},
               checkpoint=_checkpoint(reads_per_step=8 * 100))
    card = score.score_run(d, frozen=frozen)
    m = card["metrics"]
    assert card["b"] == 8 and card["prerequisite_failures"] == []
    assert m["decode_tps"]["measured"] == pytest.approx(10.0)
    assert m["decode_tps"]["kill_hi"] == 28.0                          # P14 TPS cross-check
    assert m["usd_per_1k_tok"]["kill_hi"] == pytest.approx(min(0.20, 0.95 * 0.12892))
    assert "dense_ratio" in card and "b8_steady_gate" not in card
    assert any("units defect" in w for w in card["warnings"])


@pytest.mark.parametrize("over, needle", [
    ({"gpu": "H100"}, "not a frozen cell"),
    ({"host_gib_requested": 17.0, "host_gib_effective": 17.0}, "not a frozen budget"),
    ({"host_budget_honored": False}, "host budget not honored"),
    ({"measured_b_ssd_gib_s": None}, "no measured B_SSD"),
])
def test_modal_prerequisites(tmp_path, frozen, over, needle):
    d = _write(tmp_path, _modal_summary(**over), {"result.json": _b1_result()})
    card = score.score_run(d, frozen=frozen)
    assert card["verdict"] == "UNTESTABLE"
    assert any(needle in f for f in card["prerequisite_failures"])


def test_modal_cli_uses_frozen_without_numpy(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("AWS registry must not load for a Modal run")
    monkeypatch.setattr(score, "load_registry", boom)
    d = _write(tmp_path / "m", _modal_summary(), {"result.json": _b1_result(decode_tok_s=1.5)})
    assert score.main([str(d), "--write"]) == 0
    assert "Measured B_SSD / cell spec" in (d / "scorecard.md").read_text()


# --- Addendum 3: Modal rows at measured B_SSD (needs numpy + scipy) -------------

def _modal_addendum():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    return score.load_modal_addendum()


def test_addendum3_self_check_reproduces_frozen_b1_rows():
    pm = _modal_addendum()
    chk = pm.self_check()
    assert chk["pass"] and chk["rows_checked"] == 120          # 5 GPUs x 2 models x 3 hosts x 4


def test_addendum3_scored_beside_frozen_rows(tmp_path, frozen):
    pm = _modal_addendum()
    bssd = 1.6 * 1e9                                            # far outside the 2.5 GB/s spec
    p = pm.predict("L4", "dsv4_flash", 1, 64, bssd)
    s = _modal_summary(measured_b_ssd_gib_s=bssd / (1 << 30), container_mib=98304)
    d = _write(tmp_path, s, {"result.json": _b1_result(
        decode_tok_s=p["decode_tps"][0], host_pack={"cuda0": {"hits": 54, "misses": 46}})})
    card = score.score_run(d, frozen=frozen, addendum=pm)
    assert card["metrics"]["decode_tps"]["grade"] == "UNTESTABLE"     # frozen: B_SSD gate
    a3 = card["addendum3"]
    assert a3["prerequisite_failures"] == []
    assert a3["metrics"]["decode_tps"]["grade"] == "PASS"             # addendum: measured B_SSD
    assert a3["rate_s"] == pytest.approx(pm.rate_s("L4", 64))
    assert "Addendum 3 (Modal, measured B_SSD)" in score.render_md(card)


def test_addendum3_host256_needs_the_big_container(tmp_path, frozen):
    pm = _modal_addendum()
    s = _modal_summary(host_gib_requested=256.0, host_gib_effective=256.0, container_mib=98304)
    card = score.score_run(_write(tmp_path, s, {"result.json": _b1_result()}),
                           frozen=frozen, addendum=pm)
    assert card["addendum3"]["verdict"] == "UNTESTABLE"
    assert "bench_<GPU>_h256" in card["addendum3"]["prerequisite_failures"][0]


def test_suspect_bssd_untests_addendum3_and_warns(tmp_path, frozen):
    # run p6m-l4-h64-b1-r3: fio read page cache at 53.8 GiB/s
    pm = _modal_addendum()
    s = _modal_summary(measured_b_ssd_gib_s=53.84, ssd_probe_suspect=True, container_mib=98304)
    card = score.score_run(_write(tmp_path, s, {"result.json": _b1_result()}),
                           frozen=frozen, addendum=pm)
    assert card["addendum3"]["verdict"] == "UNTESTABLE"
    assert "implausible" in card["addendum3"]["prerequisite_failures"][0]
    assert any("plausible NVMe" in w for w in card["warnings"])


def test_rejected_correctness_is_untestable(tmp_path, frozen):
    # run p6m-l4-h64-b1-r4: 16 tokens on L4 diverged from the sealed T4 tokens
    res = _b1_result(classification="REJECT_NUMERICAL",
                     sealed_contract_gates={"exact_16_token_ids": False,
                                            "exact_decoded_text": False,
                            "all_43_layers": True, "required_performance_hardware": False})
    card = score.score_run(_write(tmp_path, _modal_summary(), {"result.json": res}),
                           frozen=frozen)
    assert card["verdict"] == "UNTESTABLE"
    f = [x for x in card["prerequisite_failures"] if "correctness gate" in x]
    assert f and "exact_16_token_ids" in f[0] and "required_performance" not in f[0]


def test_later_rows_of_a_container_warn_about_warm_cache(tmp_path, frozen):
    card = score.score_run(_write(tmp_path, _modal_summary(config_index=1),
                                  {"result.json": _b1_result()}), frozen=frozen)
    assert any("warmed the page cache" in w for w in card["warnings"])


def test_runner_fatal_is_untestable_despite_exit_0(tmp_path, frozen):
    s = _modal_summary(runner_fatal=True, verdict="RUNNER_FATAL: RuntimeError: x")
    card = score.score_run(_write(tmp_path, s, {"result.json": _b1_result()}), frozen=frozen)
    assert card["verdict"] == "UNTESTABLE"
    assert any("FATAL" in f for f in card["prerequisite_failures"])


def test_addendum3_mimo_b8_scales_from_dsv4():
    pm = _modal_addendum()
    d8 = pm.predict("L40S", "dsv4_flash", 8, 64, 3e9)
    m8 = pm.predict("L40S", "mimo_v2_flash", 8, 64, 3e9)
    f = 288 / 258
    assert m8["decode_tps"][0] == pytest.approx(d8["decode_tps"][0] / f)
    assert m8["cold_records_per_tok"][0] == pytest.approx(d8["cold_records_per_tok"][0] * f)


def test_addendum3_missing_deps_reported(tmp_path, frozen, monkeypatch):
    def nope():
        raise ImportError("no numpy")
    monkeypatch.setattr(score, "load_modal_addendum", nope)
    card = score.score_run(_write(tmp_path, _modal_summary(), {"result.json": _b1_result()}),
                           frozen=frozen)
    assert card["addendum3"]["verdict"] == "NOT SCORED"


# --- b=1 own-reference gate (Addendum 4) -----------------------------------------

def _reject_result(**over):
    gates = {"exact_16_token_ids": False, "exact_decoded_text": False, "all_43_layers": True,
             "required_performance_hardware": False}
    return _b1_result(classification="REJECT_NUMERICAL", sealed_contract_gates=gates, **over)


def _pair(tmp_path, journal_b=None, host_b=16.0, result_b=None):
    j = [{"forward_step": 0, "layer": 0, "run_id": "A", "chain_sha256": "x",
          "expert_ids_rank_order": [[1, 2]]}]
    ja = "\n".join(json.dumps(r) for r in j) + "\n"
    jb = ja if journal_b is None else journal_b
    dirs = []
    for name, host, journal, res in (("a", 64.0, ja, _reject_result()),
                                     ("b", host_b, jb, result_b or _reject_result())):
        d = _write(tmp_path / name, _modal_summary(run_id=name, host_gib_requested=host,
                                                   host_gib_effective=host), {"result.json": res})
        (d / "routed_experts.jsonl").write_text(journal)
        (d / "route_weights.jsonl").write_text(journal)
        dirs.append(d)
    return dirs


def test_own_reference_default_still_rejects(tmp_path, frozen):
    a, _ = _pair(tmp_path)
    card = score.score_run(a, frozen=frozen)
    assert any("REJECT_NUMERICAL" in f for f in card["prerequisite_failures"])


def test_own_reference_pair_waives_only_the_two_exact_gates(tmp_path, frozen):
    a, b = _pair(tmp_path)
    ea, eb = score.load_evidence(a), score.load_evidence(b)
    assert score.verify_own_reference(ea, eb) == []
    own = {"peer_run_id": "b", "waived_gates": sorted(score.OWN_REFERENCE_WAIVABLE)}
    card = score.score_run(a, frozen=frozen, own_reference=own)
    assert not any("correctness gate" in f for f in card["prerequisite_failures"])
    assert any("OWN-REFERENCE" in w for w in card["warnings"])
    # a different failed gate is NOT waived
    bad = _reject_result()
    bad["sealed_contract_gates"]["all_43_layers"] = False
    (a / "result.json").write_text(json.dumps(bad))
    card = score.score_run(a, frozen=frozen, own_reference=own)
    assert any("all_43_layers" in f for f in card["prerequisite_failures"])


@pytest.mark.parametrize("kw,needle", [
    ({"host_b": 64.0}, "same host budget"),
    ({"journal_b": '{"forward_step": 0, "layer": 0, "expert_ids_rank_order": [[1, 3]]}\n'},
     "routed_experts.jsonl differs"),
    ({"result_b": _reject_result(generated_token_ids=list(range(1, 17)))}, "token ids differ"),
])
def test_own_reference_pair_must_be_bit_identical(tmp_path, kw, needle):
    a, b = _pair(tmp_path, **kw)
    why = score.verify_own_reference(score.load_evidence(a), score.load_evidence(b))
    assert any(needle in w for w in why), why
