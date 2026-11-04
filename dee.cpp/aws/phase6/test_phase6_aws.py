"""Unit tests for the Phase-6 AWS harness pure logic (no AWS calls).

Run from this directory:  python -m pytest -q test_phase6_aws.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

import bench_runner as br
import config
import dense_manifest as dm
import infra
import launch
from bench_runner import (
    build_runner_env,
    choose_instance_store,
    cohort_json,
    fio_command,
    host_budget_honored,
    host_pack_effective_bytes,
    mem_total_gib,
    parse_cohort_spec,
    parse_fio_bandwidth_gib_s,
    run_logged_process,
    summary_for_run,
)


def _job(**overrides) -> dict:
    job = {
        "run_id": "t-1", "model": "dsv4-flash", "gpu": "L4",
        "instance_type": "g6.2xlarge", "visible_gpus": 1, "cuda_archs": "89",
        "budget_mib": 3584, "host_gib": 17.0, "n_tokens": 64, "prompt": "",
        "cohort": "", "prompts": [], "cache_reset": "cold", "pinned": "",
        "spot": False, "max_hours": 3.0,
    }
    job.update(overrides)
    return job


FAMILIES = {name: spec["family"] for name, spec in config.GPU_SPECS.items()}


@pytest.mark.parametrize("gpu, host, expected", [
    ("L4", 16, "g6.2xlarge"),
    ("L4", 20, "g6.2xlarge"),       # 20 + 12 == 32: exact fit, no clamp
    ("L4", 21, "g6.4xlarge"),
    ("L4", 64, "g6.8xlarge"),       # 64 GiB box cannot hold 64 + 12
    ("L4", 116, "g6.8xlarge"),
    ("L4", 117, "g6.16xlarge"),
    ("A10", 16, "g5.2xlarge"),
    ("A10", 64, "g5.8xlarge"),
    ("L40S", 16, "g6e.2xlarge"),
    ("L40S", 64, "g6e.4xlarge"),
    ("L40S", 256, "g6e.16xlarge"),  # 256 box cannot hold 256 + 12
    ("2xL4", 16, "g6.12xlarge"),
])
def test_smallest_fitting_instance(gpu, host, expected):
    assert config.smallest_fitting_instance(FAMILIES[gpu], host) == expected


@pytest.mark.parametrize("gpu, host", [
    ("L4", 256),     # no L4 size has 268 GiB
    ("A10", 256),
    ("L4", 1024),    # the required too-large case
    ("2xL4", 181),   # only g6.12xlarge (192 GiB) exists here; 192 - 12 == 180
])
def test_too_large_host_budget_fails_closed(gpu, host):
    with pytest.raises(config.HostBudgetError, match="no instance"):
        config.smallest_fitting_instance(FAMILIES[gpu], host)


def test_require_host_budget_never_clamps():
    assert config.require_host_budget(20, 32) == 20.0
    with pytest.raises(config.HostBudgetError, match="needs 84.0 GiB"):
        config.require_host_budget(72, 64)
    with pytest.raises(config.HostBudgetError, match="must be positive"):
        config.require_host_budget(0, 32)


def test_instance_override_validation():
    assert config.validate_instance_override(
        "g6.16xlarge", FAMILIES["L4"], 16) == "g6.16xlarge"
    with pytest.raises(config.HostBudgetError, match="not in this GPU's family"):
        config.validate_instance_override("g5.2xlarge", FAMILIES["L4"], 16)
    with pytest.raises(config.HostBudgetError, match="needs"):
        config.validate_instance_override("g6.2xlarge", FAMILIES["L4"], 64)


def test_families_are_real_and_ascending_in_ram():
    for name, spec in config.GPU_SPECS.items():
        rams = [config.INSTANCE_TYPES[i]["ram_gib"] for i in spec["family"]]
        assert rams == sorted(rams), name
        for inst in spec["family"]:
            row = config.INSTANCE_TYPES[inst]
            assert row["vcpus"] > 0 and row["usd_per_hour"] > 0 and row["nvme_gb"] > 0


def test_runtime_ram_check_uses_memtotal_and_fails_closed():
    meminfo = "MemTotal:       32778748 kB\nMemFree:  1 kB\n"
    total = mem_total_gib(meminfo)
    assert 31.0 < total < 31.5
    config.runtime_ram_check(16, total)
    with pytest.raises(config.HostBudgetError, match="runtime MemTotal"):
        config.runtime_ram_check(20, total)
    with pytest.raises(RuntimeError, match="MemTotal missing"):
        mem_total_gib("MemFree: 1 kB\n")


def test_host_budget_honored_compares_runner_knobs():
    gib = 1 << 30
    assert host_budget_honored({"host_pack_bytes_gpu0_effective": 16 * gib},
                               16.0, 1)
    assert not host_budget_honored({"host_pack_bytes_gpu0_effective": 12 * gib},
                                   16.0, 1)
    assert host_budget_honored({"host_pack_bytes_gpu0_effective": 8 * gib,
                                "host_pack_bytes_gpu1_effective": 8 * gib}, 16.0, 2)
    assert not host_budget_honored({}, 16.0, 1)


def test_host_pack_knobs_parsed_from_runner_log():
    lines = ["[p4cfg] host_pack_bytes_gpu0_effective = 17179869184",
             "unrelated", "[p4cfg] host_pack_bytes_gpu0_requested = 1"]
    assert host_pack_effective_bytes(lines) == {
        "host_pack_bytes_gpu0_effective": 17179869184}


def test_run_logged_process_timeout_kills_hung_child():
    import time
    lines: list[str] = []
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        run_logged_process(
            [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(60)"],
            cwd=Path("."), env=dict(os.environ), timeout=2, on_line=lines.append)
    assert time.monotonic() - started < 20
    assert lines == ["start"]


def test_run_logged_process_nonzero_exit_raises():
    with pytest.raises(RuntimeError, match="command failed"):
        run_logged_process([sys.executable, "-c", "import sys; sys.exit(3)"],
                           cwd=Path("."), env=dict(os.environ), timeout=30,
                           on_line=lambda _: None)


def test_run_logged_process_success_streams_lines():
    lines: list[str] = []
    run_logged_process([sys.executable, "-c", "print('a'); print('b')"],
                       cwd=Path("."), env=dict(os.environ), timeout=30,
                       on_line=lines.append)
    assert lines == ["a", "b"]


def test_parse_cohort_range_and_groups():
    assert parse_cohort_spec("0-7") == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert parse_cohort_spec("0-3;4-7") == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert parse_cohort_spec("0,2;1,3") == [[0, 2], [1, 3]]


@pytest.mark.parametrize("bad", ["", "3-1", "1,1", "0;0-1", "0,,1", ";"])
def test_parse_cohort_fails_closed(bad):
    with pytest.raises(ValueError):
        parse_cohort_spec(bad)


def test_cohort_json_contract():
    assert cohort_json([[0, 1], [2, 3]]) == '{"groups":[[0,1],[2,3]]}'


def test_env_matches_modal_bench_defaults():
    env = build_runner_env(
        _job(), base_env={"PATH": "/usr/bin"}, store_path="/opt/dlami/nvme/stores/x",
        ckpt_dir="/kaggle/input/y", host_gib=17.0, headers_dir="/h",
        build_dir="/b", src_root="/s")
    assert env["NATIVE_EXPERT_STORE"] == "dee4_segmented"
    assert env["NATIVE_DEE4_SEGMENTED_STORE"] == "/opt/dlami/nvme/stores/x"
    assert env["NATIVE_CACHE_DTYPE"] == "fp4"
    assert env["NATIVE_N_TOKENS"] == "64"
    assert env["NATIVE_BUDGET_BYTES"] == str(3584 << 20)
    assert env["NATIVE_HOST_PACK_GPU0_BYTES"] == str(17 << 30)
    assert env["NATIVE_SINGLE_GPU"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert "NATIVE_COHORT_JSON" not in env
    assert env["PATH"] == "/usr/bin"


def test_env_two_gpu_splits_host_budget_and_hides_single_gpu_flag():
    env = build_runner_env(
        _job(gpu="2xL4", visible_gpus=2, host_gib=16.0), base_env={},
        store_path="/s", ckpt_dir="/c", host_gib=16.0, headers_dir="/h",
        build_dir="/b", src_root="/r")
    assert env["NATIVE_HOST_PACK_GPU0_BYTES"] == str(8 << 30)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "NATIVE_SINGLE_GPU" not in env


def test_env_cohort_sets_groups_and_prompts():
    prompts = [f"p{i}" for i in range(8)]
    env = build_runner_env(
        _job(cohort="0-7", prompts=prompts), base_env={}, store_path="/s",
        ckpt_dir="/c", host_gib=17.0, headers_dir="/h", build_dir="/b",
        src_root="/r")
    assert json.loads(env["NATIVE_COHORT_JSON"]) == {"groups": [list(range(8))]}
    assert json.loads(env["NATIVE_PROMPTS_JSON"]) == prompts
    assert "NATIVE_PROMPT" not in env


def test_env_cohort_without_prompts_fails_closed():
    with pytest.raises(ValueError):
        build_runner_env(
            _job(cohort="0-7"), base_env={}, store_path="/s", ckpt_dir="/c",
            host_gib=17.0, headers_dir="/h", build_dir="/b", src_root="/r")


def test_env_cohort_index_outside_prompts_fails_closed():
    with pytest.raises(ValueError):
        build_runner_env(
            _job(cohort="0-7", prompts=["a", "b"]), base_env={}, store_path="/s",
            ckpt_dir="/c", host_gib=17.0, headers_dir="/h", build_dir="/b",
            src_root="/r")


def test_instance_store_prefers_dlami_mount():
    mounts = "/dev/nvme0n1p1 / ext4 rw 0 0\n/dev/nvme1n1 /opt/dlami/nvme ext4 rw 0 0\n"
    decision = choose_instance_store({"blockdevices": []}, mounts)
    assert decision == {"action": "use", "mount": "/opt/dlami/nvme",
                        "device": "/dev/nvme1n1"}


def test_instance_store_formats_unmounted_nvme_fallback():
    lsblk = {"blockdevices": [
        {"name": "nvme0n1", "type": "disk", "size": "107374182400",
         "children": [{"name": "nvme0n1p1", "type": "part", "mountpoint": "/",
                       "size": "107374182400"}]},
        {"name": "nvme1n1", "type": "disk", "size": "483183820800", "children": []},
    ]}
    decision = choose_instance_store(lsblk, "/dev/nvme0n1p1 / ext4 rw 0 0\n")
    assert decision == {"action": "format_mount", "device": "/dev/nvme1n1",
                        "mount": "/mnt/dee-nvme"}


def test_instance_store_none_raises():
    lsblk = {"blockdevices": [{"name": "nvme0n1", "type": "disk",
                               "size": "107374182400",
                               "children": [{"name": "nvme0n1p1", "type": "part",
                                             "mountpoint": "/", "size": "1"}]}]}
    with pytest.raises(RuntimeError):
        choose_instance_store(lsblk, "/dev/nvme0n1p1 / ext4 rw 0 0\n")


def test_fio_bandwidth_parses_bw_bytes_and_legacy_bw():
    gib = 1 << 30
    assert parse_fio_bandwidth_gib_s({"jobs": [{"read": {"bw_bytes": 2 * gib}}]}) == 2.0
    assert parse_fio_bandwidth_gib_s(
        {"jobs": [{"read": {"bw": 1024 * 1024}}]}) == 1.0
    with pytest.raises(ValueError):
        parse_fio_bandwidth_gib_s({"jobs": [{"read": {"bw_bytes": 0}}]})


def test_fio_command_uses_record_block_and_direct_io(tmp_path):
    cmd = fio_command(rw="randread", path=tmp_path / "seg", size_gib=6,
                      out_json=tmp_path / "o.json")
    assert f"--bs={br.PROBE_RECORD_BYTES}" in cmd
    assert "--direct=1" in cmd and "--rw=randread" in cmd
    assert "--iodepth=6" in cmd and "--numjobs=3" in cmd


def test_cost_summary_uses_price_table():
    assert config.est_cost_usd(0.9776, 3600) == 0.9776
    assert config.est_cost_usd(0.9776, 1800) == 0.4888
    out = summary_for_run(base={"run_id": "r"}, wall_s=7200, usd_per_hour=2.0)
    assert out["est_cost_usd"] == 4.0
    assert out["pricing"]["date"] == config.PRICES_DATE
    assert "ASSUMPTION" in config.PRICES_NOTE


def test_dense_manifest_roundtrip_is_sparse_and_exact(tmp_path):
    header = json.dumps({
        "a.weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "routed.ffn.experts.0.w1.weight": {"dtype": "U8", "shape": [6],
                                           "data_offsets": [4, 10]},
        "b.weight": {"dtype": "U8", "shape": [2], "data_offsets": [10, 12]},
    }).encode()
    header += b" " * (-len(header) % 8)
    prefix = len(header).to_bytes(8, "little")
    full = prefix + header + bytes(range(1, 13))
    src = tmp_path / "orig.safetensors"
    src.write_bytes(full)

    header_bytes = prefix + header
    blob_dir = tmp_path / "blobs"
    (blob_dir / "blobs").mkdir(parents=True)
    dense_bytes = full[8 + len(header) + 0:8 + len(header) + 4] + \
        full[8 + len(header) + 10:8 + len(header) + 12]
    blob = blob_dir / "blobs" / "orig.safetensors.bin"
    blob.write_bytes(dense_bytes)
    file_size = dm.shard_file_size(header_bytes, [12, 10, 12])
    entries = [
        ["a.weight", 8 + len(header) + 0, 4, 0],
        ["b.weight", 8 + len(header) + 10, 2, 4],
    ]
    rec = dm.shard_record(
        header_bytes=header_bytes, file_size=file_size, blob="blobs/orig.safetensors.bin",
        blob_bytes=len(dense_bytes), blob_sha256=dm.sha256_file(blob), entries=entries)
    doc = dm.manifest(model="m", repo="r", revision="v", shards={"orig.safetensors": rec})
    (blob_dir / "manifest.json").write_text(json.dumps(doc))

    out = tmp_path / "out"
    (out_file,) = dm.materialize(blob_dir / "manifest.json", blob_dir, out)
    rebuilt = out_file.read_bytes()
    assert len(rebuilt) == len(full)
    assert rebuilt[:8 + len(header)] == full[:8 + len(header)]
    assert rebuilt[8 + len(header):8 + len(header) + 4] == full[8 + len(header):8 + len(header) + 4]
    assert rebuilt[8 + len(header) + 10:8 + len(header) + 12] == \
        full[8 + len(header) + 10:8 + len(header) + 12]
    assert rebuilt[8 + len(header) + 4:8 + len(header) + 10] == b"\x00" * 6


def test_dense_manifest_rejects_corrupt_blob(tmp_path):
    header = json.dumps({"t": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    header += b" " * (-len(header) % 8)
    header_bytes = len(header).to_bytes(8, "little") + header
    blob_dir = tmp_path
    (blob_dir / "blobs").mkdir()
    blob = blob_dir / "blobs" / "s.bin"
    blob.write_bytes(b"\x07")
    rec = dm.shard_record(header_bytes=header_bytes,
                          file_size=dm.shard_file_size(header_bytes, [1]),
                          blob="blobs/s.bin", blob_bytes=1,
                          blob_sha256="0" * 64, entries=[["t", 8 + len(header), 1, 0]])
    doc = dm.manifest(model="m", repo="r", revision="v", shards={"s": rec})
    (blob_dir / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sha256"):
        dm.materialize(blob_dir / "manifest.json", blob_dir, tmp_path / "out")


def test_userdata_watchdog_first_and_no_secrets():
    text = launch.render_userdata(
        run_id="r1", max_hours=3, job_json='{"a":1}', pre_entry=[],
        entry="/opt/dee-p6/venv/bin/python /opt/dee-p6/harness/bench_runner.py",
        pip_packages=["numpy"], hf_token_from_ssm=False)
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1].startswith("shutdown -h +180 ")
    assert "trap 'aws s3 cp /var/log/dee-p6-bootstrap.log" in text
    assert "shutdown -h now' EXIT" in text
    job_line = next(line for line in lines if line.startswith("echo "))
    assert base64.b64decode(job_line.split()[1]).decode() == '{"a":1}'
    assert "HF_TOKEN" not in text


def test_userdata_ssm_token_fetch_has_no_literal_value():
    text = launch.render_userdata(
        run_id="r2", max_hours=1, job_json="{}", pre_entry=["git clone x"],
        entry="run", pip_packages=[], hf_token_from_ssm=True)
    assert "aws ssm get-parameter --name /dee-p6/hf-token --with-decryption" in text
    assert text.index("git clone x") < text.index("\nrun\n")


def test_instance_policy_scopes_s3_and_tag_gated_terminate():
    policy = infra.instance_policy()
    flat = json.dumps(policy)
    assert "arn:aws:s3:::" + config.BUCKET in flat
    assert '"Resource": "*"' not in json.dumps(
        [s for s in policy["Statement"] if s.get("Sid") != "DecryptHfTokenViaSsm"])
    terminate = next(s for s in policy["Statement"]
                     if s["Sid"] == "SelfTerminateTaggedInstancesOnly")
    assert terminate["Condition"]["StringEquals"] == {"ec2:ResourceTag/dee-p6": "1"}


def test_s3_prefix_layout():
    assert config.s3_prefix("stores", "dsv4-flash") == \
        f"s3://{config.BUCKET}/stores/dsv4-flash/"
    assert config.s3_prefix("dense", "dsv4-flash").endswith("/dense/dsv4-flash/")
    assert config.s3_prefix("evidence", run_id="r").endswith("/evidence/r/")
    assert config.BUCKET == "dee-p6-147224001180-use2"


def test_store_resume_cutoff_drops_committed_but_lost_buckets():
    import store_build as sb
    assert sb.lost_bucket_cutoff({0, 1, 2}, {0, 1, 2}, set()) is None
    assert sb.lost_bucket_cutoff({0, 1, 2}, {0, 1}, {2}) is None
    assert sb.lost_bucket_cutoff({0, 1, 2}, {0}, set()) == 1


def test_prune_journal_keeps_only_buckets_below_cutoff(tmp_path):
    import store_build as sb
    journal = tmp_path / "build.journal.jsonl"
    journal.write_text(
        '{"bucket": 0, "segment_sha256": "a"}\n'
        '{"bucket": 1, "segment_sha256": "b"}\n'
        '{"bucket": 2, "segment_sha256": "c"}\n')
    assert sb.prune_journal_from(journal, 1) == 2
    assert sb.read_bucket_set(journal) == {0}


def test_read_bucket_set_ignores_torn_trailing_line(tmp_path):
    import store_build as sb
    journal = tmp_path / "j.jsonl"
    journal.write_text('{"bucket": 0}\n{"bucket": 1, "seg')
    assert sb.read_bucket_set(journal) == {0}


def test_local_segment_buckets_ignores_tombstones(tmp_path):
    import store_build as sb
    seg = tmp_path / "segments"
    seg.mkdir()
    (seg / "experts-bucket-03.dee4").write_bytes(b"x")
    (seg / "experts-bucket-04.dee4.tomb").write_bytes(b"")
    (seg / "experts-bucket-05.dee4").write_bytes(b"")
    assert sb.local_segment_buckets(seg) == {3, 5}


def test_progress_snapshot_eta_from_observed_rate():
    import store_build as sb
    snap = sb.progress_snapshot(
        n_buckets=46, committed={0, 1, 2, 3}, pushed={0, 1, 2}, baseline_pushed=0, elapsed_s=3600,
        disk_free_gib=50.0, http_status={"429": 0}, source_stats={"requests": 9},
        push_errors=[], peak_rss_kb=123, now_utc="t")
    assert snap["rate_buckets_per_h"] == 3.0
    assert snap["eta_h"] == 14.33
    assert snap["buckets_sealed"] == 4 and snap["buckets_pushed"] == 3
    resumed = sb.progress_snapshot(
        n_buckets=46, committed={0, 1, 2, 3}, pushed={0, 1, 2, 3}, baseline_pushed=2,
        elapsed_s=3600, disk_free_gib=1, http_status={}, source_stats={}, push_errors=[],
        peak_rss_kb=0, now_utc="t")
    assert resumed["rate_buckets_per_h"] == 2.0 and resumed["eta_h"] == 21.0
    empty = sb.progress_snapshot(
        n_buckets=46, committed=set(), pushed=set(), baseline_pushed=0, elapsed_s=0, disk_free_gib=1,
        http_status={}, source_stats={}, push_errors=[], peak_rss_kb=0, now_utc="t")
    assert empty["eta_h"] is None and empty["rate_buckets_per_h"] is None


def test_parse_s3_uri():
    import store_build as sb
    assert sb.parse_s3_uri("s3://b/stores/dsv4-flash/segments/x.dee4") == (
        "b", "stores/dsv4-flash/segments/x.dee4")
    with pytest.raises(ValueError):
        sb.parse_s3_uri("https://x")


def test_m7i_flex_is_free_tier_size_in_config():
    row = config.INSTANCE_TYPES["m7i-flex.large"]
    assert (row["vcpus"], row["ram_gib"], row["gpus"]) == (2, 8, 0)
    assert row["usd_per_hour"] == 0.09576


def test_retry_delay_honors_retry_after_and_caps():
    from hf_source import RETRY_CAP_S, retry_delay_s
    assert retry_delay_s(429, 0, "12") == 12.0
    assert retry_delay_s(429, 0, "99999") == RETRY_CAP_S
    assert retry_delay_s(429, 0) == 30.0
    assert retry_delay_s(429, 10) == RETRY_CAP_S
    assert retry_delay_s(503, 1) == 60.0
    assert retry_delay_s(404, 3) == 16.0


def test_cpu_userdata_has_no_venv_and_uses_system_python():
    text = launch.render_userdata(
        run_id="r3", max_hours=2, job_json="{}", pre_entry=[],
        entry="python3 /opt/dee-p6/harness/dense_extract.py", pip_packages=[],
        hf_token_from_ssm=True)
    assert "venv" not in text.replace("export PATH=/opt/dee-p6/venv/bin", "")
    assert "python3 /opt/dee-p6/harness/dense_extract.py" in text
