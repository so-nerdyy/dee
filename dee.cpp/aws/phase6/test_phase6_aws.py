"""Unit tests for the Phase-6 AWS harness pure logic (no AWS calls).

Run from this directory:  python -m pytest -q test_phase6_aws.py
"""

from __future__ import annotations

import base64
import json

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
    effective_host_gib,
    fio_command,
    parse_cohort_spec,
    parse_fio_bandwidth_gib_s,
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


def test_effective_host_gib_clamps_to_ram_minus_reserve():
    assert effective_host_gib(17.0, 32) == 17.0
    assert effective_host_gib(64.0, 32) == 32 - config.HOST_RAM_RESERVE_GIB
    assert effective_host_gib(64.0, 192) == 64.0


def test_effective_host_gib_rejects_nonpositive():
    with pytest.raises(ValueError):
        effective_host_gib(0, 32)


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
