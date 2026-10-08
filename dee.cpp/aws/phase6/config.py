"""Shared constants for the Phase-6 AWS harness.

Everything lives in us-east-2 (new AWS experience plan). This module is
stdlib-only because bench_runner.py imports it on the GPU instance.
"""

from __future__ import annotations

REGION = "us-east-2"
PROFILE = "dee"
ACCOUNT_ID = "147224001180"

BUCKET = f"dee-p6-{ACCOUNT_ID}-use2"
ROLE_NAME = "dee-p6-ec2"
INSTANCE_PROFILE_NAME = "dee-p6-ec2"
SECURITY_GROUP_NAME = "dee-p6-egress-only"
SSM_HF_TOKEN_PARAM = "/dee-p6/hf-token"

TAG_KEY = "dee-p6"
TAG_VALUE = "1"

REPO_URL = "https://github.com/so-nerdyy/dee.git"
REPO_BRANCH = "research/phase6-modal"

ROOT_VOLUME_GIB = 100
CPU_ROOT_VOLUME_GIB = 400
DLAMI_NAME_FILTER = "Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*"
DLAMI_OWNER = "amazon"

INSTANCE_STORE_MOUNT = "/opt/dlami/nvme"
INSTANCE_STORE_FALLBACK_MOUNT = "/mnt/dee-nvme"
INSTANCE_STORE_MIN_GB = 100

# On-demand Linux prices, us-east-2, read from the AWS Price List API on
# 2026-10-08. ASSUMPTION: price is current for that date only; spot is not
# priced, so spot runs are reported at the on-demand rate (an upper bound).
PRICES_DATE = "2026-10-08"
PRICES_NOTE = (
    "ASSUMPTION: on-demand Linux us-east-2 USD/h from the AWS Price List API "
    f"as read {PRICES_DATE}; spot runs are reported at on-demand (upper bound)."
)

INSTANCE_TYPES = {
    "g6.2xlarge": {"usd_per_hour": 0.9776, "ram_gib": 32, "nvme_gb": 450, "gpus": 1},
    "g6.12xlarge": {"usd_per_hour": 4.6016, "ram_gib": 192, "nvme_gb": 3760, "gpus": 4},
    "g5.2xlarge": {"usd_per_hour": 1.212, "ram_gib": 32, "nvme_gb": 450, "gpus": 1},
    "g6e.2xlarge": {"usd_per_hour": 2.24208, "ram_gib": 64, "nvme_gb": 450, "gpus": 1},
    "c7i.4xlarge": {"usd_per_hour": 0.714, "ram_gib": 32, "nvme_gb": 0, "gpus": 0},
}

# Mirrors GPU_SPECS / bench_* in modal/phase6/modal_p6.py.
# visible_gpus: how many of the instance's GPUs the runner may use.
GPU_SPECS = {
    "L4": {"instance": "g6.2xlarge", "visible_gpus": 1, "cuda_archs": "89",
           "budget_mib": 3584, "host_gib": 17.0},
    "2xL4": {"instance": "g6.12xlarge", "visible_gpus": 2, "cuda_archs": "89",
             "budget_mib": 3584, "host_gib": 17.0},
    "A10": {"instance": "g5.2xlarge", "visible_gpus": 1, "cuda_archs": "86",
            "budget_mib": 3584, "host_gib": 17.0},
    "L40S": {"instance": "g6e.2xlarge", "visible_gpus": 1, "cuda_archs": "89",
             "budget_mib": 28672, "host_gib": 32.0},
}

# Host-tier sizing: keep this much RAM free for torch, CUDA contexts, python
# and the OS. ASSUMPTION carried from the Modal baseline (5-8 GiB resident).
HOST_RAM_RESERVE_GIB = 12.0

MODELS = {
    "dsv4-flash": {
        "hf_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hf_rev": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        "store_dir": "dsv4-flash",
        "dataset_mount": "deepseek-v4-flash-0731-shards",
        "headers": ("dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
                    "shard-headers"),
        "runner": "dee.cpp/kaggle/deepseek-v4-flash-0731/"
                  "deepseek_v4_native_generate.py",
        "dense_supported": True,
    },
    "mimo-flash": {
        "hf_repo": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "hf_rev": "5711b268169967567844e1e560e8a3966da959b1",
        "store_dir": "mimo-flash",
        "dataset_mount": "mimo-v26-flash-ckpt",
        "spec": "dee.cpp/tools/phase3/specs/mimo_v2_flash.json",
        "runner": "dee.cpp/modal/phase6/mimo_native_generate.py",
        "dense_supported": False,
    },
}


def s3_prefix(kind: str, model: str = "", run_id: str = "") -> str:
    """Return the S3 URI prefix for one artifact family."""
    if kind == "src":
        return f"s3://{BUCKET}/src/phase6/"
    if kind == "stores":
        return f"s3://{BUCKET}/stores/{model}/"
    if kind == "dense":
        return f"s3://{BUCKET}/dense/{model}/"
    if kind == "evidence":
        return f"s3://{BUCKET}/evidence/{run_id}/"
    raise ValueError(f"unknown artifact kind: {kind!r}")


def est_cost_usd(usd_per_hour: float, seconds: float) -> float:
    """On-demand cost estimate for an instance that ran for `seconds`."""
    if usd_per_hour < 0 or seconds < 0:
        raise ValueError("price and duration must be non-negative")
    return round(usd_per_hour * seconds / 3600.0, 4)
