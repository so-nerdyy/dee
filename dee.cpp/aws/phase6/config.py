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
REPO_BRANCH = "research/phase6-aws"

GPU_ON_DEMAND_QUOTA_CODE = "L-DB2E81BA"
GPU_ON_DEMAND_QUOTA_REQUESTED_VCPUS = 8

ROOT_VOLUME_GIB = 100
CPU_ROOT_VOLUME_GIB = 400
DLAMI_NAME_FILTER = "Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*"
DLAMI_OWNER = "amazon"

INSTANCE_STORE_MOUNT = "/opt/dlami/nvme"
INSTANCE_STORE_FALLBACK_MOUNT = "/mnt/dee-nvme"
INSTANCE_STORE_MIN_GB = 100

# On-demand Linux prices, us-east-2, read from the AWS Price List API on
# 2026-10-08. RAM and vCPU from EC2 DescribeInstanceTypes on the same day.
# ASSUMPTION: prices are current for that date only; spot is not priced, so
# spot runs are reported at the on-demand rate (an upper bound).
PRICES_DATE = "2026-10-08"
PRICES_NOTE = (
    "ASSUMPTION: on-demand Linux us-east-2 USD/h from the AWS Price List API "
    f"as read {PRICES_DATE}; spot runs are reported at on-demand (upper bound)."
)

# name -> RAM GiB, vCPUs, instance-store NVMe GB, GPUs on the instance, USD/h
INSTANCE_TYPES = {
    "g6.2xlarge": {"usd_per_hour": 0.9776, "ram_gib": 32, "vcpus": 8, "nvme_gb": 450, "gpus": 1},
    "g6.4xlarge": {"usd_per_hour": 1.3232, "ram_gib": 64, "vcpus": 16, "nvme_gb": 600, "gpus": 1},
    "g6.8xlarge": {"usd_per_hour": 2.0144, "ram_gib": 128, "vcpus": 32, "nvme_gb": 900, "gpus": 1},
    "g6.12xlarge": {"usd_per_hour": 4.6016, "ram_gib": 192, "vcpus": 48, "nvme_gb": 3760, "gpus": 4},
    "g6.16xlarge": {"usd_per_hour": 3.3968, "ram_gib": 256, "vcpus": 64, "nvme_gb": 1880, "gpus": 1},
    "g5.2xlarge": {"usd_per_hour": 1.212, "ram_gib": 32, "vcpus": 8, "nvme_gb": 450, "gpus": 1},
    "g5.4xlarge": {"usd_per_hour": 1.624, "ram_gib": 64, "vcpus": 16, "nvme_gb": 600, "gpus": 1},
    "g5.8xlarge": {"usd_per_hour": 2.448, "ram_gib": 128, "vcpus": 32, "nvme_gb": 900, "gpus": 1},
    "g5.16xlarge": {"usd_per_hour": 4.096, "ram_gib": 256, "vcpus": 64, "nvme_gb": 1900, "gpus": 1},
    "g6e.2xlarge": {"usd_per_hour": 2.24208, "ram_gib": 64, "vcpus": 8, "nvme_gb": 450, "gpus": 1},
    "g6e.4xlarge": {"usd_per_hour": 3.00424, "ram_gib": 128, "vcpus": 16, "nvme_gb": 600, "gpus": 1},
    "g6e.8xlarge": {"usd_per_hour": 4.52856, "ram_gib": 256, "vcpus": 32, "nvme_gb": 900, "gpus": 1},
    "g6e.16xlarge": {"usd_per_hour": 7.57719, "ram_gib": 512, "vcpus": 64, "nvme_gb": 1900, "gpus": 1},
    "m7i-flex.large": {"usd_per_hour": 0.09576, "ram_gib": 8, "vcpus": 2, "nvme_gb": 0, "gpus": 0},
    "c7i.xlarge": {"usd_per_hour": 0.1785, "ram_gib": 8, "vcpus": 4, "nvme_gb": 0, "gpus": 0},
    "c7i.2xlarge": {"usd_per_hour": 0.357, "ram_gib": 16, "vcpus": 8, "nvme_gb": 0, "gpus": 0},
    "c7i.4xlarge": {"usd_per_hour": 0.714, "ram_gib": 32, "vcpus": 16, "nvme_gb": 0, "gpus": 0},
}

# --gpu name -> the instance family it may run on (smallest-fit order is by RAM),
# visible_gpus (how many of the instance's GPUs the runner uses), CUDA archs,
# per-GPU VRAM expert budget, and the default host-tier budget.
GPU_SPECS = {
    "L4": {"family": ["g6.2xlarge", "g6.4xlarge", "g6.8xlarge", "g6.16xlarge"],
           "visible_gpus": 1, "cuda_archs": "89", "budget_mib": 3584,
           "default_host_gib": 16.0},
    "2xL4": {"family": ["g6.12xlarge"], "visible_gpus": 2, "cuda_archs": "89",
             "budget_mib": 3584, "default_host_gib": 16.0},
    "A10": {"family": ["g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.16xlarge"],
            "visible_gpus": 1, "cuda_archs": "86", "budget_mib": 3584,
            "default_host_gib": 16.0},
    "L40S": {"family": ["g6e.2xlarge", "g6e.4xlarge", "g6e.8xlarge", "g6e.16xlarge"],
             "visible_gpus": 1, "cuda_archs": "89", "budget_mib": 28672,
             "default_host_gib": 32.0},
}

# Host-tier sizing: RAM kept free for torch, CUDA contexts, python and the OS.
# ASSUMPTION carried from the Modal baseline (5-8 GiB resident).
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


class HostBudgetError(ValueError):
    """Requested host-tier budget cannot be honored on the instance."""


def require_host_budget(host_gib: float, ram_gib: float,
                        reserve_gib: float = HOST_RAM_RESERVE_GIB) -> float:
    """Return host_gib if RAM covers it plus the reserve; otherwise raise.

    Never clamps: a clamped run would be graded against the wrong
    pre-registered host-budget row.
    """
    if host_gib <= 0:
        raise HostBudgetError("host budget must be positive")
    if ram_gib - reserve_gib < host_gib:
        raise HostBudgetError(
            f"host_gib={host_gib} needs {host_gib + reserve_gib:.1f} GiB RAM "
            f"(budget + {reserve_gib} GiB reserve) but the instance has {ram_gib} GiB")
    return float(host_gib)


def smallest_fitting_instance(family: list[str], host_gib: float,
                              reserve_gib: float = HOST_RAM_RESERVE_GIB) -> str:
    """Smallest RAM size in `family` whose RAM covers host_gib + reserve."""
    if host_gib <= 0:
        raise HostBudgetError("host budget must be positive")
    fits = sorted((INSTANCE_TYPES[name]["ram_gib"], name) for name in family
                  if INSTANCE_TYPES[name]["ram_gib"] - reserve_gib >= host_gib)
    if not fits:
        largest = max(INSTANCE_TYPES[name]["ram_gib"] for name in family)
        raise HostBudgetError(
            f"no instance in {family} fits host_gib={host_gib} "
            f"(needs {host_gib + reserve_gib:.1f} GiB RAM; largest is {largest} GiB)")
    return fits[0][1]


def validate_instance_override(instance_type: str, family: list[str],
                               host_gib: float,
                               reserve_gib: float = HOST_RAM_RESERVE_GIB) -> str:
    """Validate a manual --instance-type the same way as the family pick."""
    if instance_type not in family:
        raise HostBudgetError(f"{instance_type} is not in this GPU's family {family}")
    require_host_budget(host_gib, INSTANCE_TYPES[instance_type]["ram_gib"], reserve_gib)
    return instance_type


def runtime_ram_check(host_gib: float, mem_total_gib: float,
                      reserve_gib: float = HOST_RAM_RESERVE_GIB) -> None:
    """Runtime twin of require_host_budget, using MemTotal from /proc/meminfo."""
    if mem_total_gib - reserve_gib < host_gib:
        raise HostBudgetError(
            f"runtime MemTotal {mem_total_gib:.2f} GiB cannot hold host_gib={host_gib} "
            f"plus the {reserve_gib} GiB reserve")
