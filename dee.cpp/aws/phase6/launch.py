"""Launch one Phase-6 GPU bench run on EC2. Dry-run is the default.

  python launch.py --gpu L4 --model dsv4-flash --n-tokens 64 --host-gib 64          # dry run
  python launch.py --gpu L4 --model dsv4-flash --n-tokens 64 --host-gib 64 --launch # real

Dry run calls RunInstances with DryRun=True (validates permissions and quota,
creates nothing) and prints the rendered user-data. A real launch uploads the
harness to s3://<bucket>/src/phase6/ and then calls RunInstances for real.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import re
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

import config
from bench_runner import parse_cohort_spec

HARNESS_FILES = ["config.py", "dense_manifest.py", "bench_runner.py",
                 "dense_extract.py", "store_build.py", "hf_source.py"]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
BOOTSTRAP_LOG = "/var/log/dee-p6-bootstrap.log"


def session() -> boto3.Session:
    return boto3.Session(profile_name=config.PROFILE, region_name=config.REGION)


def resolve_dlami(ec2) -> dict:
    images = ec2.describe_images(
        Owners=[config.DLAMI_OWNER],
        Filters=[{"Name": "name", "Values": [config.DLAMI_NAME_FILTER]},
                 {"Name": "state", "Values": ["available"]},
                 {"Name": "architecture", "Values": ["x86_64"]}])["Images"]
    if not images:
        raise RuntimeError(f"no AMI matches {config.DLAMI_NAME_FILTER!r}")
    return max(images, key=lambda i: i["CreationDate"])


def default_subnet(ec2, vpc_id: str) -> str:
    subnets = ec2.describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]},
        {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    if not subnets:
        raise RuntimeError(f"no default subnet in {vpc_id}")
    return sorted(subnets, key=lambda s: s["AvailabilityZone"])[0]["SubnetId"]


def harness_security_group(ec2) -> tuple[str, str]:
    groups = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [config.SECURITY_GROUP_NAME]}])[
        "SecurityGroups"]
    if not groups:
        raise RuntimeError("security group missing: run `python infra.py setup` first")
    return groups[0]["GroupId"], groups[0]["VpcId"]


def render_userdata(*, run_id: str, max_hours: float, job_json: str,
                    pre_entry: list[str], entry: str, pip_packages: list[str],
                    hf_token_from_ssm: bool) -> str:
    minutes = max(1, int(round(max_hours * 60)))
    ev = f"{config.s3_prefix('evidence', run_id=run_id)}"
    job_b64 = base64.b64encode(job_json.encode("utf-8")).decode("ascii")
    lines = [
        "#!/bin/bash",
        f'shutdown -h +{minutes} "dee-p6 watchdog: max-hours reached for {run_id}"',
        "set -o pipefail",
        "mkdir -p /var/log /opt/dee-p6 /kaggle/working /kaggle/input",
        f"exec >>{BOOTSTRAP_LOG} 2>&1",
        f"trap 'aws s3 cp {BOOTSTRAP_LOG} {ev}bootstrap.log --only-show-errors; "
        "shutdown -h now' EXIT",
        f"export AWS_DEFAULT_REGION={config.REGION}",
        f"echo {job_b64} | base64 -d > /opt/dee-p6/job.json",
        f"aws s3 cp --recursive {config.s3_prefix('src')} /opt/dee-p6/harness/ "
        "--only-show-errors",
        "export PATH=/opt/dee-p6/venv/bin:/usr/local/cuda/bin:$PATH",
    ]
    if pip_packages:
        lines[-1:-1] = [
            "python3 -m venv /opt/dee-p6/venv",
            "/opt/dee-p6/venv/bin/pip install --quiet --upgrade pip",
            f"/opt/dee-p6/venv/bin/pip install --quiet {' '.join(pip_packages)}",
        ]
    if hf_token_from_ssm:
        lines.append(
            f"HF_TOKEN=$(aws ssm get-parameter --name {config.SSM_HF_TOKEN_PARAM} "
            "--with-decryption --query Parameter.Value --output text 2>/dev/null || true)")
        lines.append("export HF_TOKEN")
    lines += pre_entry
    lines.append(entry)
    return "\n".join(lines) + "\n"


def gpu_pip_packages() -> list[str]:
    return ["cmake", "pybind11", "numpy", "safetensors", "transformers",
            "huggingface_hub", "torch"]


def upload_harness(sess) -> None:
    s3 = sess.client("s3")
    for name in HARNESS_FILES:
        s3.upload_file(str(Path(__file__).with_name(name)), config.BUCKET,
                       f"src/phase6/{name}")
    print(f"uploaded harness to {config.s3_prefix('src')}")


def build_plan(args: argparse.Namespace) -> dict:
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"invalid --run-id {args.run_id!r}")
    gpu = config.GPU_SPECS[args.gpu]
    model = config.MODELS[args.model]
    host_gib = args.host_gib or gpu["default_host_gib"]
    try:
        if args.instance_type:
            instance_type = config.validate_instance_override(
                args.instance_type, gpu["family"], host_gib)
        else:
            instance_type = config.smallest_fitting_instance(gpu["family"], host_gib)
    except config.HostBudgetError as exc:
        raise SystemExit(f"refusing to launch: {exc}")
    instance = config.INSTANCE_TYPES[instance_type]
    if not model["dense_supported"]:
        raise SystemExit(f"{args.model} has no dense-only extraction yet; "
                         "launch is limited to dsv4-flash")
    prompts: list[str] = []
    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise SystemExit("--prompts-file must be a JSON list of strings")
    if args.cohort:
        groups = parse_cohort_spec(args.cohort)
        if not prompts:
            raise SystemExit("--cohort needs --prompts-file (fail-closed)")
        for group in groups:
            if max(group) >= len(prompts):
                raise SystemExit(f"cohort group {group} exceeds {len(prompts)} prompts")
    if args.max_hours <= 0:
        raise SystemExit("--max-hours must be positive")
    job = {
        "run_id": args.run_id,
        "model": args.model,
        "gpu": args.gpu,
        "instance_type": instance_type,
        "visible_gpus": gpu["visible_gpus"],
        "cuda_archs": gpu["cuda_archs"],
        "budget_mib": args.budget_mib or gpu["budget_mib"],
        "host_gib": host_gib,
        "n_tokens": args.n_tokens,
        "prompt": args.prompt,
        "cohort": args.cohort,
        "prompts": prompts,
        "cache_reset": args.cache_reset,
        "pinned": args.pinned,
        "spot": args.spot,
        "max_hours": args.max_hours,
        "repo": config.REPO_URL,
        "branch": config.REPO_BRANCH,
    }
    entry = ("/opt/dee-p6/venv/bin/python /opt/dee-p6/harness/bench_runner.py "
             "/opt/dee-p6/job.json")
    job_json = json.dumps(job, sort_keys=True)
    return {"job": job, "job_json": job_json, "instance_type": instance_type,
            "entry": entry, "instance": instance}


def launch(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    job = plan["job"]
    sess = session()
    ec2 = sess.client("ec2")
    ami = resolve_dlami(ec2)
    sg_id, vpc_id = harness_security_group(ec2)
    subnet = default_subnet(ec2, vpc_id)
    userdata = render_userdata(
        run_id=job["run_id"], max_hours=job["max_hours"], job_json=plan["job_json"],
        pre_entry=[], entry=plan["entry"], pip_packages=gpu_pip_packages(),
        hf_token_from_ssm=False)
    tags = [{"Key": config.TAG_KEY, "Value": config.TAG_VALUE},
            {"Key": "Name", "Value": f"dee-p6-{job['run_id']}"},
            {"Key": "dee-p6-run", "Value": job["run_id"]}]
    params = {
        "ImageId": ami["ImageId"],
        "InstanceType": plan["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": config.INSTANCE_PROFILE_NAME},
        "SecurityGroupIds": [sg_id],
        "SubnetId": subnet,
        "BlockDeviceMappings": [{
            "DeviceName": ami["RootDeviceName"],
            "Ebs": {"VolumeSize": config.ROOT_VOLUME_GIB, "VolumeType": "gp3",
                    "DeleteOnTermination": True, "Encrypted": True}}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled",
                            "HttpPutResponseHopLimit": 1},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "UserData": userdata,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags}],
    }
    if job["spot"]:
        params["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {"SpotInstanceType": "one-time",
                            "InstanceInterruptionBehavior": "terminate"}}

    quota_vcpus = sess.client("service-quotas").get_service_quota(
        ServiceCode="ec2", QuotaCode=config.GPU_ON_DEMAND_QUOTA_CODE)["Quota"]["Value"]
    vcpus = plan["instance"]["vcpus"]
    summary = {
        "mode": "launch" if args.launch else "dry-run",
        "run_id": job["run_id"],
        "gpu": job["gpu"], "host_gib_requested": job["host_gib"],
        "host_gib_effective": job["host_gib"],
        "instance_type": plan["instance_type"],
        "ram_gib": plan["instance"]["ram_gib"],
        "vcpus": vcpus,
        "usd_per_hour": plan["instance"]["usd_per_hour"],
        "max_cost_usd": config.est_cost_usd(plan["instance"]["usd_per_hour"],
                                            job["max_hours"] * 3600),
        "gpu_vcpu_quota": {
            "approved_on_demand_vcpus": quota_vcpus,
            "requested_on_demand_vcpus": config.GPU_ON_DEMAND_QUOTA_REQUESTED_VCPUS,
            "instance_fits_approved_quota": vcpus <= quota_vcpus,
        },
        "ami": {"id": ami["ImageId"], "name": ami["Name"]},
        "security_group": sg_id, "subnet": subnet,
        "spot": job["spot"], "pricing_date": config.PRICES_DATE,
    }
    print(json.dumps(summary, indent=2))

    if args.launch:
        upload_harness(sess)
    params["DryRun"] = not args.launch
    try:
        resp = ec2.run_instances(**params)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "DryRunOperation" and not args.launch:
            print("DryRunOperation: RunInstances would have succeeded "
                  "(validated permissions and quota; nothing was created)")
        else:
            print(f"RunInstances error: {code}: "
                  f"{exc.response['Error'].get('Message', '')}", file=sys.stderr)
            print("\n--- rendered user-data ---\n" + userdata)
            return 1
    else:
        print(f"launched instance {resp['Instances'][0]['InstanceId']}")
    print("\n--- rendered user-data ---\n" + userdata)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", required=True, choices=sorted(config.GPU_SPECS))
    parser.add_argument("--model", default="dsv4-flash", choices=sorted(config.MODELS))
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--host-gib", type=float, default=0.0,
                        help="total host-tier GiB (0 = GPU default). Never clamped: "
                             "the smallest family size with RAM >= host+reserve is "
                             "chosen, or the launch fails")
    parser.add_argument("--instance-type", default="",
                        help="override the size; must be in the GPU's family and fit host-gib")
    parser.add_argument("--budget-mib", type=int, default=0,
                        help="per-GPU VRAM expert budget MiB (0 = GPU default)")
    parser.add_argument("--cache-reset", default="cold")
    parser.add_argument("--cohort", default="",
                        help='cohort spec, e.g. "0-7" (needs --prompts-file)')
    parser.add_argument("--prompts-file", default="",
                        help="JSON list of prompt strings for cohort mode")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--pinned", default="", help="commit to check out")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="default: RunInstances(DryRun=True), creates nothing")
    mode.add_argument("--launch", action="store_true",
                      help="create the instance (billable)")
    args = parser.parse_args(argv)
    if not args.run_id:
        args.run_id = (f"{args.model}-{args.gpu}-"
                       f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    try:
        return launch(args)
    except ClientError as exc:
        print(f"AWS error: {exc.response['Error']['Code']}: "
              f"{exc.response['Error'].get('Message', '')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
