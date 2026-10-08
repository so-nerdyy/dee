"""Launch a CPU job: dense-only checkpoint extraction or segmented-store build.

  python launch_cpu.py --job dense --model dsv4-flash --run-id dense-dsv4        # dry run
  python launch_cpu.py --job store --model dsv4-flash --run-id store-dsv4 --launch

Dry-run is the default: RunInstances(DryRun=True), nothing created, the
rendered user-data is printed. HF_TOKEN from the local environment is written
to an SSM SecureString only on --launch; it never appears in user-data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

from botocore.exceptions import ClientError

import config
from launch import (
    default_subnet,
    harness_security_group,
    render_userdata,
    resolve_dlami,
    session,
    upload_harness,
)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
CPU_PIP_PACKAGES: list[str] = []  # stdlib only; no venv on the DLAMI (no ensurepip)
DEFAULT_INSTANCE = "c7i.4xlarge"


def sync_hf_token(sess, token: str) -> None:
    sess.client("ssm").put_parameter(
        Name=config.SSM_HF_TOKEN_PARAM, Value=token, Type="SecureString",
        Overwrite=True)
    print(f"stored HF token in SSM SecureString {config.SSM_HF_TOKEN_PARAM}")


def hf_token_param_exists(sess) -> bool:
    try:
        sess.client("ssm").get_parameter(Name=config.SSM_HF_TOKEN_PARAM)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ParameterNotFound":
            return False
        raise
    return True


def build_entry(args: argparse.Namespace) -> str:
    base = "python3 /opt/dee-p6/harness"
    if args.job == "dense":
        return (f"{base}/dense_extract.py --model {args.model} "
                "--repo-root /opt/dee-p6/repo "
                f"--out-dir /opt/dee-p6/dense-out/{args.model} --s3")
    evidence = config.s3_prefix("evidence", run_id=args.run_id)
    return (f"{base}/store_build.py --model {args.model} "
            "--src-root /opt/dee-p6/repo "
            f"--build-dir /opt/dee-p6/store-build/{args.model} --s3 "
            f"--evidence {evidence} --prefetch {args.prefetch}")


def pre_entry(args: argparse.Namespace) -> list[str]:
    clone = (f"git clone --branch {config.REPO_BRANCH} --single-branch "
             f"{config.REPO_URL} /opt/dee-p6/repo")
    lines = [clone]
    if args.pinned:
        lines.append(f"git -C /opt/dee-p6/repo checkout --quiet {args.pinned}")
    return lines


def launch(args: argparse.Namespace) -> int:
    sess = session()
    ec2 = sess.client("ec2")
    if args.instance_type not in config.INSTANCE_TYPES or \
            config.INSTANCE_TYPES[args.instance_type]["gpus"] != 0:
        raise SystemExit(f"{args.instance_type} is not a CPU instance in config")
    ami = resolve_dlami(ec2)
    sg_id, vpc_id = harness_security_group(ec2)
    subnet = default_subnet(ec2, vpc_id)
    token = os.environ.get("HF_TOKEN", "")
    if args.launch and token:
        sync_hf_token(sess, token)
    if not (token or hf_token_param_exists(sess)):
        print("WARNING: no HF token (env or SSM); anonymous reads may hit HTTP 429")

    job = {"job": args.job, "model": args.model, "run_id": args.run_id,
           "instance_type": args.instance_type, "max_hours": args.max_hours,
           "pinned": args.pinned}
    userdata = render_userdata(
        run_id=args.run_id, max_hours=args.max_hours,
        job_json=json.dumps(job, sort_keys=True), pre_entry=pre_entry(args),
        entry=build_entry(args), pip_packages=CPU_PIP_PACKAGES,
        hf_token_from_ssm=True)
    tags = [{"Key": config.TAG_KEY, "Value": config.TAG_VALUE},
            {"Key": "Name", "Value": f"dee-p6-{args.job}-{args.run_id}"},
            {"Key": "dee-p6-run", "Value": args.run_id}]
    params = {
        "ImageId": ami["ImageId"],
        "InstanceType": args.instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": config.INSTANCE_PROFILE_NAME},
        "SecurityGroupIds": [sg_id],
        "SubnetId": subnet,
        "BlockDeviceMappings": [{
            "DeviceName": ami["RootDeviceName"],
            "Ebs": {"VolumeSize": args.disk_gb, "VolumeType": "gp3",
                    "DeleteOnTermination": True, "Encrypted": True}}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled",
                            "HttpPutResponseHopLimit": 1},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "UserData": userdata,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags}],
    }
    price = config.INSTANCE_TYPES[args.instance_type]["usd_per_hour"]
    summary = {
        "mode": "launch" if args.launch else "dry-run",
        "job": args.job, "model": args.model, "run_id": args.run_id,
        "instance_type": args.instance_type, "usd_per_hour": price,
        "max_cost_usd": config.est_cost_usd(price, args.max_hours * 3600),
        "disk_gb": args.disk_gb, "ami": {"id": ami["ImageId"], "name": ami["Name"]},
        "security_group": sg_id, "subnet": subnet,
        "hf_token_in_env": bool(token),
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
                  "(nothing was created)")
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
    parser.add_argument("--job", required=True, choices=["dense", "store"])
    parser.add_argument("--model", default="dsv4-flash", choices=sorted(config.MODELS))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE)
    parser.add_argument("--disk-gb", type=int, default=config.CPU_ROOT_VOLUME_GIB)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--pinned", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="default: RunInstances(DryRun=True), creates nothing")
    mode.add_argument("--launch", action="store_true",
                      help="create the instance (billable)")
    args = parser.parse_args(argv)
    if not args.run_id:
        args.run_id = (f"{args.job}-{args.model}-"
                       f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"invalid --run-id {args.run_id!r}")
    try:
        return launch(args)
    except ClientError as exc:
        print(f"AWS error: {exc.response['Error']['Code']}: "
              f"{exc.response['Error'].get('Message', '')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
