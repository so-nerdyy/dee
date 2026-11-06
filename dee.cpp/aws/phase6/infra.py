"""Idempotent AWS setup for the Phase-6 harness (us-east-2 only).

  python infra.py setup    create or refresh bucket, IAM role + instance
                           profile, and the egress-only security group
  python infra.py status   print bucket, role, security group and GPU quotas

Creates no compute. Re-running setup converges the same resources.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

from config import (
    ACCOUNT_ID,
    BUCKET,
    INSTANCE_PROFILE_NAME,
    PROFILE,
    REGION,
    ROLE_NAME,
    SECURITY_GROUP_NAME,
    TAG_KEY,
    TAG_VALUE,
)

GPU_QUOTAS = {
    "Running On-Demand G and VT instances": "L-DB2E81BA",
    "All G and VT Spot Instance Requests": "L-3819A6DF",
}

INLINE_POLICY_NAME = "dee-p6-least-privilege"
BUCKET_ARN = f"arn:aws:s3:::{BUCKET}"
PREFIXES = ("stores", "dense", "evidence", "src")


def session() -> boto3.Session:
    return boto3.Session(profile_name=PROFILE, region_name=REGION)


def trust_policy() -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT_ID}},
        }],
    }


def instance_policy() -> dict:
    object_arns = [f"{BUCKET_ARN}/{p}/*" for p in PREFIXES]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListHarnessPrefixes",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": BUCKET_ARN,
                "Condition": {"StringLike": {
                    "s3:prefix": [f"{p}/*" for p in PREFIXES]}},
            },
            {
                "Sid": "ReadHarnessArtifacts",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": object_arns,
            },
            {
                "Sid": "WriteStoresDenseEvidence",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
                "Resource": [f"{BUCKET_ARN}/{p}/*"
                             for p in ("stores", "dense", "evidence")],
            },
            {
                "Sid": "SelfTerminateTaggedInstancesOnly",
                "Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*",
                "Condition": {"StringEquals": {
                    f"ec2:ResourceTag/{TAG_KEY}": TAG_VALUE}},
            },
            {
                "Sid": "ReadHfTokenParameter",
                "Effect": "Allow",
                "Action": "ssm:GetParameter",
                "Resource": (f"arn:aws:ssm:{REGION}:{ACCOUNT_ID}:parameter"
                             "/dee-p6/*"),
            },
            {
                "Sid": "DecryptHfTokenViaSsm",
                "Effect": "Allow",
                "Action": "kms:Decrypt",
                "Resource": "*",
                "Condition": {"StringEquals": {
                    "kms:ViaService": f"ssm.{REGION}.amazonaws.com"}},
            },
        ],
    }


def bucket_policy_deny_insecure() -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "DenyInsecureTransport",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [BUCKET_ARN, f"{BUCKET_ARN}/*"],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }],
    }


def ensure_bucket(s3) -> str:
    try:
        s3.head_bucket(Bucket=BUCKET)
        state = "exists"
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION})
        state = "created"
    s3.put_public_access_block(
        Bucket=BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(
        Bucket=BUCKET,
        ServerSideEncryptionConfiguration={"Rules": [{
            "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
            "BucketKeyEnabled": False}]})
    s3.put_bucket_policy(Bucket=BUCKET,
                         Policy=json.dumps(bucket_policy_deny_insecure()))
    return state


def ensure_role(iam) -> str:
    try:
        iam.get_role(RoleName=ROLE_NAME)
        state = "exists"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy()),
            Description="dee-p6 benchmark instances: S3 prefixes + self-terminate")
        state = "created"
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName=INLINE_POLICY_NAME,
                        PolicyDocument=json.dumps(instance_policy()))
    iam.attach_role_policy(
        RoleName=ROLE_NAME,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    return state


def ensure_instance_profile(iam) -> str:
    try:
        iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        state = "exists"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "NoSuchEntity":
            raise
        iam.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        state = "created"
    profile = iam.get_instance_profile(
        InstanceProfileName=INSTANCE_PROFILE_NAME)["InstanceProfile"]
    if not any(r["RoleName"] == ROLE_NAME for r in profile["Roles"]):
        try:
            iam.add_role_to_instance_profile(
                InstanceProfileName=INSTANCE_PROFILE_NAME, RoleName=ROLE_NAME)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "LimitExceeded":
                raise
        if state == "created":
            # IAM instance-profile propagation (aws-iam guidance: sleep briefly).
            time.sleep(10)
    return state


def default_vpc_id(ec2) -> str:
    vpcs = ec2.describe_vpcs(
        Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise RuntimeError(f"no default VPC in {REGION}")
    return vpcs[0]["VpcId"]


def ensure_security_group(ec2) -> tuple[str, str]:
    vpc_id = default_vpc_id(ec2)
    found = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SECURITY_GROUP_NAME]},
        {"Name": "vpc-id", "Values": [vpc_id]},
    ])["SecurityGroups"]
    if found:
        group_id = found[0]["GroupId"]
        state = "exists"
        if found[0]["IpPermissions"]:
            ec2.revoke_security_group_ingress(
                GroupId=group_id, IpPermissions=found[0]["IpPermissions"])
    else:
        group_id = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME, VpcId=vpc_id,
            Description="dee-p6: egress only, no inbound (use SSM)",
            TagSpecifications=[{"ResourceType": "security-group", "Tags": [
                {"Key": TAG_KEY, "Value": TAG_VALUE}]}],
        )["GroupId"]
        state = "created"
    return group_id, state


def setup() -> dict:
    sess = session()
    s3 = sess.client("s3")
    iam = sess.client("iam")
    ec2 = sess.client("ec2")
    result = {
        "bucket": BUCKET,
        "bucket_state": ensure_bucket(s3),
        "role_state": ensure_role(iam),
        "instance_profile_state": ensure_instance_profile(iam),
    }
    group_id, sg_state = ensure_security_group(ec2)
    result.update({"security_group": group_id, "security_group_state": sg_state})
    return result


def bucket_summary(s3) -> dict:
    out = {"bucket": BUCKET}
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as exc:
        out["error"] = exc.response["Error"]["Code"]
        return out
    pab = s3.get_public_access_block(Bucket=BUCKET)[
        "PublicAccessBlockConfiguration"]
    enc = s3.get_bucket_encryption(Bucket=BUCKET)[
        "ServerSideEncryptionConfiguration"]["Rules"][0][
        "ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
    out.update({"public_access_block_all": all(pab.values()),
                "default_encryption": enc,
                "location": s3.get_bucket_location(Bucket=BUCKET)[
                    "LocationConstraint"]})
    return out


def quota_summary(sess) -> dict:
    sq = sess.client("service-quotas")
    out = {}
    for name, code in GPU_QUOTAS.items():
        value = sq.get_service_quota(ServiceCode="ec2", QuotaCode=code)[
            "Quota"]["Value"]
        history = sq.list_requested_service_quota_change_history_by_quota(
            ServiceCode="ec2", QuotaCode=code).get("RequestedQuotas", [])
        latest = max(history, key=lambda r: r["Created"], default=None)
        out[code] = {
            "name": name, "value": value,
            "latest_request": (
                {"id": latest["Id"], "status": latest["Status"],
                 "desired": latest["DesiredValue"]} if latest else None),
        }
    return out


def status() -> dict:
    sess = session()
    s3 = sess.client("s3")
    iam = sess.client("iam")
    ec2 = sess.client("ec2")
    out = {"region": REGION, "account": ACCOUNT_ID}
    out.update(bucket_summary(s3))
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        out["role"] = {"name": ROLE_NAME, "arn": role["Arn"]}
    except ClientError as exc:
        out["role"] = {"name": ROLE_NAME, "error": exc.response["Error"]["Code"]}
    try:
        profile = iam.get_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME)["InstanceProfile"]
        out["instance_profile"] = {
            "name": profile["InstanceProfileName"],
            "roles": [r["RoleName"] for r in profile["Roles"]]}
    except ClientError as exc:
        out["instance_profile"] = {
            "name": INSTANCE_PROFILE_NAME,
            "error": exc.response["Error"]["Code"]}
    groups = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SECURITY_GROUP_NAME]}])[
        "SecurityGroups"]
    out["security_groups"] = [
        {"id": g["GroupId"], "vpc": g["VpcId"],
         "ingress_rules": len(g["IpPermissions"])} for g in groups]
    out["gpu_quotas"] = quota_summary(sess)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["setup", "status"])
    args = parser.parse_args(argv)
    try:
        result = setup() if args.command == "setup" else status()
    except ClientError as exc:
        print(f"AWS error: {exc.response['Error']['Code']}: "
              f"{exc.response['Error'].get('Message', '')}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
