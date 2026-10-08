# Phase-6 AWS harness

Port of the Modal Phase-6 bench (`modal/phase6/modal_p6.py`) to EC2, us-east-2 only.
Nothing here creates billable compute unless `--launch` is passed.

| file | role | runs on |
|---|---|---|
| `config.py` | region, bucket, GPU and instance table, prices, model registry | everywhere (stdlib only) |
| `infra.py` | `setup` (idempotent bucket, IAM role and profile, security group) and `status` | your machine |
| `launch.py` | one GPU bench run: `--dry-run` default, `--launch` to create | your machine |
| `launch_cpu.py` | one CPU job (`--job dense` or `--job store`): dry-run default | your machine |
| `bench_runner.py` | `_bench_impl` port: build, store sync, fio probe, dense, native run, summary | GPU instance |
| `dense_extract.py` | range-reads the dense-only DSv4 checkpoint to S3 | CPU instance |
| `store_build.py` | segmented DEE4 store build, resume-safe, synced to S3 | CPU instance |
| `dense_manifest.py` | compact dense-only shard format (materialize on the instance) | both |
| `test_phase6_aws.py` | pytest for the pure logic and the user-data contract | local |

## S3 layout (`s3://dee-p6-147224001180-use2/`)

- `src/phase6/`: harness files, uploaded by `launch.py --launch` and `launch_cpu.py --launch`
- `stores/<model>/`: segmented DEE4 store (`metadata.json`, segments, journal)
- `dense/<model>/`: `manifest.json` plus `blobs/<shard>.bin` (compact dense tensors only)
- `evidence/<run_id>/`: `bench.log`, `environment.json`, `fio-*.json`, `summary.json`, runner output

## Setup

```
python -m pip install boto3 "botocore[crt]"        # botocore[crt] is needed for the `dee` login profile
cd dee.cpp/aws/phase6
python infra.py setup       # idempotent
python infra.py status      # bucket, role, profile, security group, GPU quota values
python -m pytest -q test_phase6_aws.py
```

`setup` creates: the bucket (block public access, SSE-S3, deny non-TLS); IAM role
`dee-p6-ec2` with least-privilege S3 access to this bucket, tag-gated
`ec2:TerminateInstances` (`dee-p6=1`), and read of the `/dee-p6/*` SSM parameters
(plus `AmazonSSMManagedInstanceCore`); instance profile `dee-p6-ec2`; and security
group `dee-p6-egress-only` with no inbound rules (use SSM Session Manager, not SSH).

## Dry-run, then launch

GPU bench:

```
python launch.py --gpu L4 --model dsv4-flash --n-tokens 64 --host-gib 64            # DRY RUN (default)
python launch.py --gpu L4 --model dsv4-flash --n-tokens 64 --host-gib 64 --launch   # bills
```

Dry run calls `RunInstances(DryRun=True)` (validates permissions and parameters,
creates nothing) and prints the rendered user-data. A real launch uploads the harness
to `src/phase6/` and then calls `RunInstances`.

Flags: `--gpu {L4,2xL4,A10,L40S}`, `--model`, `--n-tokens`, `--host-gib` (0 = GPU default),
`--budget-mib`, `--cache-reset`, `--cohort` (e.g. `0-7`, needs `--prompts-file`), `--prompt`,
`--pinned <commit>`, `--run-id`, `--spot`, `--max-hours` (default 3).

CPU jobs (same flow, default c7i.4xlarge, 400 GiB root):

```
python launch_cpu.py --job dense --model dsv4-flash                 # dry run
python launch_cpu.py --job store --model dsv4-flash --launch        # bills
```

`launch_cpu.py --launch` writes `HF_TOKEN` from the local environment to SSM SecureString
`/dee-p6/hf-token`. The token is never placed in user-data. Instances fetch it at boot.

### Runtime safety

- user-data arms `shutdown -h +<max_hours*60>` as its first action. A bootstrap `trap ... EXIT`
  uploads the bootstrap log and powers off on any failure. `bench_runner.py` powers off in `finally`.
- Instances terminate on shutdown (`InstanceInitiatedShutdownBehavior=terminate`).
- IMDSv2 is required. Root is 100 GiB gp3 encrypted (GPU), 400 GiB (CPU).
- Evidence is synced every 5 min during a run, and at exit.

### Store and dense flow

1. `launch_cpu.py --job dense`: `dense_extract.py` range-reads the 1564 tensors that the
   segmented-store runner needs from the pinned HF revision, checks each shard header against
   the committed headers and the total length against the server, then writes `dense/dsv4-flash/`.
2. `launch_cpu.py --job store`: `store_build.py` runs `p3_kaggle_job.py build --source remote`
   (same command as `modal_p6_ops.build_store`), restoring from and syncing to `stores/dsv4-flash/`.
3. GPU run: `bench_runner.py` syncs the store to `/opt/dlami/nvme` (or formats `/dev/nvme1n1` to
   `/mnt/dee-nvme`), runs the fio probe, materializes the dense shards, and symlinks
   `/kaggle/input/deepseek-v4-flash-0731-shards`.

### Dense-only checkpoint (verified design)

The segmented-store path is not a plain "non-expert shards" case. The runner
(`kaggle/deepseek-v4-flash-0731/deepseek_v4_native_generate.py`) calls
`download_all_shards()` before the segmented branch, and that only requires all 48 shard
files to exist. The C++ engine mmaps every shard at its original length and reads tensor
offsets from the committed headers. So each rebuilt shard keeps the original 8-byte prefix,
header, and length, with only the needed tensor bytes written at their original offsets
(everything else is a sparse hole that the segmented path never reads). S3 stores only the
compact blobs (8.238 GiB, 1564 tensors, 45 shards). See `dense_manifest.py`.

## Costs

Instance prices are on-demand Linux us-east-2, read from the AWS Price List API on 2026-10-08
(`config.py`, `PRICES_NOTE`). They are an ASSUMPTION for that date. Spot runs are reported
at the on-demand rate, which is an upper bound.

| instance | GPUs used | vCPUs | USD/h | RAM | NVMe | used for |
|---|---|---|---|---|---|---|
| g6.2xlarge | 1x L4 | 8 | 0.9776 | 32 GiB | 450 GB | `L4` |
| g6.4xlarge | 1x L4 | 16 | 1.3232 | 64 GiB | 600 GB | `L4` |
| g6.8xlarge | 1x L4 | 32 | 2.0144 | 128 GiB | 900 GB | `L4` |
| g6.16xlarge | 1x L4 | 64 | 3.3968 | 256 GiB | 1880 GB | `L4` |
| g6.12xlarge | 2 of 4 L4 | 48 | 4.6016 | 192 GiB | 3760 GB | `2xL4` (runner pins 2) |
| g5.2xlarge / 4x / 8x / 16x | 1x A10G | 8 / 16 / 32 / 64 | 1.212 / 1.624 / 2.448 / 4.096 | 32 / 64 / 128 / 256 GiB | 450 / 600 / 900 / 1900 GB | `A10` |
| g6e.2xlarge / 4x / 8x / 16x | 1x L40S | 8 / 16 / 32 / 64 | 2.24208 / 3.00424 / 4.52856 / 7.57719 | 64 / 128 / 256 / 512 GiB | 450 / 600 / 900 / 1900 GB | `L40S` |
| c7i.4xlarge | CPU | 16 | 0.714 | 32 GiB | none | dense / store jobs |

G/VT on-demand quota is counted in vCPUs. It is 0 approved and 8 requested, so only the 8 vCPU
sizes (2xlarge) fit the current request. The dry run prints this check.

`summary.json` records `est_cost_usd = USD/h x wall_s / 3600`. `launch.py` prints
`max_cost_usd = USD/h x max_hours`. Storage (S3 and EBS) is not priced here.

### Host budget: fail closed, never clamp

`--host-gib` is the pre-registered budget, so it is never reduced. `launch.py` picks the smallest
size in the GPU's family whose RAM covers `host_gib + HOST_RAM_RESERVE_GIB` (12 GiB, ASSUMPTION
carried from Modal). If no size fits, the launch refuses. `--instance-type` overrides the size, but
it must be in the family and must also fit. At runtime, `bench_runner.py` reads MemTotal from
`/proc/meminfo` and fails before building if RAM is short. After the run it compares the runner's
logged `host_pack_bytes_gpuN_effective` against the request. Mismatch sets
`host_budget_honored: false` and the verdict `HOST_BUDGET_NOT_HONORED`. `summary.json` carries
`host_gib_requested` and `host_gib_effective`, which are always equal.

Families (RAM from DescribeInstanceTypes; `--host-gib` 16 / 64 / 256 result shown):

| --gpu | sizes (RAM GiB) | 16 | 64 | 256 |
|---|---|---|---|---|
| L4 | g6.2xl (32) / g6.4xl (64) / g6.8xl (128) / g6.16xl (256) | g6.2xlarge | g6.8xlarge | refused |
| A10 | g5.2xl (32) / g5.4xl (64) / g5.8xl (128) / g5.16xl (256) | g5.2xlarge | g5.8xlarge | refused |
| L40S | g6e.2xl (64) / g6e.4xl (128) / g6e.8xl (256) / g6e.16xl (512) | g6e.2xlarge | g6e.4xlarge | g6e.16xlarge |
| 2xL4 | g6.12xl (192; 4 GPUs, runner uses 2) | g6.12xlarge | g6.12xlarge | refused |

Because the reserve is 12 GiB, a 256 GiB budget does not fit a 256 GiB L4 or A10 box.
Reaching it there would need a larger instance or a smaller reserve. Both are decisions for the lead.

Prices and vCPUs are in `config.py`, read on 2026-10-08. `--host-gib` is always an explicit
decision, never a silent reduction.

## Evidence

Each GPU run writes `s3://<bucket>/evidence/<run_id>/`: `summary.json` (with
`measured_b_ssd_gib_s`, `measured_b_ssd_seq_gib_s`, `ssd_probe`, `est_cost_usd`,
`host_gib_effective`, `commit`, `verdict`), `environment.json`, `fio-randread.json`,
`fio-read.json`, `bench.log`, and the runner's JSON/log outputs.

The SSD probe runs before the native generate: fio, O_DIRECT, block = 13,369,344 B (one packed
expert record), libaio, iodepth 6, numjobs 3, 30 s, on the largest store segment. The
prediction matrix grades on `measured_b_ssd_gib_s`, the random-read value. A run whose probe
fails stops before the native generate (unscorable).
