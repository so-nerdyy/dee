"""Phase-6 GPU bench runner. Executes on the EC2 instance via user-data.

Port of modal/phase6/modal_p6.py::_bench_impl with Modal removed:
  watchdog (user-data) -> env capture -> clone pinned commit -> cmake/ctest/pydee
  -> instance-store stage (s3 sync) -> fio SSD probe -> dense-only checkpoint
  -> native generate runner under NATIVE_* env -> evidence -> summary.json
  -> s3 upload -> shutdown.

Usage (on the instance):  python bench_runner.py /opt/dee-p6/job.json
stdlib only, plus the aws CLI and the venv python for the runner itself.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import config
from dense_manifest import materialize

WORK = Path("/kaggle/working")
REPO_ROOT = Path("/opt/dee-p6/repo")
DENSE_ROOT = Path("/opt/dee-p6/dense-blobs")
KAGGLE_INPUT = Path("/kaggle/input")
PROBE_RECORD_BYTES = 13_369_344
PROBE_IODEPTH = 6
PROBE_NUMJOBS = 3
PROBE_RUNTIME_S = 30
PROBE_SIZE_GIB = 6
HEARTBEAT_S = 300


# ---------------------------------------------------------------------------
# Pure logic (unit-tested in test_phase6_aws.py)
# ---------------------------------------------------------------------------

def effective_host_gib(requested: float, ram_gib: float,
                       reserve_gib: float = config.HOST_RAM_RESERVE_GIB) -> float:
    """Clamp the host-tier budget so the instance keeps `reserve_gib` free."""
    if requested <= 0:
        raise ValueError("host budget must be positive")
    return round(min(float(requested), max(0.0, ram_gib - reserve_gib)), 3)


def parse_cohort_spec(spec: str) -> list[list[int]]:
    """'0-7' -> [[0..7]];  '0-3;4-7' -> [[0,1,2,3],[4,5,6,7]];  '0,2;1,3' ok.

    Fail-closed: empty groups, descending ranges, and duplicate indices raise.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty cohort spec")
    groups: list[list[int]] = []
    seen: set[int] = set()
    for group_text in spec.split(";"):
        indices: list[int] = []
        for token in group_text.split(","):
            token = token.strip()
            if not token:
                raise ValueError(f"empty index in cohort spec {spec!r}")
            if "-" in token:
                lo_text, hi_text = token.split("-", 1)
                lo, hi = int(lo_text), int(hi_text)
                if hi < lo:
                    raise ValueError(f"descending range {token!r}")
                indices.extend(range(lo, hi + 1))
            else:
                indices.append(int(token))
        if not indices:
            raise ValueError(f"empty cohort group in {spec!r}")
        dupes = (set(indices) & seen) | {i for i in indices if indices.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate prompt indices {sorted(dupes)} in {spec!r}")
        seen.update(indices)
        groups.append(indices)
    return groups


def cohort_json(groups: list[list[int]]) -> str:
    """The NATIVE_COHORT_JSON contract: {"groups": [[i, ...], ...]}."""
    return json.dumps({"groups": groups}, separators=(",", ":"))


def check_cohort_against_prompts(groups: list[list[int]], n_prompts: int) -> None:
    for group in groups:
        if max(group) >= n_prompts or min(group) < 0:
            raise ValueError(f"cohort group {group} indexes outside {n_prompts} prompts")


def build_runner_env(job: dict, *, base_env: dict[str, str], store_path: str,
                     ckpt_dir: str, host_gib: float, headers_dir: str,
                     build_dir: str, src_root: str) -> dict[str, str]:
    """NATIVE_* environment, mirroring modal_p6._bench_impl section 4."""
    visible = int(job["visible_gpus"])
    pack_per_gpu = host_gib / visible
    env = dict(base_env)
    env.update({
        "NATIVE_SOURCE_TREE": src_root,
        "NATIVE_RUN_ID": job["run_id"],
        "NATIVE_ARM_ID": f"p6-{job['gpu']}",
        "NATIVE_EXPERT_STORE": "dee4_segmented",
        "NATIVE_DEE4_SEGMENTED_STORE": store_path,
        "NATIVE_CACHE_DTYPE": "fp4",
        "NATIVE_EVICTION_POLICY": "lru",
        "NATIVE_HOST_CACHE_MODE": "lru",
        "NATIVE_SOURCE_READ_LANES": "4",
        "NATIVE_SOURCE_READ_QUEUE_DEPTH": "6",
        "NATIVE_N_TOKENS": str(job["n_tokens"]),
        "NATIVE_LRU_TOTAL_CAP_GIB": str(host_gib),
        "NATIVE_BUDGET_BYTES": str(int(job["budget_mib"]) << 20),
        "NATIVE_HOST_PACK_GPU0_BYTES": str(int(pack_per_gpu * (1 << 30))),
        "NATIVE_HOST_PACK_GPU1_BYTES": str(int(pack_per_gpu * (1 << 30))),
        "NATIVE_CACHE_RESET": job["cache_reset"],
        "NATIVE_FORCE_TMP": "1",
        "NATIVE_TRACE_REQUESTS": "1",
        "NATIVE_PROFILE": "1",
        "NATIVE_MODEL_CKPT": ckpt_dir,
        "NATIVE_HEADERS_DIR": headers_dir,
        "DEE_BUILD_DIR": build_dir,
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(visible)),
    })
    if visible == 1:
        env["NATIVE_SINGLE_GPU"] = "1"
    if job.get("cohort"):
        groups = parse_cohort_spec(job["cohort"])
        prompts = job.get("prompts") or []
        if not prompts:
            raise ValueError("cohort mode requires a prompt list (fail-closed)")
        check_cohort_against_prompts(groups, len(prompts))
        env["NATIVE_COHORT_JSON"] = cohort_json(groups)
        env["NATIVE_PROMPTS_JSON"] = json.dumps(prompts)
    elif job.get("prompt"):
        env["NATIVE_PROMPT"] = job["prompt"]
    return env


def choose_instance_store(lsblk: dict, mounts_text: str) -> dict:
    """Pick the instance-store NVMe, preferring the DLAMI mount.

    Returns {"action": "use", "mount": path} when already mounted, or
    {"action": "format_mount", "device": dev, "mount": path} when an
    unformatted instance-store disk must be prepared. Raises if none exists.
    """
    mounted = {}
    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            mounted[parts[1]] = parts[0]
    if config.INSTANCE_STORE_MOUNT in mounted:
        return {"action": "use", "mount": config.INSTANCE_STORE_MOUNT,
                "device": mounted[config.INSTANCE_STORE_MOUNT]}
    if config.INSTANCE_STORE_FALLBACK_MOUNT in mounted:
        return {"action": "use", "mount": config.INSTANCE_STORE_FALLBACK_MOUNT,
                "device": mounted[config.INSTANCE_STORE_FALLBACK_MOUNT]}

    def has_mount(node: dict) -> bool:
        if node.get("mountpoint"):
            return True
        return any(has_mount(child) for child in node.get("children") or [])

    candidates = []
    for dev in lsblk.get("blockdevices", []):
        if dev.get("type") != "disk" or not str(dev.get("name", "")).startswith("nvme"):
            continue
        if has_mount(dev):
            continue
        size_gb = int(dev.get("size") or 0) / 1e9
        if size_gb >= config.INSTANCE_STORE_MIN_GB:
            candidates.append((dev["name"], size_gb))
    if not candidates:
        raise RuntimeError("no mounted or unmounted instance-store NVMe disk found")
    name, _ = max(candidates, key=lambda c: c[1])
    return {"action": "format_mount", "device": f"/dev/{name}",
            "mount": config.INSTANCE_STORE_FALLBACK_MOUNT}


def parse_fio_bandwidth_gib_s(fio_json: dict) -> float:
    read = fio_json["jobs"][0]["read"]
    if "bw_bytes" in read:
        bytes_per_s = float(read["bw_bytes"])
    else:
        bytes_per_s = float(read["bw"]) * 1024.0
    if bytes_per_s <= 0:
        raise ValueError("fio reported zero read bandwidth")
    return round(bytes_per_s / (1 << 30), 4)


def fio_command(*, rw: str, path: Path, size_gib: float, out_json: Path) -> list[str]:
    return [
        "fio", f"--name=dee_p6_{rw}", f"--filename={path}",
        f"--size={int(size_gib * (1 << 30))}", f"--rw={rw}",
        f"--bs={PROBE_RECORD_BYTES}", "--direct=1", "--ioengine=libaio",
        f"--iodepth={PROBE_IODEPTH}", f"--numjobs={PROBE_NUMJOBS}",
        f"--runtime={PROBE_RUNTIME_S}", "--time_based", "--group_reporting",
        "--output-format=json", f"--output={out_json}",
    ]


def summary_for_run(*, base: dict, wall_s: float, usd_per_hour: float) -> dict:
    out = dict(base)
    out.update({
        "wall_s": round(wall_s, 1),
        "est_cost_usd": config.est_cost_usd(usd_per_hour, wall_s),
        "pricing": {"usd_per_hour": usd_per_hour, "date": config.PRICES_DATE,
                    "note": config.PRICES_NOTE},
    })
    return out


# ---------------------------------------------------------------------------
# Side-effecting helpers (run on the instance)
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.run_id = job["run_id"]
        self.ev_dir = Path("/opt/dee-p6/evidence") / self.run_id
        self.ev_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.ev_dir / "bench.log"
        self.s3_evidence = config.s3_prefix("evidence", run_id=self.run_id)
        self.t_start = time.monotonic()
        self.wall_start = time.time()
        self.summary: dict = {"run_id": self.run_id, "model": job["model"],
                              "gpu": job["gpu"], "instance_type": job["instance_type"],
                              "spot": bool(job.get("spot")),
                              "commit": None, "verdict": "UNKNOWN"}

    def log(self, line: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {line}\n")
        print(line, flush=True)

    def upload_evidence(self) -> None:
        subprocess.run(["aws", "s3", "sync", str(self.ev_dir), self.s3_evidence,
                        "--only-show-errors"], check=False, timeout=900)

    def heartbeat(self, proc: subprocess.Popen | None, stop: threading.Event) -> None:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        while not stop.wait(HEARTBEAT_S):
            try:
                alive = proc is None or proc.poll() is None
                snap = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=15).stdout.strip()
                self.log(f"[hb] alive={alive} gpu=[{snap}] "
                         f"elapsed={time.monotonic() - self.t_start:.0f}s")
                pool.submit(self.upload_evidence).result(timeout=900)
            except Exception as exc:  # noqa: BLE001
                self.log(f"[hb] error (continuing): {exc!r}")
        pool.shutdown(wait=False)

    def run_logged(self, cmd: list[str], *, cwd: Path, env: dict[str, str],
                   timeout: int) -> None:
        self.log("+ " + " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            tail.append(line)
            tail = tail[-40:]
        if proc.wait(timeout=timeout) != 0:
            raise RuntimeError(f"command failed: {' '.join(cmd)}\n" + "\n".join(tail[-20:]))


def env_capture(run: Run) -> None:
    def cmd_out(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc!r}"

    env = {
        "nvidia_smi": cmd_out(["nvidia-smi"]),
        "nvcc": cmd_out(["bash", "-lc", "nvcc --version 2>/dev/null | tail -1"]),
        "free_gib": cmd_out(["free", "-g"]),
        "disk": cmd_out(["df", "-h", "/", "/opt"]),
        "uname": cmd_out(["uname", "-a"]),
        "python": sys.version,
        "aws_cli": cmd_out(["aws", "--version"]),
    }
    (run.ev_dir / "environment.json").write_text(json.dumps(env, indent=2))
    run.log(f"[env] gpu: {env['nvidia_smi'].splitlines()[:3]}")


def clone_and_build(run: Run, build_env: dict[str, str]) -> Path:
    job = run.job
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    run.run_logged(["git", "clone", "--branch", config.REPO_BRANCH,
                    "--single-branch", config.REPO_URL, str(REPO_ROOT)],
                   cwd=Path("/"), env=build_env, timeout=900)
    if job.get("pinned"):
        run.run_logged(["git", "checkout", "--quiet", job["pinned"]],
                       cwd=REPO_ROOT, env=build_env, timeout=120)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    run.summary["commit"] = head
    dee = REPO_ROOT / "dee.cpp"
    build = dee / "build-modal"
    jobs = str(max(1, os.cpu_count() or 8))
    run.run_logged(["cmake", "-S", str(dee), "-B", str(build),
                    f"-DCMAKE_CUDA_ARCHITECTURES={job['cuda_archs']}",
                    "-DDEE_CUDA=ON", "-DDEE_BUILD_TESTS=ON",
                    "-DCMAKE_BUILD_TYPE=Release"],
                   cwd=dee, env=build_env, timeout=900)
    run.run_logged(["cmake", "--build", str(build), "--target", "dee_core",
                    "-j", jobs], cwd=dee, env=build_env, timeout=3600)
    run.run_logged(["cmake", "--build", str(build), "--target",
                    "test_dee4_segmented", "-j", jobs],
                   cwd=dee, env=build_env, timeout=1200)
    run.run_logged(["ctest", "--test-dir", str(build), "-R",
                    "^test_dee4_segmented$", "--output-on-failure"],
                   cwd=dee, env=build_env, timeout=600)
    run.summary["store_reader_test"] = "PASS"
    run.run_logged([sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
                   cwd=dee, env={**build_env, "DEE_BUILD_DIR": str(build)},
                   timeout=1800)
    return build


def ensure_instance_store(run: Run) -> Path:
    lsblk = json.loads(subprocess.run(
        ["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,MOUNTPOINT"],
        capture_output=True, text=True, check=True).stdout)
    mounts = Path("/proc/mounts").read_text()
    decision = choose_instance_store(lsblk, mounts)
    run.log(f"[store] instance store decision: {decision}")
    mount = Path(decision["mount"])
    if decision["action"] == "format_mount":
        subprocess.run(["mkfs.ext4", "-F", "-q", decision["device"]], check=True)
        mount.mkdir(parents=True, exist_ok=True)
        subprocess.run(["mount", decision["device"], str(mount)], check=True)
    run.summary["instance_store"] = decision
    return mount


def sync_store(run: Run, mount: Path, model: str) -> Path:
    local = mount / "stores" / model
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    subprocess.run(["aws", "s3", "sync", config.s3_prefix("stores", model),
                    str(local), "--only-show-errors"], check=True)
    meta = local / "metadata.json"
    if not meta.is_file():
        raise RuntimeError(f"store metadata missing after sync: {meta}")
    smeta = json.loads(meta.read_text())
    store_bytes = sum(int(s["bytes"]) for s in smeta.get("segments", []))
    dt = time.monotonic() - t0
    run.log(f"[store] synced {store_bytes / (1 << 30):.1f} GiB in {dt:.0f}s")
    run.summary.update({"store_gib": round(store_bytes / (1 << 30), 1),
                        "store_sync_s": round(dt, 1)})
    return local


def ssd_probe(run: Run, store_dir: Path) -> None:
    meta = json.loads((store_dir / "metadata.json").read_text())
    segments = sorted(meta.get("segments", []), key=lambda s: int(s["bytes"]),
                      reverse=True)
    target = store_dir / segments[0]["file"]
    results = {}
    for rw in ("randread", "read"):
        out_json = run.ev_dir / f"fio-{rw}.json"
        cmd = fio_command(rw=rw, path=target, size_gib=PROBE_SIZE_GIB,
                          out_json=out_json)
        run.log("+ " + " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=PROBE_RUNTIME_S * 4 + 120)
        results[rw] = parse_fio_bandwidth_gib_s(json.loads(out_json.read_text()))
    run.summary.update({
        "measured_b_ssd_gib_s": results["randread"],
        "measured_b_ssd_seq_gib_s": results["read"],
        "ssd_probe": {
            "file": str(target), "block_bytes": PROBE_RECORD_BYTES,
            "iodepth": PROBE_IODEPTH, "numjobs": PROBE_NUMJOBS,
            "direct": True, "runtime_s": PROBE_RUNTIME_S,
            "size_gib": PROBE_SIZE_GIB, "ioengine": "libaio",
            "measured_randread_gib_s": results["randread"],
            "measured_seq_gib_s": results["read"],
        },
    })
    run.log(f"[ssd] randread {results['randread']} GiB/s, "
            f"seq {results['read']} GiB/s")
    if results["randread"] <= 0:
        raise RuntimeError("SSD probe produced no bandwidth; run is unscorable")


def materialize_dense(run: Run, model: str, mount: Path) -> Path:
    blobs = DENSE_ROOT / model
    blobs.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", config.s3_prefix("dense", model),
                    str(blobs), "--only-show-errors"], check=True)
    shards_dir = mount / "dense" / model / "shards"
    materialize(blobs / "manifest.json", blobs, shards_dir)
    dataset_mount = config.MODELS[model]["dataset_mount"]
    KAGGLE_INPUT.mkdir(parents=True, exist_ok=True)
    link = KAGGLE_INPUT / dataset_mount
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(shards_dir, target_is_directory=True)
    run.log(f"[dense] {link} -> {shards_dir}")
    return shards_dir


def collect_evidence(run: Run, started_at: float) -> None:
    for f in WORK.iterdir() if WORK.exists() else []:
        if (f.is_file() and f.name.endswith((".json", ".jsonl", ".log"))
                and f.stat().st_mtime >= started_at):
            try:
                shutil.copy2(f, run.ev_dir / f.name)
            except OSError:
                pass


def run_native(run: Run, env: dict[str, str], runner: Path) -> int:
    proc = subprocess.Popen([sys.executable, str(runner)], env=env, cwd=str(WORK),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    stop = threading.Event()
    threading.Thread(target=run.heartbeat, args=(proc, stop), daemon=True).start()
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        run.log(f"[runner] {line}")
        tail.append(line)
        tail = tail[-80:]
    code = proc.wait()
    stop.set()
    run.summary["tail"] = tail[-50:]
    for ln in tail:
        if ln.startswith("VERDICT:"):
            run.summary["verdict"] = ln.split(":", 1)[1].strip()
    return code


def run_bench(job: dict) -> dict:
    run = Run(job)
    model = job["model"]
    spec = config.MODELS[model]
    instance = config.INSTANCE_TYPES[job["instance_type"]]
    usd_per_hour = instance["usd_per_hour"]
    host_gib = effective_host_gib(job["host_gib"], instance["ram_gib"])
    run.summary["host_gib_requested"] = job["host_gib"]
    run.summary["host_gib_effective"] = host_gib
    run.summary["budget_mib"] = job["budget_mib"]
    run.summary["n_tokens"] = job["n_tokens"]
    run.summary["cohort"] = job.get("cohort") or None
    run.summary["cache_reset"] = job["cache_reset"]
    base_env = dict(os.environ)
    base_env["PATH"] = (f"/opt/dee-p6/venv/bin:/usr/local/cuda/bin:"
                        f"{base_env.get('PATH', '')}")
    base_env["AWS_DEFAULT_REGION"] = config.REGION
    try:
        env_capture(run)
        build = clone_and_build(run, base_env)
        mount = ensure_instance_store(run)
        store_dir = sync_store(run, mount, spec["store_dir"])
        ssd_probe(run, store_dir)
        shards_dir = materialize_dense(run, model, mount)
        env = build_runner_env(
            job, base_env=base_env, store_path=str(store_dir), ckpt_dir=str(shards_dir),
            host_gib=host_gib, headers_dir=str(REPO_ROOT / spec.get("headers", "")),
            build_dir=str(build), src_root=str(REPO_ROOT))
        run.log(f"[run] n_tokens={job['n_tokens']} host_gib={host_gib} "
                f"budget_mib={job['budget_mib']} cache={job['cache_reset']} "
                f"visible_gpus={job['visible_gpus']} cohort={job.get('cohort') or '-'}")
        started_at = time.time()
        code = run_native(run, env, REPO_ROOT / spec["runner"])
        collect_evidence(run, started_at)
        run.summary["exit_code"] = code
    except Exception as exc:  # noqa: BLE001
        run.summary["verdict"] = f"{type(exc).__name__}: {exc}"
        run.summary["traceback"] = traceback.format_exc()[-4000:]
        run.log(f"[fatal] {exc!r}")
    finally:
        wall_s = time.monotonic() - run.t_start
        final = summary_for_run(base=run.summary, wall_s=wall_s,
                                usd_per_hour=usd_per_hour)
        final["launched_at_wall"] = run.wall_start
        (run.ev_dir / "summary.json").write_text(json.dumps(final, indent=2))
        run.log(f"[done] wall={wall_s:.0f}s est_cost=${final['est_cost_usd']}")
        try:
            run.upload_evidence()
        except Exception as exc:  # noqa: BLE001
            print(f"evidence upload failed: {exc!r}", flush=True)
    return final


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[1]).read_text())
    try:
        run_bench(job)
        return 0
    finally:
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
