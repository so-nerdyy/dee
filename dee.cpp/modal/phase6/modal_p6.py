"""Phase 6 — Modal harness: stage checkpoints, build DEE4 stores, run TPS benchmarks.

Limited-environment thesis: GPUs are L4 / A10 / L40S (RTX-PRO-6000 only if a
backbone cannot fit 48 GB).  No A100.

Functions
  stage_ckpt   CPU  snapshot_download HF checkpoint -> dee-models volume
  build_store  CPU  clone pinned commit -> p3 manifest -> segmented build via
               RemoteRangeSource (HF byte-range reads; the checkpoint is never
               downloaded for repack) -> dee-stores volume.  Resume-safe:
               build.journal.jsonl + *.partial records survive restarts.
  bench-<GPU>  GPU  clone pinned commit -> cmake dee_core + pydee -> run the
               native generate driver under NATIVE_* env -> evidence +
               report -> dee-p6-evidence volume.

Volumes
  dee-models         HF checkpoints (shared with earlier campaigns)
  dee-stores         dee4-v4-segmented stores (one subdir per model)
  dee-p6-evidence    benchmark evidence + reports

Usage
  modal run modal/phase6/modal_p6.py::stage_ckpt --model dsv4-flash
  modal run --detach modal/phase6/modal_p6.py::build_store --model dsv4-flash
  modal run modal/phase6/modal_p6.py::bench_L4 --model dsv4-flash --n-tokens 64
"""

from __future__ import annotations

import modal

# --- Pinned source -----------------------------------------------------------
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "research/phase6-modal"  # Phase 6 runs on Modal (2026-11-04)
PINNED_COMMIT = ""  # empty = branch head; set for sealed runs

# --- Volumes -----------------------------------------------------------------
VOL_MODELS = "/vol/models"
VOL_STORES = "/vol/stores"
VOL_EVID = "/kaggle/working"  # the runner writes evidence to WORK=/kaggle/working

vol_models = modal.Volume.from_name("dee-models", create_if_missing=True)
vol_stores = modal.Volume.from_name("dee-stores", create_if_missing=True)
vol_evid = modal.Volume.from_name("dee-p6-evidence", create_if_missing=True)

# --- Model registry ----------------------------------------------------------
# Adapter work adds entries here; each needs ckpt_dir (dee-models layout),
# store_dir (dee-stores layout) and the runner script that drives generation.
MODELS = {
    "dsv4-flash": {
        "hf_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "hf_rev": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
        "ckpt_dir": "DeepSeek-V4-Flash-0731",
        "store_dir": "dsv4-flash",
        "runner": "dee.cpp/kaggle/deepseek-v4-flash-0731/"
                  "deepseek_v4_native_generate.py",
        # DATASET_DIR probe target inside the runner (symlink -> volume dir).
        "dataset_mount": "deepseek-v4-flash-0731-shards",
        # committed shard headers for p3 manifest rebuild
        "headers": "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
                   "shard-headers",
        # ~147 GiB store; dense backbone ~14 GiB -> VRAM floor fits one L4
    },
    "mimo-flash": {
        "hf_repo": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "hf_rev": "5711b268169967567844e1e560e8a3966da959b1",
        "ckpt_dir": "MiMo-V2.6-Flash-RL",
        "store_dir": "mimo-flash",
        "runner": "dee.cpp/modal/phase6/mimo_native_generate.py",
        "dataset_mount": "mimo-v26-flash-ckpt",
        # headers fetched to <store>/headers by the ops build_store spec path
    },
}

# --- GPU matrix (limited environments only) ----------------------------------
# name -> (modal gpu spec, CMAKE_CUDA_ARCHS, default VRAM budget MiB,
#          container RAM MiB, default host-tier cap GiB)
# Defaults mirror the sealed campaign config (3584 MiB VRAM arena, 17 GiB
# host tier split 8.5/8.5 per GPU) so the first run is a clean A/B vs T4;
# pass --budget-mib/--host-gib to scale residency on the bigger cards.
GPU_SPECS = {
    "L4":         ("L4",         "89",        3584,  98304, 17.0),
    "2xL4":       ("L4:2",       "89",        3584,  98304, 17.0),
    "A10":        ("A10",        "86",        3584,  98304, 17.0),
    "L40S":       ("L40S",       "89",       28672, 131072, 32.0),
    "RTXPRO6000": ("RTX-PRO-6000", "120",    61440, 196608, 64.0),
    # Diagnostic only (no frozen cell): the sealed reference hardware, used to
    # bisect the r4 L4 token divergence (GPU arch vs one-GPU mode vs env/store).
    "T4":         ("T4",         "75",        3584,  98304, 17.0),
    "2xT4":       ("T4:2",       "75",        3584,  98304, 17.0),
}
GPU_PRICE_PER_S = {
    "L4": 0.000222, "2xL4": 2 * 0.000222, "A10": 0.000306,
    "L40S": 0.000542, "RTXPRO6000": 0.000842,
    "T4": 0.000164, "2xT4": 2 * 0.000164,
}
# RAM $0.00000222/GiB/s, CPU $0.0000131/core/s -> container overhead matters;
# RAM is sized per spec so host-tier sweeps stay inside the container.

# --- Images ------------------------------------------------------------------
image_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("huggingface_hub", "hf_transfer", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# CUDA 12.8 devel covers sm_75..sm_120; add_python keeps a clean 3.11.
image_gpu = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "zlib1g-dev", "fio")
    # Same torch build as every sealed T4 run (environment.json: 2.10.0+cu128);
    # an unpinned torch is one more variable in a cross-GPU token comparison.
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "torch==2.10.0",   # restated so no later resolve can move it
        "transformers", "safetensors", "numpy", "setuptools",  # 3.12 ships none
        "huggingface_hub", "hf_transfer", "pybind11", "requests",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # add_python's interpreter records clang/clang++ in sysconfig and the image
    # has only gcc: pin every setuptools build (harness and runner) to gcc.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CC": "gcc", "CXX": "g++",
          "LDSHARED": "g++ -shared", "LDCXXSHARED": "g++ -shared"})
)

app = modal.App("dee-p6")


def _clone(pinned: str = "") -> list[str]:
    """Shell lines: clone the pinned source tree the runner expects."""
    commit = pinned or PINNED_COMMIT
    lines = [
        "set -e",
        f"git clone --branch {BRANCH} --single-branch "
        f"{REPOSITORY} /tmp/dsv4-native-src",
    ]
    if commit:
        lines.append(f"cd /tmp/dsv4-native-src && git checkout --quiet {commit}")
    lines.append("cd /tmp/dsv4-native-src && git rev-parse HEAD")
    return lines


def _sh(lines: list[str]) -> str:
    return " && ".join(lines)


# ---------------------------------------------------------------------------
# stage_ckpt: HF checkpoint -> dee-models volume (idempotent)
# ---------------------------------------------------------------------------
@app.function(
    image=image_cpu,
    cpu=4,
    memory=16384,
    timeout=8 * 3600,
    volumes={VOL_MODELS: vol_models},
)
def stage_ckpt(model: str) -> dict:
    import os
    import time

    from huggingface_hub import snapshot_download

    spec = MODELS[model]
    dest = os.path.join(VOL_MODELS, spec["ckpt_dir"])
    os.makedirs(dest, exist_ok=True)
    t0 = time.monotonic()
    snap = snapshot_download(
        repo_id=spec["hf_repo"],
        revision=spec["hf_rev"],
        local_dir=dest,
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.txt",
                        "*.model", "tokenizer*"],
        max_workers=8,
    )
    vol_models.commit()
    n_files = sum(1 for _ in __import__("pathlib").Path(snap).rglob("*")
                  if _.is_file())
    gb = sum(_.stat().st_size for _ in __import__("pathlib").Path(snap)
             .rglob("*") if _.is_file()) / (1 << 30)
    return {"model": model, "snap": snap, "files": n_files,
            "gib": round(gb, 1), "wall_s": round(time.monotonic() - t0, 1)}


# ---------------------------------------------------------------------------
# build_store: segmented DEE4 store via HF range reads -> dee-stores volume
# ---------------------------------------------------------------------------
@app.function(
    image=image_cpu,
    cpu=8,
    memory=32768,
    timeout=12 * 3600,
    volumes={VOL_STORES: vol_stores},
)
def build_store(model: str, prefetch: int = 8, buckets: int = 0) -> dict:
    import os
    import subprocess
    import time

    spec = MODELS[model]
    src = "/tmp/dsv4-native-src"
    t0 = time.monotonic()
    subprocess.run(_sh(_clone()), shell=True, check=True)

    build_dir = os.path.join(VOL_STORES, spec["store_dir"])
    os.makedirs(build_dir, exist_ok=True)
    cmd = [
        "python", "tools/phase3/p3_kaggle_job.py", "build",
        "--build-dir", build_dir,
        "--headers", os.path.join(src, spec["headers"]),
        "--source", "remote",
        "--prefetch", str(prefetch),
        "--publisher", "none",
    ]
    if buckets:
        cmd += ["--buckets", str(buckets)]
    proc = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    vol_stores.commit()
    tail = (proc.stdout + "\n---STDERR---\n" + proc.stderr)[-8000:]
    return {"model": model, "rc": proc.returncode, "tail": tail,
            "wall_s": round(time.monotonic() - t0, 1)}


@app.function(
    image=image_cpu,
    cpu=2,
    memory=8192,
    timeout=3600,
    volumes={VOL_STORES: vol_stores},
)
def store_status(model: str) -> str:
    import subprocess

    spec = MODELS[model]
    src = "/tmp/dsv4-native-src"
    subprocess.run(_sh(_clone()), shell=True, check=True)
    r = subprocess.run(
        ["python", "tools/phase3/p3_kaggle_job.py", "status",
         "--build-dir", f"{VOL_STORES}/{spec['store_dir']}"],
        cwd=src, capture_output=True, text=True)
    return r.stdout + r.stderr


# ---------------------------------------------------------------------------
# bench: GPU TPS run — clone, build, generate, evidence -> volume
# ---------------------------------------------------------------------------
def _bench_impl(
    gpu_name: str,
    model: str = "dsv4-flash",
    n_tokens: int = 64,
    prompt: str = "",
    host_gib: float = 0.0,        # 0 -> GPU_SPECS default
    budget_mib: int = 0,          # 0 -> GPU_SPECS default
    cache_reset: str = "cold",
    local_store: bool = True,     # stage store to instance NVMe first
    pinned: str = "",
    run_id: str = "",
    ignore_eos: bool = False,     # decode the full n_tokens past EOS (b=8 scoring)
    cohort: str = "",             # e.g. "0-7": one lockstep cohort (b=8)
    prompts_file: str = "",       # repo-relative JSON list, e.g. dee.cpp/aws/phase6/prompts_phase4_8.json
    ssd_probe: bool = True,       # fio B_SSD probe (frozen section 0 prerequisite 5)
    container_mib: int = 0,       # 0 -> GPU_SPECS memory; bench_*_h256 pass HOST_256_MIB
    configs: str = "",            # JSON list of per-config overrides (CONFIG_KEYS); one
                                  # container runs them in order, each -> runs/<run_id>-<tag>
) -> dict:
        import concurrent.futures
        import json
        import os
        import shutil
        import subprocess
        import sys
        import threading
        import time
        from pathlib import Path

        gpu_spec, cuda_archs, dfl_budget_mib, _mem, dfl_host_gib = \
            GPU_SPECS[gpu_name]
        gpu_price = GPU_PRICE_PER_S[gpu_name]
        n_gpus = int(gpu_spec.split(":")[1]) if ":" in gpu_spec else 1
        spec = MODELS[model]
        budget_mib = budget_mib or dfl_budget_mib
        host_gib = host_gib or dfl_host_gib
        run_id = run_id or f"{model}-{gpu_name}-{int(time.time())}"
        src_root = Path("/tmp/dsv4-native-src")
        dee = src_root / "dee.cpp"
        build = dee / "build-modal"
        ev_root = Path(VOL_EVID)
        run_ev = ev_root / "runs" / run_id
        run_ev.mkdir(parents=True, exist_ok=True)
        log_path = run_ev / "bench.log"

        summary = {"run_id": run_id, "model": model, "gpu": gpu_name,
                   "platform": "modal", "gpu_price_s": gpu_price,
                   "n_gpus": n_gpus, "host_gib_requested": host_gib,
                   "host_gib_effective": host_gib, "host_basis": "TOTAL",
                   "budget_mib": budget_mib, "n_tokens": n_tokens,
                   "cache_reset": cache_reset, "ignore_eos": bool(ignore_eos),
                   "cohort": cohort or None, "verdict": "UNKNOWN",
                   "container_mib": container_mib or _mem, "cpu_cores": 8}

        def log(line: str) -> None:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[{ts}] {line}\n")
            print(line, flush=True)

        t_start = time.monotonic()
        # Files at the volume root may be leftovers from prior runs; only
        # collect files created/changed after this run started.
        preexisting = {f.name: f.stat().st_mtime
                       for f in ev_root.iterdir() if f.is_file()}
        Path("/kaggle/temp").mkdir(parents=True, exist_ok=True)

        def cost_sofar() -> float:
            return round((time.monotonic() - t_start) * gpu_price, 3)

        try:
            # ---- 0. environment -------------------------------------------------
            # Host budget is the pre-registered row: never clamp, refuse instead
            # (12 GiB reserve, same ASSUMPTION as aws/phase6/config.py).
            ram_gib = (container_mib or _mem) / 1024.0
            summary["container_ram_gib"] = ram_gib
            if host_gib + 12.0 > ram_gib:
                raise RuntimeError(
                    f"host budget {host_gib} GiB + 12 GiB reserve exceeds the "
                    f"{ram_gib:.0f} GiB container; raise GPU_SPECS memory first")
            # Validate every config before anything bills past this point.
            cfg_list = _parse_configs(configs, host_gib, ram_gib)
            nvidia = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            log(f"[env] gpu: {nvidia}")
            free_tmp = shutil.disk_usage("/tmp").free / (1 << 30)
            log(f"[env] /tmp free: {free_tmp:.0f} GiB")
            summary["nvidia"] = nvidia

            # ---- 1. clone + build ----------------------------------------------
            log(f"[build] clone {BRANCH} @ {pinned or PINNED_COMMIT or 'HEAD'}")
            subprocess.run(_sh(_clone(pinned)), shell=True, check=True)
            head = subprocess.check_output(
                ["git", "-C", str(src_root), "rev-parse", "HEAD"],
                text=True).strip()
            summary["commit"] = head

            cmake = [
                "cmake", "-S", str(dee), "-B", str(build),
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                "-DDEE_CUDA=ON", "-DDEE_BUILD_TESTS=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            log(f"[build] {' '.join(cmake)}")
            subprocess.run(cmake, check=True, capture_output=True, text=True,
                           timeout=900)
            subprocess.run(
                ["cmake", "--build", str(build), "--target", "dee_core",
                 "-j", "8"], check=True, capture_output=True, text=True,
                timeout=3600)
            subprocess.run(
                ["cmake", "--build", str(build), "--target",
                 "test_dee4_segmented", "-j", "8"],
                check=True, capture_output=True, text=True, timeout=1200)
            ct = subprocess.run(
                ["ctest", "--test-dir", str(build), "-R",
                 "^test_dee4_segmented$", "--output-on-failure"],
                capture_output=True, text=True, timeout=600)
            summary["store_reader_test"] = ct.returncode
            if ct.returncode != 0:
                raise RuntimeError(
                    f"test_dee4_segmented failed: {ct.stdout[-500:]}")
            subprocess.run(
                [sys.executable, "pydee/setup.py", "build_ext", "--inplace"],
                check=True, capture_output=True, text=True, timeout=1800,
                cwd=str(dee), env={**os.environ,
                                   "DEE_BUILD_DIR": str(build)})
            log(f"[build] done in {cost_sofar():+0.2f}$ gpu-time so far")

            # ---- 2. checkpoint mount -------------------------------------------
            ckpt_dir = Path(VOL_MODELS) / spec["ckpt_dir"]
            if not ckpt_dir.is_dir():
                raise RuntimeError(f"checkpoint missing: {ckpt_dir} "
                                   "(run stage_ckpt first)")
            kaggle_input = Path("/kaggle/input")
            kaggle_input.mkdir(parents=True, exist_ok=True)
            link = kaggle_input / spec["dataset_mount"]
            if not link.exists():
                link.symlink_to(ckpt_dir, target_is_directory=True)
            log(f"[ckpt] {link} -> {ckpt_dir}")

            # ---- 3. store: local NVMe stage or volume-direct --------------------
            store_vol = Path(VOL_STORES) / spec["store_dir"]
            meta = store_vol / "metadata.json"
            if not meta.is_file():
                raise RuntimeError(f"store missing: {meta} "
                                   "(run build_store first)")
            store_bytes = sum(
                int(s["bytes"]) for s in
                json.loads(meta.read_text()).get("segments", []))
            store_gib = store_bytes / (1 << 30)
            store_path = str(store_vol)
            if local_store:
                free = shutil.disk_usage("/tmp").free
                if free > store_bytes * 1.1:
                    log(f"[store] staging {store_gib:.0f} GiB -> /tmp "
                        "(local NVMe)")
                    t0 = time.monotonic()
                    local = Path("/tmp/dee4-store")
                    seal = _stage_and_seal(store_vol, local,
                                           json.loads(meta.read_text()), log)
                    summary["store_seal"] = seal
                    store_path = str(local)
                    dt = time.monotonic() - t0
                    log(f"[store] staged in {dt:.0f}s "
                        f"({store_gib / dt * 1.024:.2f} GiB/s)")
                    summary["store_stage_s"] = round(dt, 1)
                else:
                    log(f"[store] /tmp free {free / (1 << 30):.0f} GiB < "
                        f"{store_gib:.0f} GiB — serving from volume")
                    summary["store_on_volume"] = True
            summary["store_gib"] = round(store_gib, 1)

            # Shared pure helpers from the AWS port (stdlib only, same probe
            # definition: 13,369,344 B blocks, O_DIRECT, libaio, QD6 x 3 jobs).
            sys.path.insert(0, str(dee / "aws" / "phase6"))
            import bench_runner as br  # noqa: PLC0415

            # ---- 3b. SSD probe: measured B_SSD ---------------------------------
            if ssd_probe:
                segs = sorted(json.loads(meta.read_text()).get("segments", []),
                              key=lambda s_: int(s_["bytes"]), reverse=True)
                # Read across every segment file: a 6 GiB slice of one file staged
                # minutes earlier sits in page cache, and run r3 read 53.8 GiB/s.
                target = ":".join(str(Path(store_path) / sg["file"]) for sg in segs)
                results, direct = {}, True
                for rw in ("randread", "read"):
                    out_json = run_ev / f"fio-{rw}.json"
                    cmd = br.fio_command(rw=rw, path=target,
                                         size_gib=max(br.PROBE_SIZE_GIB, store_gib),
                                         out_json=out_json)
                    log("+ " + " ".join(cmd))
                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=br.PROBE_RUNTIME_S * 4 + 120)
                    if r.returncode != 0 and direct:
                        # Some container filesystems refuse O_DIRECT; record it.
                        log(f"[ssd] O_DIRECT refused ({r.stderr[-200:]!r}); buffered")
                        direct = False
                    if not direct:
                        cmd = [c if c != "--direct=1" else "--direct=0" for c in cmd]
                        subprocess.run(cmd, check=True, timeout=br.PROBE_RUNTIME_S * 4 + 120)
                    results[rw] = br.parse_fio_bandwidth_gib_s(
                        json.loads(out_json.read_text()))
                # No single cloud NVMe reads 13 MiB random blocks above this; a
                # higher reading is page cache / a virtualized FS, not the disk.
                suspect = results["randread"] > SSD_PLAUSIBLE_MAX_GIB_S
                if suspect:
                    log(f"[ssd] SUSPECT: {results['randread']} GiB/s > "
                        f"{SSD_PLAUSIBLE_MAX_GIB_S} GiB/s plausible NVMe ceiling")
                summary.update({
                    "ssd_probe_suspect": suspect,
                    "measured_b_ssd_gib_s": results["randread"],
                    "measured_b_ssd_seq_gib_s": results["read"],
                    "ssd_probe": {"file": str(target), "block_bytes": br.PROBE_RECORD_BYTES,
                                  "iodepth": br.PROBE_IODEPTH, "numjobs": br.PROBE_NUMJOBS,
                                  "direct": direct, "runtime_s": br.PROBE_RUNTIME_S,
                                  "on_volume": bool(summary.get("store_on_volume"))},
                })
                log(f"[ssd] randread {results['randread']} GiB/s, "
                    f"seq {results['read']} GiB/s (direct={direct})")
                if results["randread"] <= 0:
                    raise RuntimeError("SSD probe produced no bandwidth; unscorable")

            seg_files = [Path(store_path) / sg["file"] for sg in
                         json.loads(meta.read_text()).get("segments", [])]

            # ---- 4. one or more runner configs in this container ----------------
            # Build, store staging + seal and the B_SSD probe are paid once; each
            # config is a fresh runner process (cold engine caches), and the
            # store's page cache is dropped (best effort) before each one.
            base_summary = dict(summary)
            base_run_id, base_ev = run_id, run_ev
            outcomes: list = []
            for cfg in cfg_list:
              t_cfg = time.monotonic()
              try:
                if cfg is not None:
                    host_gib = float(cfg.get("host_gib", base_summary["host_gib_requested"]))
                    n_tokens = int(cfg.get("n_tokens", base_summary["n_tokens"]))
                    ignore_eos = bool(cfg.get("ignore_eos", False))
                    cohort = str(cfg.get("cohort", ""))
                    prompts_file = str(cfg.get("prompts_file", ""))
                    prompt = str(cfg.get("prompt", ""))
                    run_id = f"{base_run_id}-{cfg['tag']}"
                    run_ev = ev_root / "runs" / run_id
                    run_ev.mkdir(parents=True, exist_ok=True)
                    for fj in base_ev.glob("fio-*.json"):
                        shutil.copy2(fj, run_ev / fj.name)
                    log_path = run_ev / "bench.log"
                    summary = {**base_summary, "run_id": run_id,
                               "parent_run_id": base_run_id,
                               "host_gib_requested": host_gib,
                               "host_gib_effective": host_gib, "n_tokens": n_tokens,
                               "ignore_eos": ignore_eos, "cohort": cohort or None,
                               "verdict": "UNKNOWN", "config": cfg}
                summary["page_cache_drop"] = _drop_page_cache(seg_files, log)
                preexisting = {f.name: f.stat().st_mtime
                               for f in ev_root.iterdir() if f.is_file()}

                # ---- 4. run the native generate driver ------------------------------
                env = dict(os.environ)
                pack_per_gpu = host_gib / n_gpus
                env.update({
                    "NATIVE_SOURCE_TREE": str(src_root),
                    "NATIVE_RUN_ID": run_id,
                    "NATIVE_ARM_ID": f"p6-{gpu_name}",
                    # validated campaign modes (runner defaults are legacy:
                    # rank_priority eviction + fp16 cache) — set explicitly.
                    "NATIVE_EXPERT_STORE": "dee4_segmented",
                    "NATIVE_DEE4_SEGMENTED_STORE": store_path,
                    "NATIVE_CACHE_DTYPE": "fp4",
                    "NATIVE_EVICTION_POLICY": "lru",
                    "NATIVE_HOST_CACHE_MODE": "lru",
                    "NATIVE_SOURCE_READ_LANES": "4",
                    "NATIVE_SOURCE_READ_QUEUE_DEPTH": "6",
                    "NATIVE_N_TOKENS": str(n_tokens),
                    "NATIVE_LRU_TOTAL_CAP_GIB": str(host_gib),
                    "NATIVE_BUDGET_BYTES": str(budget_mib << 20),
                    "NATIVE_HOST_PACK_GPU0_BYTES": str(int(pack_per_gpu * (1 << 30))),
                    "NATIVE_HOST_PACK_GPU1_BYTES": str(int(pack_per_gpu * (1 << 30))),
                    "NATIVE_CACHE_RESET": cache_reset,
                    "NATIVE_FORCE_TMP": "1",
                    "NATIVE_TRACE_REQUESTS": "1",
                    "NATIVE_PROFILE": "1",
                    "NATIVE_MODEL_CKPT": str(ckpt_dir),
                    "NATIVE_HEADERS_DIR": str(store_vol / "headers"),
                    "DEE_BUILD_DIR": str(build),
                    "NATIVE_CUDA_ARCHS": cuda_archs,
                })
                if (summary.get("store_seal") or {}).get("verified"):
                    # every segment was sha256-checked while staging (same seal
                    # the engine would recompute); skip the ~17 min re-hash
                    env["DEE4_STORE_SKIP_SEAL"] = "1"
                if not cohort:
                    # b=1 diagnostics: per-(step, layer) tensor hashes + router
                    # top-8 scores, so a token mismatch names its layer and gap.
                    env["NATIVE_CAPTURE_JOURNAL"] = "1"
                    env["NATIVE_ROUTE_WEIGHT_JOURNAL"] = "1"
                if n_gpus == 1:
                    env["NATIVE_SINGLE_GPU"] = "1"
                    # else the runner caps one-GPU VRAM at min(2 x budget, 4 GiB)
                    env["NATIVE_SINGLE_GPU_BUDGET_BYTES"] = str(budget_mib << 20)
                if ignore_eos:
                    env["NATIVE_IGNORE_EOS"] = "1"
                if cohort:
                    if not prompts_file:
                        raise ValueError("cohort mode needs prompts_file (fail-closed)")
                    prompts = json.loads((src_root / prompts_file).read_text("utf-8"))
                    groups = br.parse_cohort_spec(cohort)
                    br.check_cohort_against_prompts(groups, len(prompts))
                    env["NATIVE_COHORT_JSON"] = br.cohort_json(groups)
                    env["NATIVE_PROMPTS_JSON"] = json.dumps(prompts)
                elif prompt:
                    env["NATIVE_PROMPT"] = prompt
                log(f"[run] n_tokens={n_tokens} host_gib={host_gib} "
                    f"budget_mib={budget_mib} cache={cache_reset} "
                    f"store={'local' if store_path.startswith('/tmp') else 'vol'}")

                runner = src_root / spec["runner"]
                proc = subprocess.Popen(
                    [sys.executable, str(runner)], env=env, cwd=str(ev_root),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                # Watchdog: a hung runner must not bill the container to its
                # timeout. From r4 (2.4 s/token at b=1, no seal): b=1 16 tokens
                # ~5 min; b=8 65 tokens est. 20-25 min. Past the limit: kill,
                # record the failure, go on to the next config.
                limit_s = CONFIG_TIMEOUT_B8_S if cohort else CONFIG_TIMEOUT_B1_S
                killed = threading.Event()

                def _kill() -> None:
                    killed.set()
                    log(f"[watchdog] runner exceeded {limit_s}s; killing")
                    proc.kill()
                watchdog = threading.Timer(limit_s, _kill)
                watchdog.daemon = True
                watchdog.start()

                # Heartbeat: GPU util + proc RSS + bounded volume commit so a
                # dark run can never happen (CACHE1c lesson).
                stop = threading.Event()
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

                def _hb() -> None:
                    while not stop.wait(300):
                        try:
                            alive = proc.poll() is None
                            snap = subprocess.run(
                                ["nvidia-smi", "--query-gpu=utilization.gpu,"
                                 "memory.used", "--format=csv,noheader"],
                                capture_output=True, text=True,
                                timeout=15).stdout.strip()
                            log(f"[hb] alive={alive} gpu=[{snap}] "
                                f"cost=${cost_sofar():.2f}")
                            fut = pool.submit(vol_evid.commit)
                            try:
                                fut.result(timeout=90)
                            except Exception:
                                log("[hb] volume commit skipped (timeout)")
                        except Exception as exc:  # noqa: BLE001
                            log(f"[hb] error (continuing): {exc!r}")

                threading.Thread(target=_hb, daemon=True).start()

                tail: list[str] = []
                all_lines: list[str] = []
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    log(f"[runner] {line}")
                    all_lines.append(line)
                    tail.append(line)
                    if len(tail) > 80:
                        tail.pop(0)
                code = proc.wait()
                watchdog.cancel()
                stop.set()
                if killed.is_set():
                    raise RuntimeError(f"runner killed by watchdog after {limit_s}s")
                wall_s = round(time.monotonic() - t_start, 1)
                log(f"[run] exit={code} wall={wall_s}s est_cost=${cost_sofar()}")
                summary.update({"exit_code": code, "wall_s": wall_s,
                                "est_gpu_cost_usd": cost_sofar(),
                                "log_tail": tail[-50:]})
                verdict = "UNKNOWN"
                for ln in tail:
                    if ln.startswith("VERDICT:"):
                        verdict = ln.split(":", 1)[1].strip()
                summary["verdict"] = verdict
                knobs = br.host_pack_effective_bytes(all_lines)
                summary["host_pack_effective_bytes"] = knobs
                honored = br.host_budget_honored(knobs, host_gib, n_gpus)
                summary["host_budget_honored"] = honored
                # The runner exits 0 on failure by design (Kaggle evidence capture);
                # its FATAL line is the real outcome.
                fatal = next((ln for ln in all_lines if "] FATAL" in ln
                              or ln.startswith("FATAL")), None)
                summary["runner_fatal"] = fatal is not None
                if fatal is not None:
                    errs = [ln for ln in all_lines if "Error" in ln]
                    summary["verdict"] = "RUNNER_FATAL: " + (errs[-1] if errs else fatal)[-300:]
                elif not honored:
                    summary["verdict"] = (f"HOST_BUDGET_NOT_HONORED: {knobs} != "
                                          f"{host_gib} GiB / {n_gpus} GPU(s)")

                # ---- 5. collect evidence -------------------------------------------
                for f in ev_root.iterdir():
                    if (f.is_file() and f.name not in ("bench.log",)
                            and f.name.endswith((".json", ".jsonl", ".log"))
                            and f.stat().st_mtime > preexisting.get(f.name, 0)):
                        try:
                            shutil.copy2(f, run_ev / f.name)
                        except OSError:
                            pass
                summary["config_wall_s"] = round(time.monotonic() - t_cfg, 1)
                (run_ev / "summary.json").write_text(
                    json.dumps(summary, indent=2))
              except Exception as exc:  # noqa: BLE001
                if cfg is None:
                    raise
                # one config failing must not cost the rest of the container
                summary["verdict"] = f"{type(exc).__name__}: {exc}"
                log(f"[fatal:{cfg['tag']}] {exc!r}")
                (run_ev / "summary.json").write_text(json.dumps(summary, indent=2))
              finally:
                vol_evid.commit()
              outcomes.append({"run_id": run_id, "verdict": summary.get("verdict")})
            if cfg_list[0] is None:
                return summary
            log_path = base_ev / "bench.log"
            log(f"[configs] {json.dumps(outcomes)}")
            return {"run_id": base_run_id, "configs": outcomes}
        except Exception as exc:  # noqa: BLE001
            summary["verdict"] = f"{type(exc).__name__}: {exc}"
            log(f"[fatal] {exc!r}")
            # Build steps run with capture_output=True; keep what they printed,
            # or a failed build is undiagnosable (run p6m-l4-h64-b1, 2026-11-04).
            if isinstance(exc, subprocess.CalledProcessError):
                for name in ("stdout", "stderr"):
                    text = getattr(exc, name, None) or ""
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", "replace")
                    if text:
                        summary[f"failed_step_{name}_tail"] = text[-8000:]
                        for line in text.splitlines()[-60:]:
                            log(f"[fatal:{name}] {line}")
            (run_ev / "summary.json").write_text(json.dumps(summary, indent=2))
            raise
        finally:
            vol_evid.commit()

# Cost guards (L4 all-in ~$1.94/h): no config may run away, and no container
# may outlive a few configs' worth of time.
CONFIG_TIMEOUT_B1_S = 20 * 60
CONFIG_TIMEOUT_B8_S = 45 * 60
FUNCTION_TIMEOUT_S = 150 * 60
CONFIG_KEYS = {"tag", "host_gib", "n_tokens", "ignore_eos", "cohort",
               "prompts_file", "prompt"}


@app.function(
    image=image_gpu,          # the exact GPU image, on CPU: nvcc needs no GPU
    cpu=8,
    memory=16384,
    timeout=3600,
    volumes={VOL_MODELS: vol_models, VOL_STORES: vol_stores},
)
def preflight(gpu: str = "L4", model: str = "dsv4-flash", configs: str = "",
              pinned: str = "") -> dict:
    """Everything a bench run can fail on before inference, for ~$0.10 of CPU.

    Builds the image, clones the same commit, runs the harness build AND the
    runner's own build (its cmake/pydee steps, arch from GPU_SPECS), imports
    pydee and torch, and checks the store table, checkpoint, prompts and
    configs. Runs r1-r3 each paid a GPU container to find one of these.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    out: dict = {"gpu": gpu, "model": model, "checks": {}}
    chk = out["checks"]

    def step(name, fn):
        try:
            chk[name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                detail += " | " + ((exc.stderr or "") + (exc.stdout or ""))[-3000:]
            chk[name] = {"ok": False, "detail": detail}
        print(f"[preflight] {name}: {'OK' if chk[name]['ok'] else 'FAIL'} "
              f"{str(chk[name]['detail'])[:300]}", flush=True)

    def run(cmd, **kw):
        return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

    _spec, archs, budget, mem, _host = GPU_SPECS[gpu]
    spec = MODELS[model]
    src = Path("/tmp/dsv4-native-src")
    dee = src / "dee.cpp"
    step("python_torch", lambda: run([sys.executable, "-c",
         "import sys, torch, transformers, numpy, setuptools, pybind11; "
         "print(sys.version.split()[0], torch.__version__, torch.version.cuda, "
         "transformers.__version__, numpy.__version__)"]).stdout.strip())
    def pinned_torch():
        got = str(chk["python_torch"]["detail"])
        if not got.startswith("3.12") or " 2.10.0+cu128 12.8 " not in got:
            raise RuntimeError(f"want python 3.12 + torch 2.10.0+cu128, got {got}")
        return got
    step("torch_pinned", pinned_torch)
    step("fio", lambda: run(["fio", "--version"]).stdout.strip())
    step("clone", lambda: subprocess.run(_sh(_clone(pinned)), shell=True, check=True,
         capture_output=True, text=True).stdout.strip()[-40:])
    step("harness_build", lambda: [run(c, cwd=str(dee), timeout=3000) for c in (
        ["cmake", "-S", ".", "-B", "build-modal", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
         "-DDEE_CUDA=ON", "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", "build-modal", "--target", "dee_core", "-j", "8"],
        ["cmake", "--build", "build-modal", "--target", "test_dee4_segmented", "-j", "8"],
        [sys.executable, "pydee/setup.py", "build_ext", "--inplace"])] and "ok")
    # the runner's own build: same commands it runs (build-kaggle, its arch env)
    env_r = {**os.environ, "DEE_BUILD_DIR": str(dee / "build-kaggle")}
    step("runner_build", lambda: [run(c, cwd=str(dee), timeout=3000, env=env_r) for c in (
        ["cmake", "-S", ".", "-B", "build-kaggle", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
         "-DDEE_CUDA=ON", "-DDEE_BUILD_TESTS=ON", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", "build-kaggle", "--target", "dee_core", "-j", "8"],
        ["cmake", "--build", "build-kaggle", "--target", "test_deepseek_v4_fp4_cuda",
         "-j", "8"],
        ["cmake", "--build", "build-kaggle", "--target", "test_deepseek_v4_fp4_expert",
         "-j", "8"],
        [sys.executable, "pydee/setup.py", "build_ext", "--inplace"])] and "ok")
    def pydee_import():
        r = subprocess.run([sys.executable, "-c", "import pydee; print(pydee.__file__)"],
                           cwd=str(dee), capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
        if "libcuda" in r.stderr:   # no driver in a CPU container: link is still checked
            return "import needs the GPU driver (expected on CPU); extension built"
        raise RuntimeError(r.stderr[-1500:])
    step("pydee_import", pydee_import)
    def runner_src():
        txt = (src / spec["runner"]).read_text()
        need = ("NATIVE_CUDA_ARCHS", "NATIVE_SINGLE_GPU_BUDGET_BYTES", "router_top8_values")
        missing = [k for k in need if k not in txt]
        if missing:
            raise RuntimeError(f"runner at this commit lacks {missing}")
        return "ok"
    step("runner_source", runner_src)

    def store():
        meta = json.loads((Path(VOL_STORES) / spec["store_dir"] / "metadata.json").read_text())
        segs = meta.get("segments", [])
        bad = [sg["file"] for sg in segs if len(str(sg.get("sha256", ""))) != 64]
        sizes = [(Path(VOL_STORES) / spec["store_dir"] / sg["file"]).stat().st_size
                 == int(sg["bytes"]) for sg in segs]
        if not segs or bad or not all(sizes):
            raise RuntimeError(f"segments={len(segs)} no_sha={bad[:3]} "
                               f"size_ok={sum(sizes)}/{len(segs)}")
        return f"{len(segs)} segments, {sum(int(x['bytes']) for x in segs) / 2**30:.1f} GiB"
    step("store_table", store)
    def ckpt():
        n = len(list((Path(VOL_MODELS) / spec["ckpt_dir"]).rglob("*.safetensors")))
        if n == 0:
            raise RuntimeError("no .safetensors under the checkpoint dir")
        return f"{n} shards"
    step("checkpoint", ckpt)

    def cfgs():
        parsed = _parse_configs(configs, _host, mem / 1024.0)
        for c in parsed:
            if c and c.get("prompts_file"):
                pr = json.loads((src / c["prompts_file"]).read_text("utf-8"))
                if not isinstance(pr, list) or not pr:
                    raise ValueError(f"{c['prompts_file']}: not a prompt list")
        return f"{len(parsed)} config(s)"
    step("configs", cfgs)
    out["ok"] = all(v["ok"] for v in chk.values())
    print(json.dumps({"preflight_ok": out["ok"]}), flush=True)
    return out


def _parse_configs(configs: str, host_gib: float, ram_gib: float) -> list:
    """[None] for a single run, else validated config dicts (fail-closed)."""
    import json
    if not configs:
        return [None]
    cfgs = json.loads(configs)
    if not isinstance(cfgs, list) or not cfgs:
        raise ValueError("configs must be a non-empty JSON list")
    tags = set()
    for c in cfgs:
        unknown = set(c) - CONFIG_KEYS
        if unknown or not c.get("tag") or c["tag"] in tags:
            raise ValueError(f"bad config {c}: unknown={sorted(unknown)} (tag required, unique)")
        tags.add(c["tag"])
        if float(c.get("host_gib", host_gib)) + 12.0 > ram_gib:
            raise RuntimeError(f"config {c['tag']}: host + 12 GiB > {ram_gib:.0f} GiB container")
        if c.get("cohort") and not c.get("prompts_file"):
            raise ValueError(f"config {c['tag']}: cohort needs prompts_file")
        if c.get("cohort") and int(c.get("n_tokens", 0)) < 65:
            raise ValueError(f"config {c['tag']}: b=8 scoring needs n_tokens >= 65")
        if c.get("cohort") and not c.get("ignore_eos"):
            raise ValueError(f"config {c['tag']}: b=8 scoring needs ignore_eos")
    return cfgs


def _stage_and_seal(src, dst, meta: dict, log) -> dict:
    """Copy the store to local disk, sha256-checking every segment on the way.

    The same per-segment seal the engine recomputes at open (DEE4 metadata
    `segments[].sha256`), computed on the bytes as they are written, so the
    runner can skip its ~17 min re-hash (DEE4_STORE_SKIP_SEAL). Each file's
    pages are fsync'd and fadvise(DONTNEED)'d after writing, so staging does
    not leave the store in page cache. A mismatch raises: never serve it.
    """
    import concurrent.futures
    import hashlib
    import os
    import shutil
    import time
    from pathlib import Path

    src, dst = Path(src), Path(dst)
    want = {sg["file"]: sg.get("sha256") for sg in meta.get("segments", [])}
    files = [f for f in src.rglob("*") if f.is_file()]
    t0 = time.monotonic()

    def one(f):
        rel = f.relative_to(src).as_posix()
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if rel not in want:
            shutil.copy2(f, out)
            return rel, None, f.stat().st_size
        h = hashlib.sha256()
        n = 0
        with open(f, "rb", buffering=0) as fi, open(out, "wb", buffering=0) as fo:
            while True:
                buf = fi.read(16 << 20)
                if not buf:
                    break
                h.update(buf)
                fo.write(buf)
                n += len(buf)
            fo.flush()
            os.fsync(fo.fileno())
            for fd in (fi.fileno(), fo.fileno()):
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        return rel, h.hexdigest(), n

    bad, total, sealed = [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for rel, digest, n in pool.map(one, files):
            total += n
            if digest is not None:
                sealed += 1
                if want[rel] and digest != want[rel]:
                    bad.append(rel)
    missing = sorted(set(want) - {f.relative_to(src).as_posix() for f in files})
    dt = time.monotonic() - t0
    log(f"[store] staged + sealed {sealed}/{len(want)} segments, "
        f"{total / (1 << 30):.1f} GiB in {dt:.0f}s ({total / (1 << 30) / max(dt, 1e-6):.2f} GiB/s)")
    if bad or missing:
        raise RuntimeError(f"store seal failed: sha256 mismatch {bad[:3]} missing {missing[:3]}")
    no_seal = [k for k, v in want.items() if not v]
    return {"verified": not no_seal, "segments": sealed, "unsealed_in_metadata": no_seal[:3],
            "stage_s": round(dt, 1), "gib": round(total / (1 << 30), 1)}


def _drop_page_cache(files, log) -> dict:
    """Best-effort: fadvise(DONTNEED) the store, and report whether it took.

    Under gVisor this may be a no-op; the Cached delta in the summary says so,
    so a "cold" config is never assumed cold without evidence.
    """
    import os

    def cached_gib():
        try:
            for line in open("/proc/meminfo", encoding="ascii"):
                if line.startswith("Cached:"):
                    return round(int(line.split()[1]) / (1 << 20), 2)
        except OSError:
            pass
        return None

    before = cached_gib()
    for f in files:
        try:
            fd = os.open(f, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except (AttributeError, OSError):
            pass
    after = cached_gib()
    log(f"[cache] drop store pages: Cached {before} -> {after} GiB")
    return {"cached_gib_before": before, "cached_gib_after": after}


# 256 GiB host-tier cells need a bigger container than GPU_SPECS gives (host +
# reserve). Registered in theory/predecl_modal.py (HOST_256_CONTAINER_MIB) and
# priced there; run them through the bench_<GPU>_h256 functions below.
HOST_256_MIB = 294912
SSD_PLAUSIBLE_MAX_GIB_S = 16.0


def _bench_dec(gpu_name: str, memory_mib: int = 0):
    """The @app.function decorator for one GPU spec (module-scope targets)."""
    _spec, _archs, _budget, _mem, _host = GPU_SPECS[gpu_name]
    return app.function(
        image=image_gpu,
        gpu=_spec,
        cpu=8,
        memory=memory_mib or _mem,
        timeout=FUNCTION_TIMEOUT_S,
        volumes={VOL_MODELS: vol_models, VOL_STORES: vol_stores,
                 VOL_EVID: vol_evid},
    )


@_bench_dec("L4")
def bench_L4(model: str = "dsv4-flash", n_tokens: int = 64,
             prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
             cache_reset: str = "cold", local_store: bool = True,
             pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("L4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("2xL4")
def bench_2xL4(model: str = "dsv4-flash", n_tokens: int = 64,
               prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
               cache_reset: str = "cold", local_store: bool = True,
               pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("2xL4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("T4")
def bench_T4(model: str = "dsv4-flash", n_tokens: int = 64,
               prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
               cache_reset: str = "cold", local_store: bool = True,
               pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("T4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("2xT4")
def bench_2xT4(model: str = "dsv4-flash", n_tokens: int = 64,
               prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
               cache_reset: str = "cold", local_store: bool = True,
               pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("2xT4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("A10")
def bench_A10(model: str = "dsv4-flash", n_tokens: int = 64,
              prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
              cache_reset: str = "cold", local_store: bool = True,
              pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("A10", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("L40S")
def bench_L40S(model: str = "dsv4-flash", n_tokens: int = 64,
               prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
               cache_reset: str = "cold", local_store: bool = True,
               pinned: str = "", run_id: str = "",
             ignore_eos: bool = False,
             cohort: str = "", prompts_file: str = "",
             ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("L40S", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("RTXPRO6000")
def bench_RTXPRO6000(model: str = "dsv4-flash", n_tokens: int = 64,
                     prompt: str = "", host_gib: float = 0.0,
                     budget_mib: int = 0, cache_reset: str = "cold",
                     local_store: bool = True, pinned: str = "",
                     run_id: str = "", ignore_eos: bool = False,
                     cohort: str = "", prompts_file: str = "",
                     ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("RTXPRO6000", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, 0, configs)


@_bench_dec("L4", HOST_256_MIB)
def bench_L4_h256(model: str = "dsv4-flash", n_tokens: int = 64,
                   prompt: str = "", host_gib: float = 256.0, budget_mib: int = 0,
                   cache_reset: str = "cold", local_store: bool = True,
                   pinned: str = "", run_id: str = "", ignore_eos: bool = False,
                   cohort: str = "", prompts_file: str = "",
                   ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("L4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, HOST_256_MIB,
                       configs)


@_bench_dec("2xL4", HOST_256_MIB)
def bench_2xL4_h256(model: str = "dsv4-flash", n_tokens: int = 64,
                   prompt: str = "", host_gib: float = 256.0, budget_mib: int = 0,
                   cache_reset: str = "cold", local_store: bool = True,
                   pinned: str = "", run_id: str = "", ignore_eos: bool = False,
                   cohort: str = "", prompts_file: str = "",
                   ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("2xL4", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, HOST_256_MIB,
                       configs)


@_bench_dec("A10", HOST_256_MIB)
def bench_A10_h256(model: str = "dsv4-flash", n_tokens: int = 64,
                   prompt: str = "", host_gib: float = 256.0, budget_mib: int = 0,
                   cache_reset: str = "cold", local_store: bool = True,
                   pinned: str = "", run_id: str = "", ignore_eos: bool = False,
                   cohort: str = "", prompts_file: str = "",
                   ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("A10", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, HOST_256_MIB,
                       configs)


@_bench_dec("L40S", HOST_256_MIB)
def bench_L40S_h256(model: str = "dsv4-flash", n_tokens: int = 64,
                   prompt: str = "", host_gib: float = 256.0, budget_mib: int = 0,
                   cache_reset: str = "cold", local_store: bool = True,
                   pinned: str = "", run_id: str = "", ignore_eos: bool = False,
                   cohort: str = "", prompts_file: str = "",
                   ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("L40S", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, HOST_256_MIB,
                       configs)


@_bench_dec("RTXPRO6000", HOST_256_MIB)
def bench_RTXPRO6000_h256(model: str = "dsv4-flash", n_tokens: int = 64,
                   prompt: str = "", host_gib: float = 256.0, budget_mib: int = 0,
                   cache_reset: str = "cold", local_store: bool = True,
                   pinned: str = "", run_id: str = "", ignore_eos: bool = False,
                   cohort: str = "", prompts_file: str = "",
                   ssd_probe: bool = True, configs: str = "") -> dict:
    return _bench_impl("RTXPRO6000", model, n_tokens, prompt, host_gib,
                       budget_mib, cache_reset, local_store, pinned, run_id,
                       ignore_eos, cohort, prompts_file, ssd_probe, HOST_256_MIB,
                       configs)


@app.local_entrypoint()
def main() -> None:
    print("dee-p6 functions: stage_ckpt, build_store, store_status, "
          + ", ".join(f"bench_{g}" for g in GPU_SPECS)
          + " (+ bench_<GPU>_h256 for 256 GiB host cells)")
    print("Models:", ", ".join(MODELS))
