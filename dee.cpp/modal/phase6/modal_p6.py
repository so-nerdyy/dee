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
BRANCH = "research/phase6-modal"
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
}
GPU_PRICE_PER_S = {
    "L4": 0.000222, "2xL4": 2 * 0.000222, "A10": 0.000306,
    "L40S": 0.000542, "RTXPRO6000": 0.000842,
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
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "cmake", "build-essential", "zlib1g-dev")
    .pip_install(
        "torch", "transformers", "safetensors", "numpy",
        "huggingface_hub", "hf_transfer", "pybind11", "requests",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
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
                   "gpu_price_s": gpu_price}

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
                    shutil.copytree(store_vol, local)
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
                "DEE_BUILD_DIR": str(build),
            })
            if n_gpus == 1:
                env["NATIVE_SINGLE_GPU"] = "1"
            if prompt:
                env["NATIVE_PROMPT"] = prompt
            log(f"[run] n_tokens={n_tokens} host_gib={host_gib} "
                f"budget_mib={budget_mib} cache={cache_reset} "
                f"store={'local' if store_path.startswith('/tmp') else 'vol'}")

            runner = src_root / spec["runner"]
            proc = subprocess.Popen(
                [sys.executable, str(runner)], env=env, cwd=str(ev_root),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)

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
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                log(f"[runner] {line}")
                tail.append(line)
                if len(tail) > 80:
                    tail.pop(0)
            code = proc.wait()
            stop.set()
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

            # ---- 5. collect evidence -------------------------------------------
            for f in ev_root.iterdir():
                if (f.is_file() and f.name not in ("bench.log",)
                        and f.name.endswith((".json", ".jsonl", ".log"))
                        and f.stat().st_mtime > preexisting.get(f.name, 0)):
                    try:
                        shutil.copy2(f, run_ev / f.name)
                    except OSError:
                        pass
            (run_ev / "summary.json").write_text(
                json.dumps(summary, indent=2))
            return summary
        except Exception as exc:  # noqa: BLE001
            summary["verdict"] = f"{type(exc).__name__}: {exc}"
            log(f"[fatal] {exc!r}")
            (run_ev / "summary.json").write_text(json.dumps(summary, indent=2))
            raise
        finally:
            vol_evid.commit()

def _bench_for(gpu_name: str):
    """Build the module-level wrapper Modal registers for one GPU spec.
    The explicit signature is required: Modal maps `modal run` CLI args
    onto function parameters by name."""
    _spec, _archs, _budget, _mem, _host = GPU_SPECS[gpu_name]

    @app.function(
        image=image_gpu,
        gpu=_spec,
        cpu=8,
        memory=_mem,
        timeout=6 * 3600,
        volumes={VOL_MODELS: vol_models, VOL_STORES: vol_stores,
                 VOL_EVID: vol_evid},
        name=f"bench_{gpu_name}",
        serialized=True,
    )
    def _fn(model: str = "dsv4-flash", n_tokens: int = 64,
            prompt: str = "", host_gib: float = 0.0, budget_mib: int = 0,
            cache_reset: str = "cold", local_store: bool = True,
            pinned: str = "", run_id: str = "") -> dict:
        return _bench_impl(
            gpu_name, model=model, n_tokens=n_tokens, prompt=prompt,
            host_gib=host_gib, budget_mib=budget_mib,
            cache_reset=cache_reset, local_store=local_store,
            pinned=pinned, run_id=run_id)

    return _fn


# bench_L4, bench_2xL4, bench_A10, bench_L40S, bench_RTXPRO6000
_globals = globals()
for _g in GPU_SPECS:
    _globals[f"bench_{_g}"] = _bench_for(_g)


@app.local_entrypoint()
def main() -> None:
    print("dee-p6 functions: stage_ckpt, build_store, store_status, "
          + ", ".join(f"bench_{g}" for g in GPU_SPECS))
    print("Models:", ", ".join(MODELS))
