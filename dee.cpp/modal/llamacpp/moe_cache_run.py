"""llama.cpp MoE cache: ClassLRU (upstream) vs DEE next-use, one bundled L4 run.

What it measures (all in one container, so the build and model load are paid once):
  1. exactness   test-llama-archs on CUDA, MoE archs: the cache config must match the CPU, and the new
                 check must find bit-identical logits for a cache that holds every expert and for next_use
  2. speed       llama-server -cmoe, policy x cache size, at 1, 4 and 8 parallel requests
  3. traces      LLAMA_MOE_CACHE_TRACE for every server, replayed afterwards on CPU through the patched
                 scripts/moe-cache-tune.py, to check the Python mirror against the C++ hit counts
  4. text        greedy output at 1 request must be identical across policies and cache sizes

Usage (from the dee repo root, on the laptop with the Modal CLI):
  modal run dee.cpp/modal/llamacpp/moe_cache_run.py::preflight      # CPU only, ~$0.05: build + models + checks
  modal run dee.cpp/modal/llamacpp/moe_cache_run.py::main           # preflight, then the L4 run, then the CPU replay
Evidence: volume dee-p6-evidence, folder llamacpp-moe-<run id>/.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import modal

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT = "374d90a"  # master on 2026-11-20, the patch is made against it
HERE = pathlib.Path(__file__).parent
PATCH = HERE / "next_use.patch"

RUN_ID = os.environ.get("RUN_ID", "r1")
GPU = "L4"
MEM_MIB = 65536
CPUS = 8
# all-in list price per second: L4 + 8 cores + 64 GiB RAM (same basis as modal/phase6/modal_p6.py)
PRICE_PER_S = 0.000222 + CPUS * 0.0000131 + (MEM_MIB / 1024) * 0.00000222
MAX_USD = float(os.environ.get("MAX_USD", "2.0"))
GPU_TIMEOUT_S = min(3600, int(MAX_USD / PRICE_PER_S))  # hard stop by Modal, ~$1.69/h

MODELS = {
    "qwen3-30b-a3b": ("unsloth/Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf"),
    "gpt-oss-20b": ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-mxfp4.gguf"),
}
POLICIES = ["classlru", "next_use"]
CACHE_FRACTIONS = [0.05, 0.10]  # of the expert bytes; next-use's sim gain is at 5-10%
CONCURRENCY = [1, 4, 8]
N_PREDICT = 128

PROMPTS = [
    "Explain how a refrigerator moves heat out of its inside, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its running time.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "A train leaves at 3:40 PM and travels 210 km at 84 km/h. When does it arrive? Show the work.",
    "Translate into Spanish and French: 'The library opens at nine and closes at six on weekdays.'",
    "Give three arguments for and three against a four-day school week.",
    "Write a haiku about debugging a program at 2 AM, then explain the imagery.",
    "What is the difference between TCP and UDP? Give one real use case for each.",
]

VOL_MODELS = "/vol/models"
VOL_EVID = "/vol/evidence"
vol_models = modal.Volume.from_name("dee-models", create_if_missing=True)
vol_evid = modal.Volume.from_name("dee-p6-evidence", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential", "libssl-dev")
    .pip_install("huggingface_hub", "hf_transfer", "requests", "gguf", "numpy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "CC": "gcc", "CXX": "g++"})
    .add_local_file(str(PATCH), "/opt/next_use.patch", copy=True)
    .run_commands(
        f"git clone {LLAMA_REPO} /opt/llama.cpp && cd /opt/llama.cpp && git checkout {LLAMA_COMMIT}",
        "sed -i 's/\\r$//' /opt/next_use.patch && cd /opt/llama.cpp && git apply --check /opt/next_use.patch && git apply /opt/next_use.patch",
        # CUDA builds without a GPU at image-build time (CPU price); the driver library comes from the stubs
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=ON "
        "-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined "
        "&& cmake --build build -j8 --target llama-server test-llama-archs",
    )
)

app = modal.App("dee-llamacpp-moe")


def _model_path(name: str) -> str:
    repo, fname = MODELS[name]
    return f"{VOL_MODELS}/gguf/{repo.replace('/', '__')}/{fname}"


def _expert_mib(path: str) -> float:
    import gguf
    r = gguf.GGUFReader(path)
    return sum(int(t.n_bytes) for t in r.tensors if "_exps" in t.name) / 2**20


@app.function(image=image, volumes={VOL_MODELS: vol_models}, cpu=4, memory=8192, timeout=3600)
def preflight() -> dict:
    """CPU only: the patched build exists, the models are on the volume, sizes are read. Downloads once."""
    from huggingface_hub import hf_hub_download

    out = {"ok": True, "models": {}}
    for b in ("llama-server", "test-llama-archs"):
        p = f"/opt/llama.cpp/build/bin/{b}"
        out[b] = os.path.exists(p)
        out["ok"] &= out[b]
    out["patch_applied"] = "next_use" in open("/opt/llama.cpp/src/llama-moe-cache.cpp").read()
    out["ok"] &= out["patch_applied"]
    for name, (repo, fname) in MODELS.items():
        dst = _model_path(name)
        try:
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                hf_hub_download(repo, fname, local_dir=os.path.dirname(dst))
                vol_models.commit()
            out["models"][name] = {"path": dst, "gib": round(os.path.getsize(dst) / 2**30, 2),
                                   "expert_mib": round(_expert_mib(dst), 1)}
        except Exception as e:  # a missing model is skipped, not fatal, as long as one remains
            out["models"][name] = {"error": repr(e)[:300]}
    out["ok"] &= any("error" not in m for m in out["models"].values())
    # the CUDA binaries need libcuda.so.1, which only a GPU container has, so they are not run here
    # (the CPU build of the same patch passes all 155 test-llama-archs rows, 2026-11-20)
    out["gpu_timeout_s"] = GPU_TIMEOUT_S
    out["est_usd_per_hour"] = round(PRICE_PER_S * 3600, 2)
    print(json.dumps(out, indent=1))
    return out


def _wait_health(port: int, proc, limit_s: int = 600) -> bool:
    import requests
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            return False
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _round(port: int, conc: int, n_predict: int, offset: int) -> dict:
    import requests
    from concurrent.futures import ThreadPoolExecutor

    def one(i: int) -> dict:
        r = requests.post(f"http://127.0.0.1:{port}/completion", timeout=900, json={
            "prompt": PROMPTS[(offset + i) % len(PROMPTS)], "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "seed": 1234, "cache_prompt": False, "ignore_eos": True})
        j = r.json()
        return {"text": j.get("content", ""), "n": j.get("timings", {}).get("predicted_n", 0),
                "tps": j.get("timings", {}).get("predicted_per_second", 0.0)}

    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    n = sum(r["n"] for r in res)
    return {"conc": conc, "tokens": n, "wall_s": round(wall, 2), "agg_tps": round(n / wall, 3),
            "per_req_tps": [round(r["tps"], 3) for r in res], "texts": [r["text"] for r in res]}


@app.function(image=image, gpu=GPU, cpu=CPUS, memory=MEM_MIB, timeout=GPU_TIMEOUT_S,
              volumes={VOL_MODELS: vol_models, VOL_EVID: vol_evid})
def run(models: dict) -> dict:
    import re
    import signal
    import subprocess

    t_start = time.time()
    root = f"{VOL_EVID}/llamacpp-moe-{RUN_ID}"
    os.makedirs(root, exist_ok=True)
    summary = {"run_id": RUN_ID, "llama_commit": LLAMA_COMMIT, "gpu": GPU, "rows": [], "notes": []}

    def spent() -> float:
        return (time.time() - t_start) * PRICE_PER_S

    def save() -> None:
        json.dump(summary, open(f"{root}/summary.json", "w"), indent=1)
        vol_evid.commit()

    summary["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()

    # 1. exactness on the GPU: the MoE cache rows of test-llama-archs, including the new invariance check
    t = subprocess.run(["/opt/llama.cpp/build/bin/test-llama-archs", "-a", "deepseek2", "-b", "CUDA0"],
                       capture_output=True, text=True, timeout=1200)
    open(f"{root}/test-llama-archs.txt", "w").write(t.stdout + "\n--- stderr ---\n" + t.stderr[-20000:])
    summary["archs_test"] = {"rc": t.returncode,
                             "moe_cache_rows": [l for l in t.stdout.splitlines() if "MoE cache" in l or "Summary" in l]}
    save()

    # 2-4. servers: one per (model, policy, cache size)
    port = 8080
    for name, info in models.items():
        if "error" in info:
            summary["notes"].append(f"{name}: skipped, {info['error']}")
            continue
        for frac in CACHE_FRACTIONS:
            mib = int(info["expert_mib"] * frac)
            for policy in POLICIES:
                if spent() > MAX_USD * 0.85:
                    summary["notes"].append(f"stopped before {name}/{policy}/{frac}: ${spent():.2f} spent")
                    save()
                    return summary
                tag = f"{name}-{policy}-c{int(frac * 100)}"
                trace = f"{root}/{tag}.mct"
                env = dict(os.environ, LLAMA_MOE_CACHE_POLICY=policy, LLAMA_MOE_CACHE_TRACE=trace)
                cmd = ["/opt/llama.cpp/build/bin/llama-server", "-m", info["path"], "-ngl", "99", "-cmoe",
                       "--moe-cache-mib", str(mib), "-np", "8", "-c", str(8 * 1024), "-t", str(CPUS),
                       "--port", str(port), "--no-webui"]
                log = open(f"{root}/{tag}.log", "w")
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                row = {"tag": tag, "model": name, "policy": policy, "cache_frac": frac, "cache_mib": mib}
                try:
                    if not _wait_health(port, proc):
                        row["error"] = "server did not start"
                    else:
                        _round(port, 8, 64, 0)  # warm-up, fills the cache
                        row["rounds"] = [_round(port, c, N_PREDICT, 0) for c in CONCURRENCY]
                except Exception as e:
                    row["error"] = repr(e)[:300]
                finally:
                    proc.send_signal(signal.SIGINT)  # clean exit: the cache logs its stats and closes the trace
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    log.close()
                text = open(f"{root}/{tag}.log").read()
                row["cache_stats"] = re.findall(r"llama_moe_cache: (ubatch [<>]=? *\d+): hits = (\d+), misses = (\d+)", text)
                row["policy_logged"] = re.findall(r"MoE cache eviction policy: (\w+)", text)
                summary["rows"].append(row)
                save()
                port += 1

    # text exactness at 1 request: every (policy, size) of a model must give the same greedy text
    for name in models:
        texts = {r["tag"]: r["rounds"][0]["texts"][0] for r in summary["rows"]
                 if r["model"] == name and "rounds" in r}
        if texts:
            summary.setdefault("text_identical", {})[name] = len(set(texts.values())) == 1
    summary["est_usd"] = round(spent(), 3)
    save()
    return summary


@app.function(image=image, volumes={VOL_EVID: vol_evid}, cpu=4, memory=16384, timeout=3600)
def analyze() -> dict:
    """CPU: replay each server's trace through the patched moe-cache-tune.py, both policies, and compare to C++."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tune", "/opt/llama.cpp/scripts/moe-cache-tune.py")
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    root = f"{VOL_EVID}/llamacpp-moe-{RUN_ID}"
    summary = json.load(open(f"{root}/summary.json"))
    out = []
    for row in summary["rows"]:
        trace_path = f"{root}/{row['tag']}.mct"
        if not os.path.exists(trace_path):
            continue
        n_layer, n_used, groups, records = tune.read_trace(trace_path)
        g = groups[0]
        n_slots = tune.n_slots_of(g, row["cache_mib"], 0, n_used)
        res = {"tag": row["tag"], "n_slots": n_slots, "cpp": row.get("cache_stats")}
        for policy in POLICIES:
            c = tune.ClassLRU(g.layers, g.n_expert, n_slots, policy)
            hits = lookups = 0
            for il, nu, ids in records:
                n_tok = len(ids) // nu
                if il not in c.slot_map or n_tok > tune.MAX_BATCH or min(n_tok * nu, g.n_expert) > n_slots:
                    continue
                uniq = len(set(ids))
                n_fill, _ = c.plan(il, list(ids), nu, 0, lambda n: 0)
                hits += uniq - n_fill
                lookups += uniq
            res[f"sim_{policy}"] = round(hits / max(lookups, 1), 4)
        out.append(res)
        print(json.dumps(res))
    json.dump(out, open(f"{root}/replay.json", "w"), indent=1)
    vol_evid.commit()
    return {"replay": out}


@app.local_entrypoint()
def main() -> None:
    pf = preflight.remote()
    if not pf["ok"]:
        print("preflight failed, no GPU launched:", json.dumps(pf, indent=1))
        return
    print(f"preflight ok. GPU run: L4, hard stop {pf['gpu_timeout_s']} s (~${MAX_USD:.2f} max)")
    s = run.remote(pf["models"])
    print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
    for r in s["rows"]:
        print(r["tag"], r.get("error", ""), [(x["conc"], x["agg_tps"]) for x in r.get("rounds", [])], r.get("cache_stats"))
    a = analyze.remote()
    print(json.dumps(a, indent=1))
