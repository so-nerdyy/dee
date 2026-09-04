#!/usr/bin/env python3
"""Three-way Kaggle microbench runner: C12 decode, C2 gate/up, C1-PRE tile.

Codex: fresh notebook (never the campaign checkout). Place this file beside
fp4_decode_vec.cu (C12, UNCHANGED control), c2_gate_up_stack.cu,
c1_tile_consume.cu and run:

    python3 kaggle_microbench_three_way.py --workdir /tmp/t4kn3 --iters 200 --reps 5
    python3 kaggle_microbench_three_way.py --workdir /tmp/t4kn3 --shard <shard> \\
        --layer 0 --expert 155

Refuses non-SM75 unless --allow-non-sm75 is given (development override;
results from other SMs are NOT T4 evidence). Builds all three binaries with
nvcc (no dee build), extracts real W1/W3 fixtures for one (layer, expert)
(default 0/155) via header-only safetensors parse, runs each test
independently with --verify-only correctness gating first, then --reps timed
repetitions, and writes one aggregate JSON. Bounded device memory
(C12 ~120 MiB, C2 ~150 MiB, C1-PRE ~50 MiB, run sequentially and freed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    "c12": ("fp4_decode_vec.cu", "fp4_decode_vec", []),
    "c2": ("c2_gate_up_stack.cu", "c2_gate_up_stack", ["-lcublas"]),
    "c1pre": ("c1_tile_consume.cu", "c1_tile_consume", []),
}
OFFICIAL_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
# Per-projection packed/e8m0 sizes (w1/w3 [2048,4096], w2 [4096,2048]).
PROJ = {"w1": (2048, 4096), "w3": (2048, 4096), "w2": (4096, 2048)}
WARMUP_DEFAULT = 20  # hardcoded warmup in all three binaries


def proj_sizes(proj: str) -> tuple[int, int]:
    out, inn = PROJ[proj]
    return out * (inn // 2), out * (inn // 32)


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_gpu(allow_other: bool) -> dict:
    try:
        proc = run(["nvidia-smi", "--query-gpu=index,name,compute_cap",
                    "--format=csv,noheader"], timeout=60)
    except FileNotFoundError:
        raise SystemExit("REFUSE: nvidia-smi not found; no GPU here.")
    if proc.returncode != 0:
        raise SystemExit(f"REFUSE: nvidia-smi failed: {proc.stderr[:300]}")
    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"index": parts[0], "name": parts[1],
                         "compute_cap": parts[2]})
    if not gpus:
        raise SystemExit("REFUSE: no GPUs enumerated.")
    t4_only = all(g["compute_cap"] == "7.5" for g in gpus)
    if not t4_only and not allow_other:
        raise SystemExit("REFUSE: non-SM75 GPU present; pass --allow-non-sm75 "
                         "for development (results are NOT T4 evidence).")
    return {"gpus": gpus, "sm75_only": t4_only,
            "override": bool(allow_other and not t4_only)}


def extract_fixtures(shard: Path | None, layer: int, expert: int,
                     workdir: Path) -> dict:
    info: dict = {"layer": layer, "expert": expert, "status": "synthetic",
                  "files": {}, "tensor_names": {}}
    if shard is None:
        info["reason"] = "no --shard; all cases synthetic"
        return info
    if not shard.is_file():
        info["reason"] = f"shard not found: {shard}; all cases synthetic"
        return info
    try:
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        with open(shard, "rb") as fh:
            for proj in ("w1", "w3"):
                for kind in ("weight", "scale"):
                    name = f"layers.{layer}.ffn.experts.{expert}.{proj}.{kind}"
                    entry = header.get(name)
                    if not entry:
                        info["reason"] = (f"tensor {name} not in this shard "
                                          "(layer may be dense/hash); synthetic")
                        info["files"] = {}
                        return info
                    expect = (proj_sizes(proj)[0] if kind == "weight"
                              else proj_sizes(proj)[1])
                    begin, end = entry["data_offsets"]
                    if end - begin != expect:
                        info["reason"] = (f"{name} size {end - begin} != {expect}; "
                                          "synthetic")
                        info["files"] = {}
                        return info
                    fh.seek(data_start + begin)
                    dest = workdir / f"l{layer}e{expert}.{proj}.{kind}.bin"
                    dest.write_bytes(fh.read(end - begin))
                    info["files"][f"{proj}.{kind}"] = str(dest)
                    info["tensor_names"][f"{proj}.{kind}"] = name
        info["status"] = "real"
        info["model"] = OFFICIAL_MODEL
        info["revision"] = OFFICIAL_REV
        info["shard"] = str(shard)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"parse failed ({exc}); synthetic"
        info["files"] = {}
    return info


def build_all(workdir: Path) -> dict:
    built = {}
    for key, (src, binary, libs) in SOURCES.items():
        dest = workdir / src
        dest.write_bytes((HERE / src).read_bytes())
        out = workdir / binary
        proc = run(["nvcc", "-O3", "-arch=sm_75", "-o", str(out), str(dest),
                    *libs], timeout=900)
        built[key] = {"binary": str(out), "exit": proc.returncode,
                      "stderr_tail": proc.stderr[-1500:]}
        if proc.returncode != 0:
            raise SystemExit(f"BUILD FAILED [{key}]:\n{proc.stderr[-4000:]}")
    return built


def one_run(binary: str, args: list[str]) -> dict:
    proc = run([binary, *args], timeout=900)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"status": "unparseable", "stdout": proc.stdout[-1500:],
                   "stderr": proc.stderr[-1500:]}
    payload["exit"] = proc.returncode
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/t4kn3"))
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--shard", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=155)
    parser.add_argument("--allow-non-sm75", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.time()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "package": "t4_kernel_next/three-way",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "controls": {"c12_unchanged": True},
    }
    report["gpu"] = check_gpu(args.allow_non_sm75)
    nvcc = run(["nvcc", "--version"], timeout=60)
    if nvcc.returncode != 0:
        raise SystemExit("REFUSE: nvcc not available.")
    report["nvcc"] = (nvcc.stdout.strip().splitlines() or ["?"])[0]
    report["built"] = build_all(args.workdir)
    report["fixture"] = extract_fixtures(args.shard, args.layer, args.expert,
                                         args.workdir)
    real = report["fixture"].get("status") == "real"
    files = report["fixture"].get("files", {})

    # Per-test command templates (fixture identity travels in every record).
    cases: dict[str, dict] = {
        "c12": {"binary": report["built"]["c12"]["binary"],
                "args": ["--out", "2048", "--in", "4096"] +
                (["--packed-file", files["w1.weight"],
                  "--scale-file", files["w1.scale"]] if real else [])},
        "c2": {"binary": report["built"]["c2"]["binary"],
               "args": ["--out", "2048", "--in", "4096"] +
               (["--packed1", files["w1.weight"], "--scale1", files["w1.scale"],
                 "--packed3", files["w3.weight"],
                 "--scale3", files["w3.scale"]] if real else [])},
        "c1pre": {"binary": report["built"]["c1pre"]["binary"],
                  "args": ["--rows", "2048" if real else "512",
                           "--in", "4096"] +
                  (["--packed-file", files["w1.weight"],
                    "--scale-file", files["w1.scale"]] if real else [])},
    }

    results: dict = {}
    failed = False
    for name, spec in cases.items():
        # Gate first: correctness without timing.
        gate = one_run(spec["binary"], [*spec["args"], "--verify-only"])
        ok_key = ("bitwise_identical" if name in ("c12", "c1pre")
                  else "gemm_bitwise")
        passed = gate.get("exit") == 0 and (
            gate.get("bitwise_identical") is True if name != "c2"
            else gate.get("admission") == "BITWISE_IDENTICAL")
        entry: dict = {"gate": gate, "gate_key": ok_key,
                       "warmup": WARMUP_DEFAULT, "reps": []}
        if not passed:
            entry["status"] = ("gate-failed: timing inadmissible; "
                               "see admission field")
            failed = True
        else:
            for _ in range(args.reps):
                rep = one_run(spec["binary"],
                              [*spec["args"], "--iters", str(args.iters)])
                entry["reps"].append(rep)
                if rep.get("exit") != 0:
                    failed = True
            tkey = time_key_for(name)
            means = [r[tkey] for r in entry["reps"]
                     if isinstance(r.get(tkey), (int, float))]
            entry["timing_summary_ms"] = {
                "key": tkey,
                "mean": statistics.fmean(means) if means else None,
                "p50": statistics.median(means) if means else None,
                "min": min(means) if means else None,
                "n": len(means),
            }
        results[name] = entry
    report["results"] = results
    report["elapsed_s"] = time.time() - t0
    report["verdict"] = ("DONE: see per-test admission + timing_summary_ms "
                         "(kernel time only, never TPS)."
                         if not failed else
                         "DONE WITH GATE FAILURES: failed tests carry no "
                         "admissible timing.")
    out_path = args.out_json or (args.workdir / "three_way_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out_path}", file=sys.stderr)
    return 0 if not failed else 1


def time_key_for(name: str) -> str:
    return {"c12": "vec_ms", "c2": "path_b_ms", "c1pre": "cand_ms"}[name]


if __name__ == "__main__":
    raise SystemExit(main())
