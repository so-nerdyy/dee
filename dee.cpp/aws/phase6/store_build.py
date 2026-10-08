"""Build the segmented DEE4 store on a CPU instance and sync it to S3.

Port of modal/phase6/modal_p6_ops.py::build_store with Modal removed. The
build reads HF byte ranges (RemoteRangeSource, honoring HF_TOKEN), so the
checkpoint is never downloaded for repack. Resume-safe: the build dir is
restored from stores/<model>/ first, so build.journal.jsonl + *.partial
survive restarts, and it is re-synced every few minutes and at exit.

  python store_build.py --model dsv4-flash --src-root <clone> \
      --build-dir <dir> [--s3] [--prefetch 8] [--buckets 0]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import config

SYNC_INTERVAL_S = 300


def _sync(src: str, dst: str) -> None:
    subprocess.run(["aws", "s3", "sync", src, dst, "--only-show-errors"], check=True)


def _manifest_for_spec(src: Path, build_dir: Path, spec: dict) -> Path:
    headers_dir = build_dir / "headers"
    subprocess.run(
        [sys.executable, "dee.cpp/tools/phase3/fetch_headers.py",
         "--repo", spec["hf_repo"], "--revision", spec["hf_rev"],
         "--out-dir", str(headers_dir)],
        cwd=str(src), check=True)
    manifest_code = (
        "import sys, json; "
        "sys.path.insert(0, 'dee.cpp/tools/phase3'); "
        "import p3_manifest, p3_manifest_spec; "
        f"sp = p3_manifest_spec.load_spec('{spec['spec']}'); "
        f"m = p3_manifest_spec.build_manifest_spec(sp, '{headers_dir}'); "
        f"p3_manifest.write_manifest(m, '{build_dir}'); "
        "print('spec manifest:', json.dumps({k: m[k] for k in "
        "('n_buckets','total_experts','record_bytes','store_gib',"
        "'universe_sha256')}))")
    subprocess.run([sys.executable, "-c", manifest_code], cwd=str(src), check=True)
    return build_dir / "p3_manifest.json"


def build(model: str, src: Path, build_dir: Path, *, s3_prefix: str | None,
          prefetch: int, buckets: int) -> int:
    spec = config.MODELS[model]
    build_dir.mkdir(parents=True, exist_ok=True)
    if s3_prefix:
        _sync(s3_prefix, str(build_dir))

    cmd = [
        sys.executable, "dee.cpp/tools/phase3/p3_kaggle_job.py", "build",
        "--build-dir", str(build_dir),
        "--source", "remote",
        "--prefetch", str(prefetch),
        "--publisher", "none",
    ]
    if spec.get("spec"):
        manifest_path = _manifest_for_spec(src, build_dir, spec)
        cmd += ["--manifest", str(manifest_path),
                "--records", str(build_dir / "p3_records.jsonl")]
    else:
        cmd += ["--headers", str(src / spec["headers"])]
    if buckets:
        cmd += ["--buckets", str(buckets)]

    proc = subprocess.Popen(cmd, cwd=str(src), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})

    stop = threading.Event()

    def heartbeat() -> None:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        while not stop.wait(SYNC_INTERVAL_S):
            if not s3_prefix:
                continue
            try:
                pool.submit(_sync, str(build_dir), s3_prefix).result(timeout=900)
                print("[hb] build dir synced", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[hb] sync skipped: {exc!r}", flush=True)
        pool.shutdown(wait=False)

    threading.Thread(target=heartbeat, daemon=True).start()
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    stop.set()
    if s3_prefix:
        _sync(str(build_dir), s3_prefix)
    status_path = build_dir / "store_build_exit.json"
    status_path.write_text(json.dumps({"model": model, "rc": rc}), encoding="utf-8")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dsv4-flash", choices=sorted(config.MODELS))
    parser.add_argument("--src-root", type=Path, required=True,
                        help="clone root that contains dee.cpp/")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--s3", action="store_true",
                        help="restore from and sync to stores/<model>/")
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--buckets", type=int, default=0)
    args = parser.parse_args(argv)
    s3 = config.s3_prefix("stores", args.model) if args.s3 else None
    rc = build(args.model, args.src_root.resolve(), args.build_dir.resolve(),
               s3_prefix=s3, prefetch=args.prefetch, buckets=args.buckets)
    return rc


if __name__ == "__main__":
    sys.exit(main())
