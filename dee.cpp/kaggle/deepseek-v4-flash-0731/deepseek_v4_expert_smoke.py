#!/usr/bin/env python3
"""DS7 Kaggle smoke: ONE official DeepSeek-V4-Flash-0731 routed expert on T4.

Goal (DS7): prove one official routed expert executes on SM75 (Tesla T4)
and matches the pinned trusted reference within predeclared tolerances.

Kaggle script kernels upload ONLY code_file, so this harness follows the
proven M5G pattern: it clones the campaign branch, reads a committed
``harness-identity.json`` sidecar, checks out the pinned repository commit,
verifies the committed harness and reference SHA256s against the sidecar and
the running file, and only then imports the trusted reference module from the
pinned tree. Nothing is imported from the uploaded script directory.

Pipeline:
  1. Bootstrap environment; require a single Tesla T4.
  2. Clone the pinned branch and verify harness identity.
  3. Import the trusted FP32 reference from the pinned tree.
  4. Download the pinned shard (model-00008) with size + ETag verification.
  5. Load the 6 expert tensors (w1/w2/w3 packed I8 + F8_E8M0 scales).
  6. Trusted reference: scripts/deepseek_v4_expert_reference.py (FP32 math,
     official FP4_TABLE + E8M0 decode + asymmetric SwiGLU clamps).
  7. Candidate on T4: FP16 dequantized GEMV with FP32 accumulation.
  8. Compare per predeclared tolerance (T4 candidate carries FP8 act-quant
     and FP16 storage error, so near-bitwise agreement is NOT expected).
  9. Archive evidence JSON + manifest hashes under /kaggle/working/.

Performance reporting is disabled in this smoke run (non-comparable mode).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

RUN_ID = "20260731T000000Z-dsv4-ds7-expert-smoke"
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
ROOT = Path("/kaggle/temp/dsv4-source")
EVIDENCE = Path(f"/kaggle/working/dsv4-ds7-evidence-{RUN_ID}")
ARCHIVE_BASE = EVIDENCE
DEE = ROOT / "dee.cpp"
IDENTITY_RELATIVE = Path("dee.cpp/kaggle/deepseek-v4-flash-0731/harness-identity.json")
HARNESS_RELATIVE = Path("dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_expert_smoke.py")
REFERENCE_RELATIVE = Path("dee.cpp/scripts/deepseek_v4_expert_reference.py")

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SHARD = "model-00008-of-00048.safetensors"
# Committed in the pinned repo tree alongside the ledger (DS1 phase).
CACHED_HEADER_RELATIVE = Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers/"
    "model-00008-of-00048.safetensors.json")

# Layers.6 / expert 0 per the validated ledger (MODEL_LEDGER.json).
EXPERT_TENSORS = [
    "layers.6.ffn.experts.0.w1.weight", "layers.6.ffn.experts.0.w1.scale",
    "layers.6.ffn.experts.0.w2.weight", "layers.6.ffn.experts.0.w2.scale",
    "layers.6.ffn.experts.0.w3.weight", "layers.6.ffn.experts.0.w3.scale",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(
    command: list[str],
    log_path: Path,
    cwd: Path,
    *,
    check: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        rc = process.wait()
    if check and rc:
        raise subprocess.CalledProcessError(rc, command)
    return rc


def canonical_header_sha256(parsed: dict[str, Any]) -> str:
    """SHA256 of the canonical safetensors header JSON."""
    canonical = json.dumps(parsed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def downloaded_header_sha256(shard_path: Path) -> str:
    """Hash the ACTUAL safetensors header bytes of the downloaded file.

    Reads the 8-byte LE header length + header JSON from the shard, then
    canonicalizes with the same form the committed cached header was written
    in (json.dumps(sort_keys=True)), so the result is directly comparable to
    the canonical hash of the committed cached header. This pins the
    downloaded file to the pinned revision without needing any network
    header (urllib follows HF's redirect to the CDN, which drops
    X-Linked-Etag).
    """
    with open(shard_path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{shard_path}: truncated 8-byte prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (1 << 31):
            raise ValueError(f"{shard_path}: implausible header length {hlen}")
        header_bytes = fh.read(hlen)
        if len(header_bytes) != hlen:
            raise ValueError(f"{shard_path}: truncated header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{shard_path}: malformed header JSON: {exc}") from exc
    return canonical_header_sha256(header)


def shard_expected_bytes() -> tuple[int, str | None]:
    """Resolve (shard_byte_size, remote_sha256_or_None).

    Uses a local manifest if present, else the remote Content-Range total.
    When remote is queried, also captures X-Linked-Etag (the LFS sha256 for
    safetensor shards) so the downloaded file can be verified against it.
    """
    manifest_path = Path("/kaggle/working/CHECKPOINT_MANIFEST.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("shards", []):
            if row.get("shard") == SHARD:
                return int(row["compressed_bytes"]), None
    import urllib.request
    url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REV}/{SHARD}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
        etag = resp.headers.get("X-Linked-Etag", "")
    if "/" not in cr:
        raise RuntimeError(f"cannot resolve remote size for {SHARD}: {cr!r}")
    remote_sha = None
    if etag:
        remote_sha = etag.strip().strip('"')
        # LFS OID is a bare 64-char sha256; drop any suffix like "-N".
        if len(remote_sha) > 64:
            remote_sha = remote_sha[:64]
    return int(cr.split("/")[1]), remote_sha


def download_shard(want: int, checkpoint_dir: Path) -> Path:
    import urllib.request
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / SHARD
    url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{REV}/{SHARD}"
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    if have > want:
        raise RuntimeError(f"shard too large {have} > {want}")
    chunk = 8 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                if have > 0 and r.status != 206:
                    raise RuntimeError(
                        f"server did not honor Range resume (status {r.status} at {have})")
                # Bound the read to the requested range: with a 206 this reads
                # exactly the chunk; with an unexpected full 200 it never loads
                # the whole multi-GB body into RAM (have == 0 -> bytes 0..chunk).
                data = r.read(chunk)
            if not data:
                raise ConnectionError(f"empty chunk at {have}")
            fh.write(data)
            have += len(data)
            print(f"  download {have}/{want} ({100.0*have/want:.1f}%)", flush=True)
    if dest.stat().st_size != want:
        raise RuntimeError(f"shard size mismatch {dest.stat().st_size} != {want}")
    return dest


def load_expert(shard_path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for name in EXPERT_TENSORS:
            if name not in f.keys():
                raise KeyError(f"missing {name} in {SHARD}")
            tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def candidate_expert_on_t4(
    x: torch.Tensor, t: dict[str, torch.Tensor], ref: Any
) -> tuple[torch.Tensor, str]:
    """FP16 dequantized GEMV with FP32 accumulation, executed ON the T4.

    Every tensor is moved to the CUDA device explicitly so the matmuls run
    on the GPU (device-authentic). Returns (output_on_cpu, device_type) so
    the caller can compare against the trusted FP32 reference and record
    where the candidate actually executed.
    """
    dev = "cuda"
    w1 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w1.weight"],
                                      t["layers.6.ffn.experts.0.w1.scale"]).half().to(dev)
    w2 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w2.weight"],
                                      t["layers.6.ffn.experts.0.w2.scale"]).half().to(dev)
    w3 = ref.dequantize_expert_weight(t["layers.6.ffn.experts.0.w3.weight"],
                                      t["layers.6.ffn.experts.0.w3.scale"]).half().to(dev)
    xc = x.half().to(dev)
    gate = (xc @ w1.t()).float()
    up = (xc @ w3.t()).float()
    gate = torch.clamp(gate, max=10.0)
    up = torch.clamp(up, min=-10.0, max=10.0)
    h = (torch.nn.functional.silu(gate) * up).half()
    out = (h @ w2.t()).float()
    torch.cuda.synchronize()  # guarantee the GEMV finished before timing/metrics
    return out.cpu(), w1.device.type


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    abs_err = (reference - candidate).abs()
    rel = abs_err / (reference.abs() + 1e-8)
    return {
        "max_abs_error": float(abs_err.max()),
        "mean_abs_error": float(abs_err.mean()),
        "max_rel_error": float(rel.max()),
        "mean_rel_error": float(rel.mean()),
        "p99_rel_error": float(rel.flatten().quantile(0.99)),
    }


def main() -> int:
    print("=== DS7 DeepSeek-V4-Flash-0731 one-expert smoke on T4 ===", flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    fatal_error: dict[str, object] | None = None
    commit: str | None = None
    reference_sha: str | None = None
    metrics: dict[str, float] | None = None
    passed: bool = False
    candidate_device: str = ""

    # Predeclared tolerances for the T4 candidate (FP16 storage + FP16 GEMV
    # vs full-FP32 trusted reference). Declared BEFORE the run, per the DS7
    # protocol: the candidate is NOT expected to be bitwise exact.
    TOLERANCE = {"max_abs_error": 2.0, "mean_abs_error": 0.5, "max_rel_error": 1e-2}

    try:
        bootstrap = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": device,
                    "name": torch.cuda.get_device_name(device),
                    "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                }
                for device in range(torch.cuda.device_count())
            ],
        }
        write_json(EVIDENCE / "bootstrap-environment.json", bootstrap)
        # The campaign initial hardware is dual-T4 and Kaggle's NvidiaTeslaT4
        # accelerator provisions two devices; the DS7 candidate only uses
        # device 0, so accept any T4-only topology with >= 1 GPU.
        if torch.cuda.device_count() < 1 or not all(
            "T4" in torch.cuda.get_device_name(device)
            for device in range(torch.cuda.device_count())
        ):
            raise RuntimeError(f"expected Tesla T4 topology, got {bootstrap['gpus']}")

        run_logged(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
             "safetensors==0.8.0"],
            EVIDENCE / "logs/pip-install.log", EVIDENCE,
        )

        if ROOT.exists():
            resolved_root = ROOT.resolve()
            if not str(resolved_root).startswith("/kaggle/temp/"):
                raise RuntimeError(f"refusing to remove unexpected path {resolved_root}")
            shutil.rmtree(ROOT)
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                        REPOSITORY, str(ROOT)], check=True)

        # Script kernels upload only code_file; read the committed sidecar
        # from the cloned tree and verify everything by content hash.
        identity_path = ROOT / IDENTITY_RELATIVE
        if not identity_path.is_file():
            raise RuntimeError(f"missing harness identity {identity_path}")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("model_revision") != REV:
            raise RuntimeError({"identity_model_revision": identity.get("model_revision"),
                                "expected": REV})
        if identity.get("shard") != SHARD:
            raise RuntimeError({"identity_shard": identity.get("shard"), "expected": SHARD})
        expected_harness_commit = identity.get("repository_commit")
        if (
            not isinstance(expected_harness_commit, str)
            or len(expected_harness_commit) != 40
            or any(c not in "0123456789abcdef" for c in expected_harness_commit)
        ):
            raise RuntimeError({"repository_commit": expected_harness_commit,
                                "reason": "identity must pin a 40-char lowercase commit"})
        subprocess.run(["git", "checkout", "--quiet", expected_harness_commit],
                       cwd=ROOT, check=True)
        checked_out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=ROOT, text=True).strip()
        if checked_out != expected_harness_commit:
            raise RuntimeError({"commit": checked_out, "expected": expected_harness_commit})

        committed_harness = ROOT / HARNESS_RELATIVE
        committed_harness_sha = sha256_file(committed_harness)
        running_sha = sha256_file(Path(__file__).resolve())
        if committed_harness_sha != identity.get("harness_sha256"):
            raise RuntimeError({"committed_harness_sha256": committed_harness_sha,
                                "expected": identity.get("harness_sha256")})
        if running_sha != identity.get("harness_sha256"):
            raise RuntimeError({"running_harness_sha256": running_sha,
                                "expected": identity.get("harness_sha256")})

        committed_reference = ROOT / REFERENCE_RELATIVE
        reference_sha = sha256_file(committed_reference)
        if reference_sha != identity.get("reference_sha256"):
            raise RuntimeError({"committed_reference_sha256": reference_sha,
                                "expected": identity.get("reference_sha256")})

        sys.path.insert(0, str(ROOT / "dee.cpp"))
        from scripts import deepseek_v4_expert_reference as ref  # noqa: E402
        # Run the reference module's own self-test from the pinned tree.
        ref.main()
        commit = checked_out
        print("pinned commit", commit, flush=True)
        print("harness sha", running_sha, flush=True)
        print("reference sha", reference_sha, flush=True)

        checkpoint_dir = Path("/kaggle/working/dsv4-checkpoint")
        want, remote_sha = shard_expected_bytes()
        shard_path = download_shard(want, checkpoint_dir)
        # Pin the ACTUAL downloaded shard header to the committed cached
        # header from the pinned repo tree (network-independent).
        cached_header_path = ROOT / CACHED_HEADER_RELATIVE
        if not cached_header_path.is_file():
            raise RuntimeError(f"missing committed cached shard header {cached_header_path}")
        # Canonicalize BOTH sides so the comparison is content-based and
        # immune to eol conversion (a Windows CRLF checkout must not false-fail).
        cached = json.loads(cached_header_path.read_text(encoding="utf-8"))
        expected_header_sha = canonical_header_sha256(cached)
        got_header_sha = downloaded_header_sha256(shard_path)
        if got_header_sha != expected_header_sha:
            raise RuntimeError({"downloaded_header_sha256": got_header_sha,
                                "expected_committed_header_sha256": expected_header_sha})
        t = load_expert(shard_path)
        print("loaded expert tensors:", {k: tuple(v.shape) for k, v in t.items()}, flush=True)

        torch.manual_seed(0)
        x = torch.randn(4, 4096)  # 4 tokens, hidden=4096

        t0 = time.time()
        reference = ref.expert_forward(
            x,
            t["layers.6.ffn.experts.0.w1.weight"], t["layers.6.ffn.experts.0.w1.scale"],
            t["layers.6.ffn.experts.0.w2.weight"], t["layers.6.ffn.experts.0.w2.scale"],
            t["layers.6.ffn.experts.0.w3.weight"], t["layers.6.ffn.experts.0.w3.scale"],
            swiglu_limit=10.0,
        )
        t_ref = time.time() - t0

        t0 = time.time()
        candidate, candidate_device = candidate_expert_on_t4(x, t, ref)
        t_cand = time.time() - t0
        print("candidate executed on device:", candidate_device, flush=True)

        metrics = compare(reference, candidate)
        print("reference time %.3fs candidate time %.3fs" % (t_ref, t_cand), flush=True)
        print("metrics:", json.dumps(metrics, indent=2), flush=True)

        passed = (
            candidate_device == "cuda"
            and metrics["max_abs_error"] <= TOLERANCE["max_abs_error"]
            and metrics["mean_abs_error"] <= TOLERANCE["mean_abs_error"]
            and metrics["max_rel_error"] <= TOLERANCE["max_rel_error"]
        )

        # Stream-hash the shard (3.5 GB) to avoid loading it into RAM.
        h = hashlib.sha256()
        with open(shard_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        shard_sha256 = h.hexdigest()
        if remote_sha is not None and shard_sha256 != remote_sha:
            raise RuntimeError(f"{SHARD}: streamed sha {shard_sha256[:16]} != HF etag {remote_sha[:16]}")
        # Keep the Kaggle output artifact small: Kaggle zips the ENTIRE working
        # directory as the downloadable output, so drop the shard now that its
        # SHA is recorded and its tensors are loaded in memory.
        shard_path.unlink()

        write_json(EVIDENCE / "environment.json", {
            "schema_version": 1,
            "run_id": RUN_ID,
            "repository": REPOSITORY,
            "branch": BRANCH,
            "repository_commit": commit,
            "harness_sha256": running_sha,
            "reference_sha256": reference_sha,
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "shard": SHARD,
            "performance_comparable": False,
        })

        write_json(EVIDENCE / "ds7-expert-evidence.json", {
            "campaign": "deepseek-v4-flash-0731",
            "phase": "DS7-expert-smoke",
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "shard": SHARD,
            "layer": 6,
            "expert": 0,
            "device": torch.cuda.get_device_name(0),
            "candidate_executed_on_cuda": candidate_device == "cuda",
            "torch": torch.__version__,
            "shard_bytes": want,
            "shard_sha256": shard_sha256,
            "integrity_gate": ("header_pin+hf_etag" if remote_sha is not None
                                else "header_pin"),
            "shard_sha_verified_vs_hf_etag": remote_sha is not None and shard_sha256 == remote_sha,
            "tensor_shapes": {k: list(v.shape) for k, v in t.items()},
            "metrics": metrics,
            "tolerance": TOLERANCE,
            "passed": bool(passed),
            "verdict": "MATCH_WITHIN_TOLERANCE" if passed else "MISMATCH",
            "note": "candidate carries FP16-storage/FP16-GEMV error vs full-FP32 reference",
            "performance_comparable": False,
        })

        shutil.copy2(Path(__file__).resolve(),
                     EVIDENCE / "deepseek_v4_expert_smoke.py")
        shutil.copy2(committed_reference, EVIDENCE / "deepseek_v4_expert_reference.py")
        (EVIDENCE / "manifest.sha256").write_text(
            hashlib.sha256((EVIDENCE / "ds7-expert-evidence.json").read_bytes()).hexdigest()
            + "  ds7-expert-evidence.json\n", encoding="utf-8")

    except Exception as exc:  # noqa: BLE001
        fatal_error = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}
        write_json(EVIDENCE / "fatal-error.json", fatal_error)

    finally:
        required_paths = [
            EVIDENCE / "bootstrap-environment.json",
            EVIDENCE / "environment.json",
            EVIDENCE / "ds7-expert-evidence.json",
            EVIDENCE / "manifest.sha256",
            EVIDENCE / "logs/pip-install.log",
        ]
        required_status = []
        for path in required_paths:
            required_status.append({
                "path": path.relative_to(EVIDENCE).as_posix() if path.is_file() else str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            })
        missing = [row["path"] for row in required_status if not row["present"]]
        if missing:
            failures.append({"name": "required_artifacts_present", "details": missing})
        # Fail closed when the candidate exceeds the predeclared tolerances:
        # a numerical mismatch must surface as FAIL with a non-zero exit, never
        # as a COMPLETE kernel with the mismatch buried in the evidence JSON.
        if metrics is not None and not passed:
            failures.append({"name": "candidate_mismatch",
                             "details": {"metrics": metrics,
                                         "candidate_device": candidate_device}})
        result = "PASS" if fatal_error is None and not failures else "FAIL"
        artifacts = [
            {"path": path.relative_to(EVIDENCE).as_posix(), "bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in sorted(EVIDENCE.rglob("*"))
            if path.is_file() and path.name != "artifact-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "result": result,
            "terminal_verdict": "INVALID_EXPERIMENT",  # smoke is never an accept verdict
            "repository_commit": commit,
            "reference_sha256": reference_sha,
            "metrics": metrics,
            "passed": bool(passed),
            "tolerance": TOLERANCE,
            "fatal_error": fatal_error,
            "validation_failures": failures,
            "required_paths": required_status,
            "artifacts": artifacts,
            "performance_comparable": False,
        }
        write_json(EVIDENCE / "artifact-manifest.json", manifest)
        archive = shutil.make_archive(str(ARCHIVE_BASE), "gztar",
                                      root_dir=EVIDENCE.parent, base_dir=EVIDENCE.name)
        write_json(EVIDENCE / "archive-metadata.json", {
            "archive": archive,
            "archive_sha256": sha256_file(Path(archive)),
            "manifest_sha256": sha256_file(EVIDENCE / "artifact-manifest.json"),
            "excluded_from_archive": ["archive-metadata.json"],
        })
        print(json.dumps({"run_id": RUN_ID, "result": result, "archive": archive,
                          "fatal_error": fatal_error}, sort_keys=True), flush=True)
        if result != "PASS":
            raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
