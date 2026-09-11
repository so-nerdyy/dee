#!/usr/bin/env python3
"""Phase-3 full-expert-store build — Kaggle kernel session driver (CPU 1/10).

Implements PHASE3_BUILD_PLAN.md section 7 end to end inside ONE Kaggle CPU
session:

  1. CLONE     the public repo branch research/phase3-build-execution pinned
               to an exact commit (immutable run code).
  2. PREFLIGHT create the ``-index`` dataset FIRST via the driver's
               ``preflight`` subcommand — auth/quota/slug refusals abort the
               session before any build time is spent (plan NO-GO #1).
  3. PILOT     ``build --buckets 2 --source remote --prefetch 8`` under the
               ``-pilot`` dataset prefix; measured rates are read back from
               the emitted job_report.json / publish.journal.jsonl.
  4. DECIDE    remote ETA for the 46-bucket store <= ~2 h  -> full build
               ``--source remote --prefetch 8`` under ``-full`` prefix;
               otherwise restart same-session ``--source local --shards
               /kaggle/input/deepseek-v4-flash-0731-shards``.
               Per the plan the full build ALWAYS uses a fresh build dir:
               the pilot ran a prefix-scoped (2-bucket) universe under a
               different dataset prefix, so its journals do not describe the
               46-bucket job and there is nothing to resume — the two pilot
               buckets are simply rebuilt (~4-5 min at mount speed), keeping
               every published artifact under uniform ``-full`` naming.
  5. ABORT     (plan NO-GO): preflight dataset-create refused; pilot remote
               rate implies > ~6 h total AND the shard mount is unreadable;
               a verify-before-push sha256 mismatch (fail-closed).
  6. OUTPUT    session-summary.json + p3-seed/ (resume seed: journals,
               metadata, manifest, records, reports) + p3-output/ (evidence
               copies, integrity head/tail, timing summary) — all under
               /kaggle/working so they are captured as kernel output.

Kaggle needs: KAGGLE_USERNAME / KAGGLE_KEY attached as kernel Secrets,
enable_internet=true, GPU off, dataset source
``nivind/deepseek-v4-flash-0731-shards`` mounted (mount fallback path).

Self-test (no Kaggle, no network, zero real bytes):
    P3_KERNEL_DRYRUN=1 python session-driver.py
runs the IDENTICAL orchestration against ``--source synthetic`` /
``--publisher local`` on a small synthetic mini-universe in a temp dir.
Force the decision branch with P3_FORCE_DECISION=remote|mount|abort.

Stdlib only; drives p3_kaggle_job.py strictly via subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Run configuration (section 7 wiring)
# ---------------------------------------------------------------------------

REPO_URL = "https://github.com/so-nerdyy/dee"
BRANCH = "research/phase3-build-execution"
PINNED_COMMIT = "42ed81059fa357161009251fb3791b8be36de994"
DRIVER_REL = "dee.cpp/tools/phase3/p3_kaggle_job.py"
HEADERS_REL = ("dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
               "shard-headers")

DATASET_OWNER = "nivind"
PREFIX_FULL = "dee4-p3-full"
PREFIX_PILOT = "dee4-p3-pilot"
SHARD_MOUNT = "/kaggle/input/deepseek-v4-flash-0731-shards"
SHARD_GLOB = "model-*-of-*.safetensors"
EXPECTED_SHARDS = 48

N_BUCKETS_FULL = 46
PILOT_BUCKETS = 2
PREFETCH = 8
RECORD_BYTES = 13_369_344
EXPERTS_PER_LAYER = 256
TOTAL_STORE_BYTES = N_BUCKETS_FULL * EXPERTS_PER_LAYER * RECORD_BYTES

# Decision thresholds (plan section 7): remote wins below ~2 h ETA; a
# remote ETA above ~6 h with a broken mount is the abort condition.
REMOTE_ETA_LIMIT_S = float(os.environ.get("P3_REMOTE_ETA_LIMIT_S", 2 * 3600))
ABORT_ETA_S = float(os.environ.get("P3_ABORT_ETA_S", 6 * 3600))
MIN_WORKING_FREE = 9 * (1 << 30)   # ~6.4 GiB transient peak + headroom

DRYRUN = os.environ.get("P3_KERNEL_DRYRUN", "0") == "1"
FORCE_DECISION = os.environ.get("P3_FORCE_DECISION", "").strip().lower()

KAGGLE_WORK = Path("/kaggle/working")

# Synthetic mini-universe for the dryrun (same 6-component DEE4 record
# shape, tiny byte counts so the whole rehearsal costs < 1 MiB).
DRY_N_BUCKETS = 4
DRY_EXPERTS = 8
DRY_COMPONENTS = (  # (component, dtype, nbytes) in DEE4 record order
    ("w1.weight", "I8", 4096),
    ("w3.weight", "I8", 4096),
    ("w2.weight", "I8", 8192),
    ("w1.scale", "F8_E8M0", 1024),
    ("w3.scale", "F8_E8M0", 1024),
    ("w2.scale", "F8_E8M0", 2048),
)
DRY_RECORD_BYTES = sum(n for _c, _d, n in DRY_COMPONENTS)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """Timestamped progress line — kernel logs are the monitoring channel."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
          f"[p3-driver] {msg}", flush=True)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except Exception:  # noqa: BLE001 — absent/corrupt report == failure
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    try:
        for line in Path(path).read_text("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except OSError:
        pass
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_step(cmd: list[str], *, tag: str, log_dir: Path,
             cwd: Path | None = None) -> int:
    """Run a subprocess, tee its output to the kernel log + a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"
    log(f"STEP {tag}: {' '.join(str(c) for c in cmd)}")
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            print(f"[{time.strftime('%H:%M:%S')}] [{tag}] "
                  f"{line.rstrip()}", flush=True)
        rc = proc.wait()
    log(f"STEP {tag} exit={rc} (log: {log_path})")
    return rc


def canonical_json(payload: Any) -> bytes:
    """Same canonicalisation convention as p3_manifest.canonical_json_bytes
    (replicated so the dryrun needs no repo import at module level)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Dryrun universe (synthetic mini shards + manifest + records)
# ---------------------------------------------------------------------------


def make_dryrun_universe(root: Path) -> dict[str, Path]:
    """Write a tiny self-consistent universe: one shard file per bucket,
    each holding DRY_EXPERTS contiguous records of DRY_COMPONENTS layout.

    The manifest/records pair feeds ``--manifest/--records`` exactly like
    the real committed-headers path; the shard files make the mount-fallback
    (``--source local``) branch real instead of synthetic.
    """
    shards_dir = root / "mini-shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    uni_dir = root / "mini-universe"
    uni_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for bucket in range(DRY_N_BUCKETS):
        shard = f"model-{bucket:05d}-of-{DRY_N_BUCKETS:05d}.safetensors"
        header: dict[str, Any] = {}
        data = bytearray()
        seed = (bucket * 977) & 0xFF
        for expert in range(DRY_EXPERTS):
            ranges = []
            record_offset = 0
            for comp, dtype, nbytes in DRY_COMPONENTS:
                start = len(data)
                pattern = bytes(((seed + expert + i) & 0xFF)
                                for i in range(min(nbytes, 256)))
                blob = (pattern * (nbytes // len(pattern) + 1))[:nbytes]
                data += blob
                tensor = (f"layers.{bucket}.ffn.experts.{expert}." + comp)
                header[tensor] = {
                    "dtype": dtype,
                    "shape": [1, nbytes],
                    "data_offsets": [start, start + nbytes],
                }
                ranges.append([comp, start, nbytes, record_offset, tensor])
                record_offset += nbytes
            records.append({
                "record_index": bucket * DRY_EXPERTS + expert,
                "bucket": bucket,
                "domain": "main",
                "layer": bucket,
                "expert": expert,
                "shard": shard,
                "record_bytes": DRY_RECORD_BYTES,
                "ranges": ranges,
            })
        hj = json.dumps(header, separators=(",", ":")).encode()
        with (shards_dir / shard).open("wb") as fh:
            fh.write(struct.pack("<Q", len(hj)))
            fh.write(hj)
            fh.write(bytes(data))

    pairs = [[b, e] for b in range(DRY_N_BUCKETS)
             for e in range(DRY_EXPERTS)]
    manifest: dict[str, Any] = {
        "schema": "dee4-p3-full-universe-v1",
        "model": "dryrun/synthetic-mini",
        "revision": "dryrun",
        "n_layers": DRY_N_BUCKETS,
        "n_hash_layers": 0,
        "n_mtp_buckets": 0,
        "n_buckets": DRY_N_BUCKETS,
        "experts_per_layer": DRY_EXPERTS,
        "total_experts": DRY_N_BUCKETS * DRY_EXPERTS,
        "record_bytes": DRY_RECORD_BYTES,
        "store_bytes": DRY_N_BUCKETS * DRY_EXPERTS * DRY_RECORD_BYTES,
        "store_gib": round(DRY_N_BUCKETS * DRY_EXPERTS * DRY_RECORD_BYTES
                           / (1 << 30), 6),
        "layer_bytes": DRY_EXPERTS * DRY_RECORD_BYTES,
        "component_order": [c[0] for c in DRY_COMPONENTS],
        "components": [
            {
                "component": comp,
                "dtype": dtype,
                "shape": [1, nbytes],
                "nbytes": nbytes,
                "record_offset": sum(c[2] for c in DRY_COMPONENTS[:i]),
            }
            for i, (comp, dtype, nbytes) in enumerate(DRY_COMPONENTS)
        ],
        "universe_sha256": hashlib.sha256(canonical_json(pairs)).hexdigest(),
        "bucket_shards": {
            str(b): {
                "shard": f"model-{b:05d}-of-{DRY_N_BUCKETS:05d}.safetensors",
                "domain": "main",
            } for b in range(DRY_N_BUCKETS)
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json({k: v for k, v in manifest.items()
                        if k != "manifest_sha256"})).hexdigest()

    mpath = uni_dir / "p3_manifest.json"
    rpath = uni_dir / "p3_records.jsonl"
    mpath.write_text(json.dumps(manifest, indent=2), "utf-8")
    with rpath.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return {"shards": shards_dir, "manifest": mpath, "records": rpath}


# ---------------------------------------------------------------------------
# Probes / measurements
# ---------------------------------------------------------------------------


def probe_mount(shards_dir: Path, n_expected: int) -> dict[str, Any]:
    """Verify the checkpoint-shards mount: every expected shard exists and
    is readable (an 8-byte prefix read is enough to prove the mount)."""
    probe: dict[str, Any] = {"dir": str(shards_dir), "ok": False}
    try:
        shards = sorted(shards_dir.glob(SHARD_GLOB))
        probe["n_shards"] = len(shards)
        if len(shards) != n_expected:
            probe["error"] = (f"{len(shards)} shards at {shards_dir}, "
                              f"expected {n_expected}")
            return probe
        probe["total_gib"] = round(
            sum(p.stat().st_size for p in shards) / (1 << 30), 2)
        with shards[min(2, len(shards) - 1)].open("rb") as fh:
            raw = fh.read(8)
        if len(raw) != 8:
            probe["error"] = "short read on mounted shard"
            return probe
        probe["ok"] = True
    except Exception as exc:  # noqa: BLE001
        probe["error"] = repr(exc)
    return probe


def measure_pilot(build_dir: Path, full_store_bytes: int) -> dict[str, Any]:
    """Turn the pilot's emitted artifacts into the measured-rate dict.

    Sources: job_report.json (bytes/wall/build report), publish journal
    (per-segment push timestamps -> upload-cycle estimate).
    """
    report = read_json(build_dir / "job_report.json") or {}
    build = report.get("build") or {}
    wall = float(report.get("wall_seconds") or 0)
    build_wall = float(build.get("wall_seconds") or 0)
    bytes_written = int(build.get("bytes_written") or 0)
    records_written = int(build.get("records_written") or 0)
    m: dict[str, Any] = {
        "success": bool(report.get("success")),
        "error": report.get("error"),
        "push_errors": report.get("push_errors") or [],
        "pushed_buckets": report.get("pushed_buckets") or [],
        "wall_seconds": wall,
        "build_wall_seconds": build_wall,
        "bytes_written": bytes_written,
        "records_written": records_written,
    }
    if wall > 0 and bytes_written > 0:
        m["job_mib_per_s"] = round(bytes_written / wall / (1 << 20), 3)
    if build_wall > 0 and bytes_written > 0:
        m["build_mib_per_s"] = round(
            bytes_written / build_wall / (1 << 20), 3)
        # ~6 range requests per record + one header probe per home shard.
        est_req = records_written * 6 + len(m["pushed_buckets"])
        m["est_requests"] = est_req
        m["est_req_per_s"] = round(est_req / build_wall, 2)
    # Upload-cycle estimate from consecutive push timestamps (coarse at 1 s
    # resolution; includes interleaved build time, i.e. conservative).
    pushes = [e for e in read_jsonl(build_dir / "publish.journal.jsonl")
              if "bucket" in e]
    if len(pushes) >= 2:
        try:
            from datetime import datetime
            ts = [datetime.strptime(e["utc"], "%Y-%m-%dT%H:%M:%SZ")
                  for e in pushes]
            dt = (ts[-1] - ts[0]).total_seconds()
            if dt > 0:
                m["push_cycle_s_per_segment"] = round(
                    dt / (len(pushes) - 1), 1)
        except Exception:  # noqa: BLE001
            pass
    # Conservative full-store ETA: job wall scaled by byte ratio.
    if bytes_written > 0 and wall > 0:
        m["eta_full_seconds"] = round(
            wall * full_store_bytes / bytes_written, 0)
    return m


# ---------------------------------------------------------------------------
# Output collection
# ---------------------------------------------------------------------------


def collect_outputs(work: Path, store: Path, tag: str,
                    out_dir: Path, *, to_seed: bool = False
                    ) -> dict[str, Any]:
    """Copy the small proof artifacts out of a build dir.

    ``p3-seed/`` mirrors exactly the flat layout P3KaggleJob.seed_from
    expects (journals + metadata + manifest + records at top level), so a
    continuation kernel can attach this output dir as --seed-dir.  Only the
    FULL build feeds the seed — the pilot ran a different (2-bucket)
    universe under a different dataset prefix and its journals must never
    be resumed into the 46-bucket job.
    ``p3-output/`` carries evidence copies + integrity head/tail.
    """
    found: dict[str, Any] = {}
    seed = work / "p3-seed"
    small = ["build.journal.jsonl", "publish.journal.jsonl",
             "integrity.jsonl", "metadata.json", "build_report.json",
             "job_report.json", "p3_manifest.json", "p3_records.jsonl"]
    for name in small:
        src = store / name
        if src.is_file():
            if to_seed:
                seed.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, seed / name)
            shutil.copyfile(src, out_dir / f"{tag}.{name}"
                            if tag else out_dir / name)
            found[name] = src.stat().st_size
    integrity = store / "integrity.jsonl"
    if integrity.is_file():
        lines = integrity.read_text("utf-8").splitlines()
        (out_dir / f"{tag}.integrity.head.jsonl").write_text(
            "\n".join(lines[:16]) + "\n", "utf-8")
        (out_dir / f"{tag}.integrity.tail.jsonl").write_text(
            "\n".join(lines[-16:]) + "\n", "utf-8")
        found["integrity_lines"] = len(lines)
    return found


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def main() -> int:
    t_start = time.monotonic()
    summary: dict[str, Any] = {
        "job": "phase3-full-expert-store build+publish (CPU 1/10)",
        "dryrun": DRYRUN,
        "branch": BRANCH,
        "pinned_commit": PINNED_COMMIT,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": False,
        "stages": {},
    }
    stage_t0 = t_start

    def mark(stage: str, **extra: Any) -> None:
        nonlocal stage_t0
        summary["stages"][stage] = {
            "wall_seconds": round(time.monotonic() - stage_t0, 2), **extra}
        stage_t0 = time.monotonic()

    # -- workspace --------------------------------------------------------
    if DRYRUN:
        work = Path(os.environ.get(
            "P3_DRYRUN_DIR",
            tempfile.mkdtemp(prefix="p3-dryrun-"))).resolve()
        work.mkdir(parents=True, exist_ok=True)
        # session-driver.py lives at <repo>/dee.cpp/tools/phase3/
        # kernel_p3_build/ -> repo root is parents[4].
        repo = Path(os.environ["P3_REPO_DIR"]).resolve() \
            if os.environ.get("P3_REPO_DIR") \
            else Path(__file__).resolve().parents[4]
        log(f"DRYRUN workspace={work} repo={repo}")
    else:
        work = KAGGLE_WORK
        repo = work / "dee-src"
    out_dir = work / "p3-output"
    logs = work / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    summary["work_dir"] = str(work)

    try:
        # -- 1. code checkout (skipped in dryrun: use the local tree) ------
        driver = repo / DRIVER_REL
        headers = repo / HEADERS_REL
        if not DRYRUN:
            rc = run_step([
                "git", "clone", "--depth", "50", "-b", BRANCH,
                REPO_URL, str(repo)], tag="clone", log_dir=logs, cwd=work)
            if rc != 0:
                raise RuntimeError(f"git clone failed rc={rc}")
            rc = run_step(["git", "-C", str(repo), "checkout",
                           PINNED_COMMIT], tag="pin", log_dir=logs)
            if rc != 0:
                raise RuntimeError(
                    f"pinned commit {PINNED_COMMIT} not reachable")
            mark("clone")
        if not driver.is_file():
            raise RuntimeError(f"driver missing at {driver}")
        log(f"driver={driver}")

        # -- environment gates --------------------------------------------
        free = shutil.disk_usage(str(work)).free
        summary["work_free_gib"] = round(free / (1 << 30), 2)
        if free < MIN_WORKING_FREE:
            raise RuntimeError(
                f"working free {free / (1 << 30):.1f} GiB < "
                f"{MIN_WORKING_FREE / (1 << 30):.0f} GiB")
        if not DRYRUN and not (
                os.environ.get("KAGGLE_USERNAME")
                and os.environ.get("KAGGLE_KEY")
                or (Path.home() / ".kaggle/kaggle.json").is_file()):
            log("WARN: no KAGGLE_USERNAME/KAGGLE_KEY env and no "
                "~/.kaggle/kaggle.json — preflight is the auth arbiter")

        # -- universe wiring -----------------------------------------------
        if DRYRUN:
            uni = make_dryrun_universe(work)
            universe_args = ["--manifest", str(uni["manifest"]),
                             "--records", str(uni["records"])]
            mount_dir = uni["shards"]
            n_mount = DRY_N_BUCKETS
            n_buckets_full = DRY_N_BUCKETS
            full_store_bytes = (DRY_N_BUCKETS * DRY_EXPERTS
                                * DRY_RECORD_BYTES)
            published_root = work / "published"
            pub_args = ["--publisher", "local",
                        "--published-root", str(published_root)]
        else:
            universe_args = ["--headers", str(headers)]
            mount_dir = Path(SHARD_MOUNT)
            n_mount = EXPECTED_SHARDS
            n_buckets_full = N_BUCKETS_FULL
            full_store_bytes = TOTAL_STORE_BYTES
            pub_args = ["--publisher", "kaggle",
                        "--dataset-owner", DATASET_OWNER, "--public"]

        # Continuation run (CPU 2/10): point P3_SEED_DIR at the previous
        # kernel's p3-seed output dir and the full build resumes in place.
        # The pilot+decide phases are skipped — P3_FULL_SOURCE carries the
        # first session's decision ("remote" or "mount"; default remote).
        continuation = bool(os.environ.get("P3_SEED_DIR"))
        seed_args = (["--seed-dir", os.environ["P3_SEED_DIR"]]
                     if continuation else [])

        def src_args(which: str) -> list[str]:
            """Map a decision ('remote'/'mount') to driver --source args."""
            if DRYRUN:
                # synthetic stands in for the remote source; prefetch kept
                # so the dryrun exercises the same wrapper stack.
                return (["--source", "synthetic",
                         "--prefetch", str(PREFETCH)]
                        if which == "remote"
                        else ["--source", "local", "--shards",
                              str(mount_dir), "--prefetch", str(PREFETCH)])
            return (["--source", "remote", "--prefetch", str(PREFETCH)]
                    if which == "remote"
                    else ["--source", "local", "--shards", str(mount_dir),
                          "--prefetch", str(PREFETCH)])

        # -- 2. PREFLIGHT: create the -full index dataset FIRST ------------
        log("PREFLIGHT: create index dataset (fail-fast auth/quota/slug)")
        rc = run_step(
            [sys.executable, str(driver), "preflight",
             "--work-dir", str(work / "_preflight"),
             *universe_args, *pub_args,
             "--dataset-prefix", PREFIX_FULL],
            tag="preflight", log_dir=logs, cwd=work)
        mark("preflight", rc=rc)
        if rc != 0:
            summary["abort_reason"] = (
                "preflight dataset create refused (auth/quota/slug) — "
                "plan NO-GO #1")
            log("ABORT: " + summary["abort_reason"])
            return finish(summary, work, t_start, 0)

        if continuation:
            # Resume the first session's full build; its decision is
            # carried in P3_FULL_SOURCE by the orchestrator.
            decision = os.environ.get("P3_FULL_SOURCE", "remote")
            if decision not in ("remote", "mount"):
                raise RuntimeError(
                    f"P3_FULL_SOURCE must be remote|mount, got {decision}")
            reason = ("continuation run (P3_SEED_DIR set); source from "
                      "P3_FULL_SOURCE")
            summary["decision"] = decision
            summary["decision_reason"] = reason
            mark("pilot", skipped=True)
            mark("decide", decision=decision)
            log(f"CONTINUATION: decision={decision} "
                f"seed={os.environ['P3_SEED_DIR']}")
        else:
            # -- 3. PILOT: buckets 0-1 remote+prefetch, -pilot prefix ------
            pilot_store = work / "p3-pilot-store"
            log(f"PILOT: build buckets 0-{PILOT_BUCKETS - 1} "
                f"(remote, prefetch {PREFETCH}) under -pilot prefix")
            rc = run_step(
                [sys.executable, str(driver), "build",
                 "--build-dir", str(pilot_store),
                 *universe_args, "--buckets", str(PILOT_BUCKETS),
                 *src_args("remote"), *pub_args,
                 "--dataset-prefix", PREFIX_PILOT],
                tag="pilot", log_dir=logs, cwd=work)
            meas = measure_pilot(pilot_store, full_store_bytes)
            summary["pilot"] = {"rc": rc, **meas}
            mark("pilot", rc=rc)
            log("PILOT measured: " + json.dumps(meas, default=str))
            collect_outputs(work, pilot_store, "pilot", out_dir)

            # -- 4. DECIDE --------------------------------------------------
            eta = meas.get("eta_full_seconds")
            mount = probe_mount(mount_dir, n_mount)
            summary["mount_probe"] = mount
            log(f"mount probe: {json.dumps(mount)}")

            if FORCE_DECISION in ("remote", "mount", "abort"):
                decision = FORCE_DECISION
                reason = "forced via P3_FORCE_DECISION"
            elif not meas["success"] or eta is None:
                # A failed pilot == effectively infinite remote ETA.
                decision = "mount" if mount["ok"] else "abort"
                reason = (
                    f"pilot failed ({meas.get('error') or 'no report'}); "
                    f"mount {'ok' if mount['ok'] else 'unavailable'}")
            elif eta <= REMOTE_ETA_LIMIT_S:
                decision = "remote"
                reason = f"remote ETA {eta / 3600:.2f} h <= limit"
            elif mount["ok"]:
                decision = "mount"
                reason = (
                    f"remote ETA {eta / 3600:.2f} h > "
                    f"{REMOTE_ETA_LIMIT_S / 3600:.1f} h; mount-sourced "
                    "restart (fresh build dir; pilot artifacts throwaway)")
            elif eta <= ABORT_ETA_S:
                decision = "remote"
                reason = (
                    f"remote ETA {eta / 3600:.2f} h exceeds target but <= "
                    f"{ABORT_ETA_S / 3600:.0f} h and mount unavailable — "
                    "remote still fits the session")
            else:
                decision = "abort"
                reason = (
                    f"remote ETA {eta / 3600:.2f} h > "
                    f"{ABORT_ETA_S / 3600:.0f} h AND mount read errors — "
                    "plan NO-GO")

            summary["decision"] = decision
            summary["decision_reason"] = reason
            mark("decide", decision=decision)
            log(f"DECIDE -> {decision}: {reason}")

        if decision == "abort":
            summary["abort_reason"] = reason
            return finish(summary, work, t_start, 0)

        # -- 5. FULL BUILD (fresh build dir, uniform -full naming) ---------
        store = work / "p3-store"
        log(f"FULL build: {n_buckets_full} buckets, source={decision}, "
            f"prefix={PREFIX_FULL}")
        rc = run_step(
            [sys.executable, str(driver), "build",
             "--build-dir", str(store),
             *universe_args, *src_args(decision), *pub_args, *seed_args,
             "--dataset-prefix", PREFIX_FULL],
            tag="build-full", log_dir=logs, cwd=work)
        full_report = read_json(store / "job_report.json") or {}
        summary["full"] = {
            "rc": rc,
            "success": bool(full_report.get("success")),
            "error": full_report.get("error"),
            "wall_seconds": full_report.get("wall_seconds"),
            "build_wall_seconds":
                (full_report.get("build") or {}).get("wall_seconds"),
            "bytes_written":
                (full_report.get("build") or {}).get("bytes_written"),
            "pushed_buckets": full_report.get("pushed_buckets") or [],
            "push_errors": full_report.get("push_errors") or [],
            "index": full_report.get("index"),
        }
        if summary["full"]["bytes_written"] and \
                summary["full"]["build_wall_seconds"]:
            summary["full"]["build_mib_per_s"] = round(
                summary["full"]["bytes_written"]
                / summary["full"]["build_wall_seconds"] / (1 << 20), 3)
        # Datasets created (segment pushes + index) from the publish journal.
        summary["datasets_created"] = [
            e.get("dataset") for e in
            read_jsonl(store / "publish.journal.jsonl") if e.get("dataset")]
        summary["buckets_completed"] = len(
            summary["full"]["pushed_buckets"])
        mark("full_build", rc=rc)
        log("FULL result: " + json.dumps(summary["full"], default=str))
        found = collect_outputs(work, store, "full", out_dir, to_seed=True)
        summary["artifacts"] = found

        # Fail-closed: a verify-before-push sha mismatch must mark the run
        # failed even if the driver kept going on other buckets.
        verify_fail = any("sha256" in e and "mismatch" in e
                          for e in summary["full"]["push_errors"])
        if verify_fail:
            summary["abort_reason"] = (
                "verify-before-push sha256 mismatch — fail-closed")
        summary["success"] = bool(
            rc == 0 and summary["full"]["success"] and not verify_fail)
        return finish(summary, work, t_start,
                      0 if summary["success"] else 1)

    except Exception as exc:  # noqa: BLE001 — any driver-level failure
        log(f"SESSION ERROR: {exc!r}")
        summary["abort_reason"] = f"session error: {exc!r}"
        return finish(summary, work, t_start, 1)


def finish(summary: dict[str, Any], work: Path, t_start: float,
           exit_code: int) -> int:
    summary["finished_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["wall_seconds"] = round(time.monotonic() - t_start, 2)
    write_json(work / "session-summary.json", summary)
    write_json(work / "p3-output" / "timing-summary.json", {
        "stages": summary["stages"],
        "wall_seconds": summary["wall_seconds"],
        "decision": summary.get("decision"),
        "pilot": summary.get("pilot"),
    })
    log(f"SESSION END success={summary['success']} "
        f"decision={summary.get('decision')} "
        f"wall={summary['wall_seconds']}s "
        f"summary={work / 'session-summary.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
