"""Build the full DSv4-Flash segmented store on a small CPU instance, pushing
each sealed bucket to S3 and keeping local disk bounded.

Runs the existing P3KaggleJob (tools/phase3/p3_kaggle_job.py) with an S3
publisher. P3KaggleJob already hardlinks each sealed segment into staging,
uploads it, journals the push, and replaces the local file with a same-size
sparse tombstone. Local disk therefore holds about one or two segments
(~3.2 GiB each), not the 147 GiB store. The full store lives in
s3://<bucket>/stores/dsv4-flash/segments/.

Resume-safe across relaunches: only the small journal, integrity, and metadata
files are restored from S3. Segment bytes come from S3 and are never
restored locally. A bucket that was committed but never pushed (its bytes
were lost with the old instance) is dropped from the journal before the
build, so the builder rebuilds it. Progress is written to progress.json and
uploaded with the bootstrap log every 5 minutes.

  python store_build.py --model dsv4-flash --src-root <clone> \
      --build-dir <dir> --s3 --evidence s3://.../evidence/<run>/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import config
from hf_source import CountingRangeSource

PROGRESS_INTERVAL_S = 300
PROGRESS_FILE = "progress.json"
BOOTSTRAP_LOG = "/var/log/dee-p6-bootstrap.log"
SEGMENT_PREFIX = "experts-bucket-"
SEGMENT_SUFFIX = ".dee4"
ROOT_SYNC_EXCLUDES = ["--exclude", "segments/*", "--exclude", "_staging/*",
                      "--exclude", "_index_staging/*"]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_phase6_aws.py)
# ---------------------------------------------------------------------------

def read_bucket_set(journal_path: Path) -> set[int]:
    """Buckets named in a JSONL journal, ignoring a torn trailing line."""
    out: set[int] = set()
    if not journal_path.is_file():
        return out
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "bucket" in entry:
            out.add(int(entry["bucket"]))
    return out


def local_segment_buckets(segments_dir: Path) -> set[int]:
    """Buckets whose real (non-tombstone) segment file is on local disk."""
    out: set[int] = set()
    if not segments_dir.is_dir():
        return out
    for path in segments_dir.iterdir():
        name = path.name
        if (path.is_file() and name.startswith(SEGMENT_PREFIX)
                and name.endswith(SEGMENT_SUFFIX)):
            out.add(int(name[len(SEGMENT_PREFIX):-len(SEGMENT_SUFFIX)]))
    return out


def lost_bucket_cutoff(committed: set[int], pushed: set[int],
                       local: set[int]) -> int | None:
    """Lowest committed bucket whose bytes are gone (not pushed, not local).

    Buckets build strictly in order, so every committed bucket at or above
    this one must be rebuilt as well.
    """
    lost = sorted(b for b in committed if b not in pushed and b not in local)
    return lost[0] if lost else None


def prune_journal_from(journal_path: Path, cutoff: int) -> int:
    """Rewrite the build journal keeping only buckets below cutoff.

    Returns the number of journal lines removed. The builder truncates the
    integrity file to match on its next start.
    """
    if not journal_path.is_file():
        return 0
    kept, dropped = [], 0
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if "bucket" in entry and int(entry["bucket"]) >= cutoff:
            dropped += 1
            continue
        kept.append(line)
    tmp = journal_path.with_name(journal_path.name + ".tmp")
    tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    tmp.replace(journal_path)
    return dropped


def progress_snapshot(*, n_buckets: int, committed: set[int], pushed: set[int],
                      elapsed_s: float, disk_free_gib: float, http_status: dict,
                      source_stats: dict, push_errors: list[str],
                      peak_rss_kb: int, now_utc: str) -> dict:
    """Progress record. ETA uses the observed push rate since this process started."""
    rate_per_h = (len(pushed) * 3600.0 / elapsed_s) if pushed and elapsed_s > 0 else None
    remaining = n_buckets - len(pushed)
    eta_h = (remaining / rate_per_h) if rate_per_h else None
    return {
        "utc": now_utc,
        "n_buckets": n_buckets,
        "buckets_sealed": len(committed),
        "buckets_pushed": len(pushed),
        "pushed": sorted(pushed),
        "elapsed_s": round(elapsed_s, 1),
        "rate_buckets_per_h": round(rate_per_h, 3) if rate_per_h else None,
        "eta_h": round(eta_h, 2) if eta_h is not None else None,
        "disk_free_gib": round(disk_free_gib, 2),
        "peak_rss_kb": peak_rss_kb,
        "http_status": dict(http_status),
        "source_stats": dict(source_stats),
        "push_errors": list(push_errors)[-5:],
    }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


# ---------------------------------------------------------------------------
# S3 publisher (SegmentPublisher protocol of p3_kaggle_job)
# ---------------------------------------------------------------------------

class S3SegmentPublisher:
    """Uploads each sealed segment to <root>segments/ and the final metadata
    files to <root>. Uses the aws CLI, which is present on the DLAMI and the
    stock Ubuntu images.
    """

    def __init__(self, root: str):
        self.root = root if root.endswith("/") else root + "/"

    def _head_size(self, uri: str) -> int | None:
        bucket, key = parse_s3_uri(uri)
        proc = subprocess.run(
            ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key,
             "--query", "ContentLength", "--output", "text"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return int(proc.stdout.strip())

    def _upload(self, local: Path, uri: str) -> None:
        subprocess.run(["aws", "s3", "cp", str(local), uri, "--only-show-errors"],
                       check=True)

    def prepare(self, staging_dir: Path, bucket: int, seg_name: str) -> None:
        return None

    def publish(self, staging_dir: Path, seg_name: str, bucket: int, *,
                sha256: str, size: int) -> dict:
        uri = f"{self.root}segments/{seg_name}"
        if self._head_size(uri) == size:
            return {"dataset": f"{self.root}segments/", "ref": uri, "uploaded": False}
        last_error: Exception | None = None
        for _ in range(3):
            try:
                self._upload(staging_dir / seg_name, uri)
            except subprocess.CalledProcessError as exc:
                last_error = exc
                continue
            if self._head_size(uri) == size:
                return {"dataset": f"{self.root}segments/", "ref": uri, "uploaded": True}
        raise RuntimeError(f"upload of {seg_name} failed after retries: {last_error!r}")

    def publish_index(self, staging_dir: Path) -> dict:
        for path in sorted(staging_dir.iterdir()):
            if path.is_file() and path.name != "dataset-metadata.json":
                self._upload(path, f"{self.root}{path.name}")
        return {"dataset": self.root, "ref": self.root, "uploaded": True}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_p3(src: Path):
    tools = src / "dee.cpp" / "tools" / "phase3"
    sys.path.insert(0, str(tools))
    import p3_kaggle_job
    return p3_kaggle_job


def restore_small_files(root: str, build_dir: Path) -> None:
    subprocess.run(["aws", "s3", "sync", root, str(build_dir), "--exclude",
                    "segments/*", "--only-show-errors"], check=True)


def sync_small_files_up(build_dir: Path, root: str) -> None:
    subprocess.run(["aws", "s3", "sync", str(build_dir), root,
                    *ROOT_SYNC_EXCLUDES, "--only-show-errors"], check=True)


def peak_rss_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


class Heartbeat:
    def __init__(self, *, job, source, build_dir: Path, root: str, evidence: str,
                 n_buckets: int, t0: float):
        self.job = job
        self.source = source
        self.build_dir = build_dir
        self.root = root
        self.evidence = evidence
        self.n_buckets = n_buckets
        self.t0 = t0
        self.stop = threading.Event()

    def beat(self) -> dict:
        pushed = read_bucket_set(self.job.publish_journal_path)
        committed = read_bucket_set(self.job.journal_path)
        free_gib = shutil.disk_usage(self.build_dir).free / (1 << 30)
        snap = progress_snapshot(
            n_buckets=self.n_buckets, committed=committed, pushed=pushed,
            elapsed_s=time.monotonic() - self.t0, disk_free_gib=free_gib,
            http_status=self.source.status_counts, source_stats=self.source.stats,
            push_errors=list(self.job.push_errors), peak_rss_kb=peak_rss_kb(),
            now_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        progress = self.build_dir / PROGRESS_FILE
        progress.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        print(f"[progress] {json.dumps(snap)}", flush=True)
        try:
            sync_small_files_up(self.build_dir, self.root)
            subprocess.run(["aws", "s3", "cp", str(progress),
                            f"{self.evidence}{PROGRESS_FILE}", "--only-show-errors"],
                           check=False)
            if Path(BOOTSTRAP_LOG).is_file():
                subprocess.run(["aws", "s3", "cp", BOOTSTRAP_LOG,
                                f"{self.evidence}bootstrap.log", "--only-show-errors"],
                               check=False)
        except subprocess.SubprocessError as exc:
            print(f"[progress] sync skipped: {exc!r}", flush=True)
        return snap

    def run(self) -> None:
        while not self.stop.wait(PROGRESS_INTERVAL_S):
            self.beat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="dsv4-flash", choices=sorted(config.MODELS))
    ap.add_argument("--src-root", type=Path, required=True,
                    help="clone root that contains dee.cpp/")
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--s3", action="store_true",
                    help="restore small files from and publish to stores/<model>/")
    ap.add_argument("--evidence", default="",
                    help="s3 prefix for progress.json and bootstrap.log")
    ap.add_argument("--prefetch", type=int, default=8)
    args = ap.parse_args(argv)
    if args.model != "dsv4-flash":
        ap.error("store build is only wired for dsv4-flash (headers-based universe)")

    spec = config.MODELS[args.model]
    src = args.src_root.resolve()
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    root = config.s3_prefix("stores", args.model)
    t0 = time.monotonic()

    p3 = load_p3(src)
    if args.s3:
        restore_small_files(root, build_dir)
    manifest, records = p3.load_universe(None, None, src / spec["headers"], None)
    cutoff = lost_bucket_cutoff(
        read_bucket_set(build_dir / p3.JOURNAL_FILE),
        read_bucket_set(build_dir / p3.PUBLISH_JOURNAL),
        local_segment_buckets(build_dir / "segments"))
    if cutoff is not None:
        removed = prune_journal_from(build_dir / p3.JOURNAL_FILE, cutoff)
        print(f"[resume] bucket {cutoff} and later were committed but not pushed; "
              f"dropped {removed} journal lines to rebuild them", flush=True)

    remote = CountingRangeSource(
        repository=manifest.get("model", p3.PINNED_REPOSITORY),
        revision=manifest.get("revision", p3.PINNED_REVISION))
    requests = [(str(rng[5]) if len(rng) > 5 else rec["shard"], int(rng[1]), int(rng[2]))
                for rec in records for rng in rec["ranges"]]
    source = p3.PrefetchRangeSource(remote, requests, workers=args.prefetch)

    job = p3.P3KaggleJob(
        manifest, records, source, build_dir,
        publisher=S3SegmentPublisher(root), evict_after_push=True,
        verify_before_push=True, interleave=True, poll_seconds=0.5)

    n_buckets = int(manifest.get("n_buckets", manifest["n_layers"]))
    beat = Heartbeat(job=job, source=remote, build_dir=build_dir, root=root,
                     evidence=args.evidence or root, n_buckets=n_buckets, t0=t0)
    thread = threading.Thread(target=beat.run, daemon=True)
    thread.start()
    try:
        report = job.run()
    finally:
        beat.stop.set()
        source.close()
    beat.beat()
    success = bool(report.get("success"))
    print(json.dumps({"success": success, "error": report.get("error"),
                      "pushed_buckets": report.get("pushed_buckets"),
                      "push_errors": report.get("push_errors")}, indent=2), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
