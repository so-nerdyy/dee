"""Phase 3: create a full-universe-geometry dee4-v2 store backed by a sparse
file (Windows/NTFS) so arbitrary-lookup tests need no real 147 GiB.

The sparse file reports the true 157,437,394,944-byte size (46 buckets:
43 main layers + 3 mtp draft layers), so ``Dee4ExpertStore.open``'s strict
``size == total_experts * record_bytes`` check passes; only seeded record
extents consume disk.

Usage:
    python tools/phase3/p3_sparse_store.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_manifest  # noqa: E402


def _make_sparse(path: Path, size: int) -> bool:
    """Mark ``path`` sparse on NTFS and extend it to ``size`` bytes.

    NOTE: ``fsutil sparse setflag`` + ``fsutil file seteof`` is the reliable
    recipe on this platform.  A Python ``truncate`` after FSCTL_SET_SPARSE
    was observed to allocate all clusters anyway (137 GiB of writes) on this
    NTFS volume, so the ctypes path is not used.
    """
    if sys.platform != "win32":
        return False
    import subprocess
    path.touch()
    if subprocess.run(
            ["fsutil", "sparse", "setflag", str(path)],
            capture_output=True).returncode != 0:
        return False
    if subprocess.run(
            ["fsutil", "file", "seteof", str(path), str(size)],
            capture_output=True).returncode != 0:
        return False
    return path.stat().st_size == size


def marker(record_index: int) -> bytes:
    return bytes(((record_index & 0xFF) + j) & 0xFF for j in range(16))


def create_sparse_store(out_dir: Path | str, headers_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = p3_manifest.build_manifest(headers_dir)
    record_bytes = int(manifest["record_bytes"])
    total = int(manifest["total_experts"])
    size = total * record_bytes

    data_path = out_dir / "experts.dee4"
    if sys.platform == "win32":
        if not _make_sparse(data_path, size):
            raise RuntimeError(
                "NTFS sparse allocation unavailable; need a filesystem that "
                "supports sparse files to host the 137 GiB fixture")
    else:
        # POSIX truncate creates a sparse (unallocated) file.
        with data_path.open("wb") as fh:
            fh.truncate(size)

    # Seed marker bytes at the start and end of sampled records so the C++
    # test can verify real content, not only resolution.  Buckets 43..45 are
    # the mtp draft layers.
    seeded = [(0, 0), (20, 128), (42, 255), (43, 0), (45, 255)]
    with data_path.open("r+b") as fh:
        for bucket, expert in seeded:
            idx = bucket * 256 + expert
            off = idx * record_bytes
            fh.seek(off)
            fh.write(marker(idx))
            fh.seek(off + record_bytes - 16)
            fh.write(marker(idx))

    from p3_builder import _store_metadata  # shared metadata shape
    metadata = _store_metadata(manifest, "dee4-v2")
    metadata["data_file"] = "experts.dee4"
    metadata["data_sha256"] = "0" * 64  # sparse fixture: content not sealed
    metadata["total_bytes"] = size
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: p3_sparse_store.py <out_dir>")
    repo = Path(__file__).resolve().parents[3]
    headers = (
        repo / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
             "shard-headers")
    create_sparse_store(sys.argv[1], headers)
    print(f"sparse full-universe store at {sys.argv[1]}")
