"""Record-level byte equivalence: safetensors source <-> dee4 segmented store.

WHY THIS EXISTS
---------------
The GPU-2 phase-3 run (commit 71320f0c, notebook v12) proved that the full
46-bucket ``dee4-v4-segmented`` store serves arbitrary prompts, but its
safetensors reference arm (A0) timed out at ~14,400 s after producing only a
12-token / 553-record partial for prompt 0.  End-to-end A0-vs-A1 token equality
is therefore unproven.

Re-running A0 is expensive (~80 min/prompt, and the cap was not even enough for
one prompt).  It is also *confounded*: A0 ran ``NATIVE_SOURCE_READ_LANES=1`` and
A1 ran ``lanes=4``, which are different materialization paths, so a divergence
would not localize to the store.

This tool answers the narrower and more useful question directly:

    For every (layer, expert) record the engine actually served, are the bytes
    in the segmented store identical to the bytes assembled from the
    authoritative safetensors checkpoint?

That isolates the store from the kernel entirely, and costs
O(n_records x 12.75 MiB) instead of three full inferences.

WHAT IT PROVES / DOES NOT PROVE
-------------------------------
Proves: byte provenance of the checked records -- the store returns exactly the
checkpoint's bytes, in DEE4 component order, at the right record offsets.

Does NOT prove: that the *engine* read those records correctly, nor anything
about records outside the checked set.  ``--scope journal`` checks only the
pairs that appear in the supplied route journals; ``--scope all`` checks the
entire 11,776-record universe and is the strongest store-level claim available.

USAGE
-----
Scope to the records three prompts actually touched::

    python p3_journal_equivalence.py \
        --store /tmp/dee4-full \
        --shards /kaggle/input/datasets/nivind/deepseek-v4-flash-0731-shards \
        --journal gpu2-out/routed_experts-a1-q0.jsonl \
        --journal gpu2-out/routed_experts-a1-q1.jsonl \
        --journal gpu2-out/routed_experts-a1-q2.jsonl \
        --out equivalence-report.json

Full-universe audit (no journals needed)::

    python p3_journal_equivalence.py --store /tmp/dee4-full \
        --shards <dir> --scope all --out full-audit.json

Cheap pre-flight (structure + segment sizes only, no hashing)::

    python p3_journal_equivalence.py --store /tmp/dee4-full \
        --shards <dir> --dry-run

Remote source instead of mounted shards (no local checkpoint needed)::

    python p3_journal_equivalence.py --store /tmp/dee4-full \
        --source remote --journal ... --out report.json

Exit status is 0 only when every checked record matched; 1 on any mismatch or
error.  The report is written even on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p3_manifest  # noqa: E402
from p3_builder import (  # noqa: E402
    LocalShardSource,
    RemoteRangeSource,
    assemble_record,
)

RECORD_BYTES = p3_manifest.RECORD_BYTES
EXPERTS_PER_LAYER = p3_manifest.EXPERTS_PER_LAYER


# ---------------------------------------------------------------------------
# Route journal parsing
# ---------------------------------------------------------------------------


def iter_journal_pairs(path: Path) -> Iterable[tuple[int, int]]:
    """Yield every (layer, expert) pair referenced by a route journal.

    The journal schema (RoutedExpertJournal in deepseek_v4_native_generate.py)
    writes one JSON object per (forward_step, layer) with the compact route
    matrix under ``expert_ids_rank_order`` as [token_rows][topk].  A record's
    ``layer`` is the bucket index: main layers are 0..42 and the mtp draft
    head -- if it is ever journaled -- occupies 43..45.
    """
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                # A truncated tail is expected for partial/timed-out journals
                # (A0's q0 died mid-write).  Everything before it is still
                # valid evidence, so warn and stop rather than discard.
                print(f"  ! {path.name}: truncated at line {lineno} ({exc});"
                      f" using the {lineno - 1} complete records",
                      file=sys.stderr)
                return
            layer = rec.get("layer")
            rows = rec.get("expert_ids_rank_order")
            if layer is None or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, list):
                    continue
                for expert in row:
                    yield int(layer), int(expert)


def collect_pairs(journals: list[Path]) -> tuple[list[tuple[int, int]], dict]:
    """Deduplicate (layer, expert) pairs across every supplied journal."""
    seen: set[tuple[int, int]] = set()
    per_journal: dict[str, int] = {}
    total_uses = 0
    for jp in journals:
        before = len(seen)
        uses = 0
        for pair in iter_journal_pairs(jp):
            seen.add(pair)
            uses += 1
        total_uses += uses
        per_journal[jp.name] = len(seen) - before
    stats = {
        "journals": [str(j) for j in journals],
        "new_pairs_per_journal": per_journal,
        "total_expert_uses": total_uses,
        "distinct_pairs": len(seen),
    }
    return sorted(seen), stats


# ---------------------------------------------------------------------------
# Segmented store reader (independent of the C++ implementation)
# ---------------------------------------------------------------------------


class SegmentedStore:
    """Minimal, read-only dee4-v4-segmented reader.

    Deliberately re-implemented here rather than reusing Dee4ExpertStore: the
    point is to check the store's *bytes* against the checkpoint with an
    independent locator, so a bug in the shipped reader cannot mask itself.
    Locator logic mirrors PHASE3_SEGMENTED_READER.md section 2.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        meta_path = (self.root if self.root.name == "metadata.json"
                     else self.root / "metadata.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"store metadata missing: {meta_path}")
        self.root = meta_path.parent
        self.meta = json.loads(meta_path.read_text("utf-8"))
        fmt = self.meta.get("format")
        if fmt != "dee4-v4-segmented":
            raise ValueError(f"expected dee4-v4-segmented, got {fmt!r}")
        self.record_bytes = int(self.meta.get("record_bytes", RECORD_BYTES))
        self.experts_per_layer = int(
            self.meta.get("experts_per_layer", EXPERTS_PER_LAYER))
        self.start_layer = int(self.meta.get("start_layer", 0))
        self.segments = list(self.meta.get("segments") or [])
        if not self.segments:
            raise ValueError("segment table is empty")
        self._fds: dict[int, int] = {}

    def structural_check(self) -> dict[str, Any]:
        """Validate the table and on-disk sizes without hashing anything."""
        problems: list[str] = []
        expected_first = 0
        for i, seg in enumerate(self.segments):
            if int(seg.get("bucket", -1)) != i:
                problems.append(f"segment {i}: bucket={seg.get('bucket')}")
            if int(seg.get("first_record", -1)) != expected_first:
                problems.append(
                    f"segment {i}: first_record={seg.get('first_record')} "
                    f"expected {expected_first}")
            count = int(seg.get("record_count", 0))
            if count != self.experts_per_layer:
                problems.append(f"segment {i}: record_count={count}")
            declared = int(seg.get("bytes", 0))
            if declared != count * self.record_bytes:
                problems.append(f"segment {i}: bytes={declared}")
            path = self.root / seg["file"]
            if not path.is_file():
                problems.append(f"segment {i}: missing {seg['file']}")
            else:
                actual = path.stat().st_size
                if actual != declared:
                    problems.append(
                        f"segment {i}: size {actual} != declared {declared}")
            expected_first += count
        return {
            "segment_count": len(self.segments),
            "total_records": expected_first,
            "ok": not problems,
            "problems": problems[:20],
        }

    def segment_table_digest(self) -> str:
        """sha256 over the ordered concatenation of declared segment seals.

        This is what Dee4ExpertStore reports as integrity_identity() for a
        segmented store, so it can be compared against the engine's runtime
        value (e.g. the 4846d482... reported by the GPU-2 A1 arm).
        """
        h = hashlib.sha256()
        for seg in self.segments:
            h.update(str(seg["sha256"]).lower().encode("ascii"))
        return h.hexdigest()

    def _fd(self, index: int) -> int:
        fd = self._fds.get(index)
        if fd is None:
            fd = os.open(str(self.root / self.segments[index]["file"]),
                         os.O_RDONLY)
            self._fds[index] = fd
        return fd

    def locate(self, layer: int, expert: int) -> tuple[int, int]:
        """Return (segment_index, byte_offset) for a (layer, expert) pair."""
        if not 0 <= expert < self.experts_per_layer:
            raise IndexError(f"expert {expert} out of range")
        record_index = ((layer - self.start_layer) * self.experts_per_layer
                        + expert)
        if record_index < 0:
            raise IndexError(f"layer {layer} below start_layer")
        for i, seg in enumerate(self.segments):
            first = int(seg["first_record"])
            if first <= record_index < first + int(seg["record_count"]):
                return i, (record_index - first) * self.record_bytes
        raise IndexError(f"record {record_index} not covered by the table")

    def read_record(self, layer: int, expert: int) -> bytes:
        index, offset = self.locate(layer, expert)
        data = os.pread(self._fd(index), self.record_bytes, offset)
        if len(data) != self.record_bytes:
            raise IOError(
                f"short read ({layer},{expert}): {len(data)}")
        return data

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def component_breakdown(record: dict[str, Any], source_blob: bytes,
                        store_blob: bytes) -> list[dict[str, Any]]:
    """Attribute a whole-record mismatch to individual components.

    Turns "record differs" into "w2.scale differs at +13107200", which is what
    actually tells you whether the bug is an offset error or a content error.
    """
    out = []
    for component, _off, nbytes, record_offset, tensor in record["ranges"]:
        lo = int(record_offset)
        hi = lo + int(nbytes)
        src = source_blob[lo:hi]
        dst = store_blob[lo:hi]
        if src == dst:
            continue
        first_bad = next(
            (i for i in range(min(len(src), len(dst))) if src[i] != dst[i]),
            min(len(src), len(dst)))
        out.append({
            "component": component,
            "tensor": tensor,
            "record_offset": lo,
            "nbytes": int(nbytes),
            "first_differing_byte_in_component": first_bad,
            "source_sha256": hashlib.sha256(src).hexdigest(),
            "store_sha256": hashlib.sha256(dst).hexdigest(),
        })
    return out


def check_one(args_tuple) -> dict[str, Any]:
    layer, expert, records_by_pair, store, source = args_tuple
    started = time.monotonic()
    try:
        record = records_by_pair[(layer, expert)]
        source_blob, component_sha, _shards = assemble_record(source, record)
        store_blob = store.read_record(layer, expert)
        match = source_blob == store_blob
        result = {
            "layer": layer,
            "expert": expert,
            "record_index": layer * EXPERTS_PER_LAYER + expert,
            "match": match,
            "source_sha256": hashlib.sha256(source_blob).hexdigest(),
            "store_sha256": hashlib.sha256(store_blob).hexdigest(),
            "seconds": round(time.monotonic() - started, 3),
        }
        if not match:
            result["components"] = component_breakdown(
                record, source_blob, store_blob)
            result["component_sha256_source"] = component_sha
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "layer": layer,
            "expert": expert,
            "match": False,
            "error": repr(exc)[:300],
            "seconds": round(time.monotonic() - started, 3),
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Byte-equivalence: safetensors source vs dee4 segmented store")
    ap.add_argument("--store", required=True,
                    help="segmented store root (or its metadata.json)")
    ap.add_argument("--shards",
                    help="directory of safetensors shards (--source local)")
    ap.add_argument("--source", choices=("local", "remote"), default="local")
    ap.add_argument("--headers",
                    default=str(Path(__file__).resolve().parents[2]
                                / "benchmark_reports"
                                / "deepseek-v4-flash-0731-t4"
                                / "shard-headers"),
                    help="committed shard-header directory")
    ap.add_argument("--journal", action="append", default=[], type=Path,
                    help="route journal (repeatable); required for --scope journal")
    ap.add_argument("--scope", choices=("journal", "all"), default="journal")
    ap.add_argument("--limit", type=int, default=0,
                    help="check at most N records (0 = no limit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("equivalence-report.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="structure + segment sizes only; no record hashing")
    ap.add_argument("--expect-identity", default="",
                    help="assert the segment-table digest equals this hex")
    args = ap.parse_args()

    report: dict[str, Any] = {
        "tool": "p3_journal_equivalence",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": args.scope,
        "store": str(args.store),
    }

    print(f"[1/5] opening store {args.store}")
    store = SegmentedStore(Path(args.store))
    structural = store.structural_check()
    report["structural"] = structural
    print(f"      segments={structural['segment_count']} "
          f"records={structural['total_records']} ok={structural['ok']}")
    if not structural["ok"]:
        for problem in structural["problems"]:
            print(f"      ! {problem}")

    digest = store.segment_table_digest()
    report["segment_table_digest"] = digest
    print(f"[2/5] segment-table digest {digest}")
    if args.expect_identity:
        ok = digest.lower() == args.expect_identity.lower()
        report["identity_matches_expected"] = ok
        print(f"      expected {args.expect_identity} -> "
              f"{'MATCH' if ok else 'MISMATCH'}")

    if args.dry_run:
        report["verdict"] = "DRY_RUN"
        args.out.write_text(json.dumps(report, indent=1))
        print(f"\ndry run complete -> {args.out}")
        return 0 if structural["ok"] else 1

    print(f"[3/5] building manifest from {args.headers}")
    manifest = p3_manifest.build_manifest(args.headers)
    # build_manifest returns its record list under "_records" (write_manifest
    # splits it out into p3_records.jsonl); "records" does not exist.
    records_by_pair = {
        (int(r["bucket"]), int(r["expert"])): r for r in manifest["_records"]}
    report["universe_sha256"] = manifest.get("universe_sha256")
    report["manifest_sha256"] = manifest.get("manifest_sha256")
    print(f"      {len(records_by_pair)} records in the universe manifest")

    if args.scope == "all":
        pairs = sorted(records_by_pair)
        report["pair_selection"] = {"mode": "all", "distinct_pairs": len(pairs)}
    else:
        if not args.journal:
            ap.error("--scope journal requires at least one --journal")
        pairs, jstats = collect_pairs(args.journal)
        report["pair_selection"] = {"mode": "journal", **jstats}
        print(f"      {jstats['total_expert_uses']} expert uses -> "
              f"{jstats['distinct_pairs']} distinct pairs")

    unknown = [p for p in pairs if p not in records_by_pair]
    if unknown:
        report["unknown_pairs"] = unknown[:20]
        print(f"      ! {len(unknown)} pairs are not in the manifest "
              f"(first: {unknown[:3]})")
        pairs = [p for p in pairs if p in records_by_pair]

    if args.limit:
        pairs = pairs[:args.limit]
    total_bytes = len(pairs) * RECORD_BYTES
    print(f"[4/5] comparing {len(pairs)} records "
          f"({total_bytes / (1 << 30):.1f} GiB per side)")

    if args.source == "local":
        if not args.shards:
            ap.error("--source local requires --shards")
        source = LocalShardSource(args.shards)
    else:
        source = RemoteRangeSource()

    results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    t0 = time.monotonic()
    work = [(l, e, records_by_pair, store, source) for l, e in pairs]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(check_one, work), 1):
            results.append(res)
            if res.get("error"):
                errors.append(res)
                print(f"      ! ({res['layer']},{res['expert']}) "
                      f"ERROR {res['error']}")
            elif not res["match"]:
                mismatches.append(res)
                print(f"      ! ({res['layer']},{res['expert']}) MISMATCH")
            if i % 100 == 0 or i == len(work):
                rate = i / max(time.monotonic() - t0, 1e-9)
                mib = (i * RECORD_BYTES) / (1 << 20) / max(
                    time.monotonic() - t0, 1e-9)
                print(f"      {i}/{len(work)}  {rate:.1f} rec/s  "
                      f"{mib:.0f} MiB/s  mismatches={len(mismatches)} "
                      f"errors={len(errors)}")
    elapsed = time.monotonic() - t0

    matched = sum(1 for r in results if r.get("match"))
    report["comparison"] = {
        "records_checked": len(results),
        "matched": matched,
        "mismatched": len(mismatches),
        "errors": len(errors),
        "bytes_per_side": total_bytes,
        "elapsed_s": round(elapsed, 1),
        "records_per_s": round(len(results) / max(elapsed, 1e-9), 2),
        "mib_per_s": round(total_bytes / (1 << 20) / max(elapsed, 1e-9), 1),
    }
    report["mismatches"] = mismatches[:50]
    report["errors"] = errors[:50]
    # Per-record detail is large; keep it only when something went wrong.
    if mismatches or errors:
        report["all_results"] = results

    clean = not mismatches and not errors and structural["ok"]
    report["verdict"] = "PASS" if clean else "FAIL"
    args.out.write_text(json.dumps(report, indent=1))

    print(f"\n[5/5] {matched}/{len(results)} records byte-identical in "
          f"{elapsed / 60:.1f} min")
    if mismatches:
        print(f"      {len(mismatches)} MISMATCHED")
        for m in mismatches[:5]:
            comps = ", ".join(c["component"] for c in m.get("components", []))
            print(f"        ({m['layer']},{m['expert']}) components: "
                  f"{comps or 'whole record'}")
    if errors:
        print(f"      {len(errors)} ERRORED")
    print(f"      verdict={report['verdict']} -> {args.out}")
    store.close()
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
