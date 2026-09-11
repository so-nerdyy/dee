"""Phase 3 prototype tests: full-universe expert store.

Covers, at prototype scale and against the REAL committed headers:

  * manifest generation + the deterministic name <-> (bucket, expert)
    mapping proved bijective over all 70,656 routed tensors (43 main-model
    layers as buckets 0..42 + 3 mtp draft layers as buckets 43..45);
  * completeness: every one of the 11,776 universe pairs resolves, no
    orphan or duplicate routed tensor exists in the committed headers, and
    the main-only scope (43 buckets / 11,008 pairs) is still provable;
  * single-file dee4-v2 build on a synthetic mini-universe (existing reader
    format), including crash/resume at record granularity;
  * segmented dee4-v4 build, resume, verify and in-place repair;
  * demand-paged lazy store (dee4-v5-lazy): arbitrary lookups materialize
    exact records, audit detects and repairs torn records;
  * OPTIONAL real-bytes check (network): fetch real expert records from the
    pinned HF revision and cross-check scale bytes against the committed
    256 MiB shard prefix in the main checkout when present.

Run:  python -m pytest dee.cpp/tools/phase3/test_p3_full_store.py -q
(from the worktree root; no network needed except the marked test)
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p3_builder  # noqa: E402
import p3_completeness  # noqa: E402
import p3_lazy_store  # noqa: E402
import p3_manifest  # noqa: E402

REPO = HERE.parents[2]  # worktree root
REPORTS = REPO / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4"
HEADERS = REPORTS / "shard-headers"
# The real 256 MiB shard-2 prefix lives only in the main checkout (untracked
# artifact, read-only for us).
MAIN_CHECKOUT = REPO.parents[2]
PARTIAL_SHARD = (
    MAIN_CHECKOUT
    / "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/"
      "ds10-probe-output-20260827/partial-shard.bin"
)


# ---------------------------------------------------------------------------
# Synthetic mini-universe fixtures
# ---------------------------------------------------------------------------

# A mini universe with the same structure: L layers x E experts, 6 tensors
# per expert (3 weights + 3 scales), one layer per shard file.
MINI_LAYERS = 3
MINI_EXPERTS = 4
MINI_COMPONENTS = (  # (component, nbytes) in DEE4 record order
    ("w1.weight", 16),
    ("w3.weight", 16),
    ("w2.weight", 24),
    ("w1.scale", 4),
    ("w3.scale", 4),
    ("w2.scale", 6),
)
MINI_RECORD_BYTES = sum(n for _c, n in MINI_COMPONENTS)  # 70


def _mini_tensor_bytes(layer: int, expert: int, component: str, n: int) -> bytes:
    """Deterministic pseudo-content: byte i = f(layer, expert, component, i)."""
    seed = (layer * 1000 + expert * 10 + len(component)) & 0xFF
    return bytes(((seed + i) & 0xFF) for i in range(n))


def make_mini_universe(tmp_path: Path):
    """Write synthetic shard files + a manifest + records for a mini universe.

    Shard file = 8-byte LE header length + JSON header + concatenated data.
    Tensors are laid out in name-sorted order inside the data section, like
    the real checkpoint.
    """
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    records = []
    for layer in range(MINI_LAYERS):
        shard = f"model-{layer:05d}-of-{MINI_LAYERS:05d}.safetensors"
        header: dict[str, dict] = {}
        data = bytearray()
        names = []
        for expert in range(MINI_EXPERTS):
            for component, nbytes in MINI_COMPONENTS:
                proj, kind = component.split(".")
                names.append((
                    f"layers.{layer}.ffn.experts.{expert}.{proj}.{kind}",
                    _mini_tensor_bytes(layer, expert, component, nbytes)))
        for name, blob in sorted(names):
            start = len(data)
            data += blob
            header[name] = {
                "dtype": "I8" if name.endswith(".weight") else "F8_E8M0",
                "shape": [1, len(blob)],
                "data_offsets": [start, start + len(blob)],
            }
        header_json = json.dumps(header, separators=(",", ":")).encode()
        with (shards_dir / shard).open("wb") as fh:
            fh.write(struct.pack("<Q", len(header_json)))
            fh.write(header_json)
            fh.write(bytes(data))
        for expert in range(MINI_EXPERTS):
            rec_offset = 0
            ranges = []
            for component, nbytes in MINI_COMPONENTS:
                proj, kind = component.split(".")
                tname = f"layers.{layer}.ffn.experts.{expert}.{proj}.{kind}"
                ranges.append([component,
                               header[tname]["data_offsets"][0],
                               nbytes, rec_offset, tname])
                rec_offset += nbytes
            records.append({
                "record_index": layer * MINI_EXPERTS + expert,
                "bucket": layer,
                "domain": "main",
                "layer": layer,
                "expert": expert,
                "shard": shard,
                "record_bytes": MINI_RECORD_BYTES,
                "ranges": ranges,
            })
    manifest = {
        "schema": "mini-universe",
        "model": "synthetic",
        "revision": "test",
        "n_layers": MINI_LAYERS,
        "n_buckets": MINI_LAYERS,
        "experts_per_layer": MINI_EXPERTS,
        "total_experts": MINI_LAYERS * MINI_EXPERTS,
        "record_bytes": MINI_RECORD_BYTES,
        "store_bytes": MINI_LAYERS * MINI_EXPERTS * MINI_RECORD_BYTES,
        "universe_sha256": "0" * 64,
        "manifest_sha256": "0" * 64,
        "components": [
            {
                "component": c,
                "dtype": "I8" if c.endswith("weight") else "F8_E8M0",
                "shape": [1, n],
                "nbytes": n,
                "record_offset": off,
            }
            for off, (c, n) in zip(
                _mini_offsets(), MINI_COMPONENTS)
        ],
    }
    return shards_dir, manifest, records


def _mini_offsets() -> list[int]:
    off = 0
    out = []
    for _c, n in MINI_COMPONENTS:
        out.append(off)
        off += n
    return out


def expected_record_blob(layer: int, expert: int) -> bytes:
    return b"".join(
        _mini_tensor_bytes(layer, expert, c, n)
        for c, n in MINI_COMPONENTS)


# ---------------------------------------------------------------------------
# Real committed-header tests (no weight bytes needed)
# ---------------------------------------------------------------------------


def test_real_manifest_and_completeness(tmp_path):
    # Full universe: 43 main layers + 3 mtp draft layers = 46 buckets.
    manifest = p3_manifest.build_manifest(HEADERS)
    assert manifest["n_buckets"] == 46
    assert manifest["n_layers"] == 43
    assert manifest["n_mtp_buckets"] == 3
    assert manifest["total_experts"] == 11_776
    assert manifest["record_bytes"] == 13_369_344
    assert manifest["store_bytes"] == 157_437_394_944
    assert manifest["store_gib"] == round(157_437_394_944 / (1 << 30), 4)
    assert manifest["experts_per_layer"] == 256
    # Bucket -> shard binding must be exact for all 46 buckets.
    for bucket in range(46):
        entry = manifest["bucket_shards"][str(bucket)]
        if bucket < 43:
            assert entry["shard"] == \
                f"model-{bucket + 2:05d}-of-00048.safetensors"
            assert entry["domain"] == "main"
        else:
            assert entry["shard"] == \
                f"model-{46 + bucket - 43:05d}-of-00048.safetensors"
            assert entry["domain"] == "mtp"

    paths = p3_manifest.write_manifest(manifest, tmp_path)
    records = p3_manifest.load_records(paths["records"])
    assert len(records) == 11_776

    report = p3_completeness.audit_headers_vs_manifest(
        HEADERS, p3_manifest.load_manifest(paths["manifest"]), records)
    assert report["all_pairs_covered"]
    assert report["pairs"] == 11_776
    assert report["routed_tensors_in_scope"] == 70_656
    assert report["routed_tensors_total"] == 70_656
    assert report["routed_tensors_out_of_scope"] == 0
    assert report["orphan_tensors"] == 0

    # Main-model-only scope: 43 buckets / 11,008 pairs / 137.0625 GiB; the
    # audit must then report the 4,608 mtp tensors as out-of-scope (counted,
    # never silently dropped).
    main = p3_manifest.build_manifest(HEADERS, n_buckets=43)
    assert main["total_experts"] == 11_008
    assert main["store_bytes"] == 147_169_738_752
    assert main["store_gib"] == 137.0625
    main_paths = p3_manifest.write_manifest(main, tmp_path / "main_only")
    main_records = p3_manifest.load_records(main_paths["records"])
    main_report = p3_completeness.audit_headers_vs_manifest(
        HEADERS, p3_manifest.load_manifest(main_paths["manifest"]),
        main_records)
    assert main_report["pairs"] == 11_008
    assert main_report["routed_tensors_in_scope"] == 66_048
    assert main_report["routed_tensors_total"] == 70_656
    assert main_report["routed_tensors_out_of_scope"] == 4_608


def test_name_mapping_bijective_over_headers():
    """Every routed name in the committed headers parses; every universe
    pair's six names exist; the map inverts exactly."""
    headers = p3_manifest.load_committed_headers(HEADERS)
    pairs = set()
    for _shard, header in headers.items():
        for name in header:
            if name == "__metadata__":
                continue
            parsed = p3_manifest.parse_expert_tensor_name(name)
            if parsed is None:
                continue
            bucket, expert, proj, kind = parsed
            assert 0 <= bucket < 46 and 0 <= expert < 256
            assert proj in ("w1", "w2", "w3") and kind in ("weight", "scale")
            # inverse must rebuild the identical name
            rebuilt = p3_manifest.tensor_name(bucket, expert, proj, kind)
            assert rebuilt == name
            pairs.add((bucket, expert))
    assert len(pairs) == 11_776
    assert pairs == {
        (b, e) for b in range(46) for e in range(256)}

    # The mapper itself: main names -> same-numbered bucket, mtp -> +43.
    assert p3_manifest.parse_expert_tensor_name(
        "layers.0.ffn.experts.0.w1.scale") == (0, 0, "w1", "scale")
    assert p3_manifest.parse_expert_tensor_name(
        "layers.42.ffn.experts.255.w2.weight") == (42, 255, "w2", "weight")
    assert p3_manifest.parse_expert_tensor_name(
        "mtp.0.ffn.experts.0.w1.scale") == (43, 0, "w1", "scale")
    assert p3_manifest.parse_expert_tensor_name(
        "mtp.2.ffn.experts.255.w3.weight") == (45, 255, "w3", "weight")
    # And non-expert tensors never parse.
    assert p3_manifest.parse_expert_tensor_name(
        "layers.7.ffn.shared_experts.w1.weight") is None
    assert p3_manifest.parse_expert_tensor_name(
        "layers.7.ffn.gate.weight") is None
    assert p3_manifest.parse_expert_tensor_name(
        "layers.0.ffn.gate.tid2eid") is None
    assert p3_manifest.parse_expert_tensor_name("embed.weight") is None


def test_sealed_trace_bank_is_a_proper_subset():
    """The existing trace bank covers only 2,364 of 11,008 pairs."""
    candidates = sorted(REPORTS.glob("v60-*/dee4-metadata.json"))
    if not candidates:
        pytest.skip("v60 dee4 metadata not in this checkout")
    meta = json.loads(candidates[0].read_text())
    assert meta["format"] == "dee4-v3-trace"
    selected = {(int(r["layer"]), int(r["expert"])) for r in meta["records"]}
    universe = {(l, e) for l in range(43) for e in range(256)}
    assert selected < universe
    assert len(selected) == 2_364
    missing = universe - selected
    assert len(missing) == 11_008 - 2_364
    # Every routed pair NOT in the bank fails closed on that bank; the full
    # store must cover all of them.
    assert all(0 <= l < 43 and 0 <= e < 256 for l, e in missing)


# ---------------------------------------------------------------------------
# Synthetic build / lookup / resume tests
# ---------------------------------------------------------------------------


def test_single_file_build_and_arbitrary_lookup(tmp_path):
    shards_dir, manifest, records = make_mini_universe(tmp_path)
    out = tmp_path / "store"
    source = p3_builder.LocalShardSource(shards_dir)
    report = p3_builder.build_single(manifest, records, source, out)
    assert report["success"]
    assert report["records_written"] == MINI_LAYERS * MINI_EXPERTS

    data = (out / "experts.dee4").read_bytes()
    assert len(data) == manifest["store_bytes"]
    # Arbitrary lookup: every pair materializes byte-exact at its stride.
    for layer in range(MINI_LAYERS):
        for expert in range(MINI_EXPERTS):
            idx = layer * MINI_EXPERTS + expert
            off = idx * MINI_RECORD_BYTES
            assert data[off:off + MINI_RECORD_BYTES] == \
                expected_record_blob(layer, expert)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["format"] == "dee4-v2"
    assert meta["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert meta["record_bytes"] == MINI_RECORD_BYTES

    ver = p3_builder.verify_store(out)
    assert ver["checked_records"] == MINI_LAYERS * MINI_EXPERTS


def test_single_file_resume_after_torn_tail(tmp_path):
    """Crash mid-write -> resume must truncate the torn record and finish."""
    shards_dir, manifest, records = make_mini_universe(tmp_path)
    out = tmp_path / "store"
    source = p3_builder.LocalShardSource(shards_dir)

    # Simulate a crashed build: partial data file with 5 whole records +
    # a torn 6th; journal committed through 4; integrity has 5 lines.
    out.mkdir()
    partial = out / "experts.dee4.partial"
    blob = bytearray()
    for i in range(5):
        r = records[i]
        blob += expected_record_blob(r["layer"], r["expert"])
    partial.write_bytes(bytes(blob) + b"\xde\xad")  # torn tail
    with (out / "integrity.jsonl").open("w", newline="\n") as fh:
        for i in range(5):
            r = records[i]
            fh.write(json.dumps({
                "layer": r["layer"], "expert": r["expert"],
                "record_index": i, "record_offset": i * MINI_RECORD_BYTES,
                "record_bytes": MINI_RECORD_BYTES,
                "record_sha256": hashlib.sha256(
                    expected_record_blob(r["layer"], r["expert"])).hexdigest(),
                "component_sha256": {}, "source_shards": [r["shard"]],
            }, separators=(",", ":")) + "\n")
    with (out / "build.journal.jsonl").open("w", newline="\n") as fh:
        fh.write(json.dumps({"committed_through": 4, "records": 4}) + "\n")

    # committed_through(4) < integrity lines(5): the 5th line is dropped and
    # the file truncated to 4 records, then the build completes all 12.
    report = p3_builder.build_single(manifest, records, source, out)
    assert report["success"]
    data = (out / "experts.dee4").read_bytes()
    assert len(data) == manifest["store_bytes"]
    for i, r in enumerate(records):
        off = i * MINI_RECORD_BYTES
        assert data[off:off + MINI_RECORD_BYTES] == \
            expected_record_blob(r["layer"], r["expert"])


def test_segmented_build_verify_and_repair(tmp_path):
    shards_dir, manifest, records = make_mini_universe(tmp_path)
    out = tmp_path / "segstore"
    source = p3_builder.LocalShardSource(shards_dir)
    report = p3_builder.build_segmented(manifest, records, source, out)
    assert report["success"]
    assert len(report["buckets_built"]) == MINI_LAYERS

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["format"] == "dee4-v4-segmented"
    assert len(meta["segments"]) == MINI_LAYERS

    # Re-run is a no-op (journal commits every bucket).
    again = p3_builder.build_segmented(manifest, records, source, out)
    assert again["buckets_built"] == []
    assert len(again["buckets_skipped"]) == MINI_LAYERS

    ver = p3_builder.verify_store(out)
    assert ver["checked_records"] == MINI_LAYERS * MINI_EXPERTS

    # Corrupt one record, detect, repair in place.
    seg = meta["segments"][1]
    seg_path = out / seg["file"]
    with seg_path.open("r+b") as fh:
        fh.seek(2 * MINI_RECORD_BYTES)
        fh.write(b"\x00" * MINI_RECORD_BYTES)
    with pytest.raises(ValueError):
        p3_builder.verify_store(out)
    fix = p3_builder.repair_record(
        out, 1 * MINI_EXPERTS + 2, source, records)
    assert fix["integrity_rewritten"] is False  # integrity was already right
    assert p3_builder.verify_store(out)["success"]


def test_lazy_store_materializes_on_demand(tmp_path):
    shards_dir, manifest, records = make_mini_universe(tmp_path)
    out = tmp_path / "lazy"
    lazy = p3_lazy_store.LazyFullStore(out, manifest)
    lazy.create()
    source = p3_builder.LocalShardSource(shards_dir)

    # The data file is full-size but zero-content; nothing materialized.
    assert (out / "experts.dee4").stat().st_size == manifest["store_bytes"]
    assert lazy.materialized_count() == 0
    with pytest.raises(KeyError):
        lazy.read(1, 1, source, records, fill=False)

    # Arbitrary order materialization.
    for layer, expert in [(2, 3), (0, 0), (1, 2), (2, 3)]:
        blob = lazy.read(layer, expert, source, records)
        assert blob == expected_record_blob(layer, expert)
    assert lazy.materialized_count() == 3

    audit = lazy.audit()
    assert audit["checked"] == 3 and audit["corrupt"] == []

    # Corrupt a materialized record: audit + source repairs it.
    idx = 1 * MINI_EXPERTS + 2
    with (out / "experts.dee4").open("r+b") as fh:
        fh.seek(idx * MINI_RECORD_BYTES)
        fh.write(b"\xAA" * MINI_RECORD_BYTES)
    audit = lazy.audit(source=source, records=records)
    assert audit["corrupt"] == [idx] and audit["repaired"] == 1
    assert lazy.read(1, 2, source, records, fill=False) == \
        expected_record_blob(1, 2)

    # Materialize the rest and finalize -> identical to a prebuilt store.
    for layer in range(MINI_LAYERS):
        for expert in range(MINI_EXPERTS):
            lazy.read(layer, expert, source, records)
    assert lazy.materialized_count() == MINI_LAYERS * MINI_EXPERTS
    done = lazy.finalize()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["format"] == "dee4-v2"
    expected = b"".join(
        expected_record_blob(r["layer"], r["expert"]) for r in records)
    assert (out / "experts.dee4").read_bytes() == expected
    assert done["data_sha256"] == hashlib.sha256(expected).hexdigest()


# ---------------------------------------------------------------------------
# C++ reader proof: real Dee4ExpertStore over full-universe geometry
# ---------------------------------------------------------------------------


def _clangxx() -> str | None:
    import shutil
    return shutil.which("clang++") or shutil.which("g++")


@pytest.mark.slow
def test_cpp_dee4_reader_covers_full_universe(tmp_path):
    """The SHIPPED dee4-v2 reader resolves arbitrary (layer, expert) over the
    full 43x256 universe on a sparse 147,169,738,752-byte data file."""
    cxx = _clangxx()
    if cxx is None:
        pytest.skip("no C++ compiler")
    import p3_sparse_store
    store_dir = tmp_path / "fullsparse"
    try:
        p3_sparse_store.create_sparse_store(store_dir, HEADERS)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    build = tmp_path / "cxx"
    build.mkdir()
    dee = REPO / "dee.cpp"
    objects = []
    for src in ("expert_store.cpp", "weight_mmap.cpp", "json_min.cpp"):
        obj = build / (src + ".o")
        import subprocess
        subprocess.run(
            [cxx, "-std=c++20", "-Iinclude", "-c", f"src/{src}", "-o",
             str(obj)], cwd=dee, check=True)
        objects.append(str(obj))
    exe = build / "test_p3_full_geometry.exe"
    import subprocess
    subprocess.run(
        [cxx, "-std=c++20", "-Iinclude",
         str(HERE / "test_p3_dee4_full_geometry.cpp"), *objects,
         "-o", str(exe)], cwd=dee, check=True)
    proc = subprocess.run(
        [str(exe), str(store_dir)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ALL PASS" in proc.stdout


# ---------------------------------------------------------------------------
# Optional real-bytes check (network to huggingface.co required)
# ---------------------------------------------------------------------------


def _real_source() -> p3_builder.RemoteRangeSource:
    return p3_builder.RemoteRangeSource()


@pytest.mark.network
def test_real_record_fetch_matches_committed_ranges(tmp_path):
    """Fetch 2 real expert records from HF; verify determinism and, when the
    main checkout's 256 MiB shard-2 prefix is present, cross-check the real
    scale bytes inside the assembled record."""
    manifest = p3_manifest.build_manifest(HEADERS)
    records = manifest.pop("_records")
    src = _real_source()
    # One expert from an early layer (0 -> shard 2) and one from the last.
    for layer, expert in [(0, 7), (42, 200)]:
        rec = records[layer * 256 + expert]
        blob, comp_sha, srcs = p3_builder.assemble_record(src, rec)
        assert len(blob) == 13_369_344
        assert srcs == [f"model-{layer + 2:05d}-of-00048.safetensors"]
        assert len(comp_sha) == 6
        # Second fetch must be byte-identical (pinned revision).
        blob2, comp_sha2, _ = p3_builder.assemble_record(src, rec)
        assert blob2 == blob and comp_sha2 == comp_sha

    # Cross-check real scale bytes against the local partial shard prefix.
    prefix = PARTIAL_SHARD
    if prefix.is_file():
        raw = prefix.read_bytes()
        hlen = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + hlen])
        base = 8 + hlen
        rec = records[7]  # layer 0, expert 7
        for component, data_offset, nbytes, record_offset, tensor in \
                rec["ranges"]:
            if component.endswith(".scale"):
                local = raw[base + data_offset: base + data_offset + nbytes]
                remote = src.fetch(rec["shard"], data_offset, nbytes)
                assert local == remote, f"{tensor}: prefix != remote"
