"""Build the dense-only DSv4-Flash checkpoint and (optionally) sync it to S3.

CPU job. Range-reads only the tensors the segmented-store runner needs
(1564 names: model-level, per-layer dense, shared experts) from the pinned HF
revision. Tensors are written as they arrive, so peak memory is one tensor
(the largest is the 1.06 GB embedding). Each shard's header is checked against
the committed shard headers and the total length against the server. Resume-
safe: shards with a finished state record are skipped on restart.

HF_TOKEN / HUGGING_FACE_HUB_TOKEN are honored (anonymous reads hit HTTP 429).

  python dense_extract.py --model dsv4-flash --repo-root <clone> \
      --out-dir <dir> [--s3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import config
import dense_manifest as dm
from hf_source import USER_AGENT, CountingRangeSource


def needed_tensor_names(repo_root: Path) -> list[str]:
    sys.path.insert(0, str(repo_root / "dee.cpp"))
    from scripts import deepseek_v4_support as sup

    cfg_path = (repo_root / "dee.cpp/benchmark_reports/"
                "deepseek-v4-flash-0731-t4/official-source/inference/config.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    n_layers = int(cfg["n_layers"])
    n_hash = int(cfg["n_hash_layers"])
    ratios = cfg["compress_ratios"]
    names = set(sup.model_level_tensor_names())
    for layer in range(n_layers):
        names.update(sup.layer_dense_tensor_names(
            layer, hash_layer=layer < n_hash, compress_ratio=ratios[layer]))
        names.update(sup.shared_expert_tensor_names(layer))
    return sorted(names)


def committed_headers(headers_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(headers_dir.glob("model-*-of-00048.safetensors.json")):
        shard = path.name[: -len(".json")]
        out[shard] = json.loads(path.read_text(encoding="utf-8"))
    if len(out) != 48:
        raise RuntimeError(f"expected 48 committed shard headers, got {len(out)}")
    return out


def _abs_range(source: CountingRangeSource, shard: str, start: int, nbytes: int) -> bytes:
    data = source._range(source._url(shard), start, start + nbytes - 1)
    if len(data) != nbytes:
        raise IOError(f"{shard}: short range {start}+{nbytes}: got {len(data)}")
    return data


def remote_total_size(source: CountingRangeSource, shard: str) -> int:
    headers = {"Range": "bytes=0-0", "User-Agent": USER_AGENT}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(source._url(shard), headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        content_range = resp.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"{shard}: no Content-Range total in response")
    return int(content_range.rsplit("/", 1)[1])


def extract_shard(source: CountingRangeSource, shard: str, names: list[str],
                  committed: dict[str, dict], blob_path: Path) -> dict:
    prefix = _abs_range(source, shard, 0, 8)
    hlen = dm.prefix_header_len(prefix)
    header = prefix + _abs_range(source, shard, 8, hlen)
    remote = json.loads(header[8:].decode("utf-8"))
    for name in names:
        if remote.get(name, {}).get("data_offsets") != committed[name]["data_offsets"]:
            raise RuntimeError(f"{shard}: header for {name} disagrees with committed")
    data_ends = [meta["data_offsets"][1] for key, meta in remote.items()
                 if key != "__metadata__"]
    file_size = dm.shard_file_size(header, data_ends)
    total = remote_total_size(source, shard)
    if total != file_size:
        raise RuntimeError(f"{shard}: server length {total} != header-derived {file_size}")

    entries = []
    digest = hashlib.sha256()
    blob_off = 0
    tmp_path = blob_path.parent / (blob_path.name + ".tmp")
    with tmp_path.open("wb") as fh:
        for name in sorted(names, key=lambda n: remote[n]["data_offsets"][0]):
            start, end = remote[name]["data_offsets"]
            nbytes = end - start
            data = source.fetch(shard, start, nbytes)
            if len(data) != nbytes:
                raise IOError(f"{shard}: {name} short fetch")
            fh.write(data)
            digest.update(data)
            entries.append([name, 8 + hlen + start, nbytes, blob_off])
            blob_off += nbytes
            del data
    tmp_path.replace(blob_path)
    return dm.shard_record(
        header_bytes=header, file_size=file_size, blob=f"blobs/{blob_path.name}",
        blob_bytes=blob_off, blob_sha256=digest.hexdigest(), entries=entries)


def build(model: str, repo_root: Path, out_dir: Path) -> Path:
    spec = config.MODELS[model]
    if not spec["dense_supported"]:
        raise SystemExit(f"dense extraction not supported for {model}")
    names = needed_tensor_names(repo_root)
    committed = committed_headers(repo_root / spec["headers"])
    by_shard: dict[str, list[str]] = {}
    for name in names:
        owner = [s for s, hdr in committed.items() if name in hdr]
        if len(owner) != 1:
            raise RuntimeError(f"{name}: expected one committed owner, got {owner}")
        by_shard.setdefault(owner[0], []).append(name)

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("WARNING: HF_TOKEN not set; anonymous range reads may return HTTP 429",
              flush=True)

    source = CountingRangeSource(repository=spec["hf_repo"], revision=spec["hf_rev"])
    blobs = out_dir / "blobs"
    state = out_dir / "work"
    blobs.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict] = {}
    for shard in sorted(by_shard):
        state_file = state / f"{shard}.json"
        blob_file = blobs / f"{shard}.bin"
        if state_file.is_file() and blob_file.is_file():
            record = json.loads(state_file.read_text(encoding="utf-8"))
            if blob_file.stat().st_size == record["blob_bytes"]:
                shards[shard] = record
                print(f"resume: {shard} already extracted", flush=True)
                continue
        record = extract_shard(source, shard, by_shard[shard], committed[shard], blob_file)
        state_file.write_text(json.dumps(record), encoding="utf-8")
        shards[shard] = record
        print(f"extracted {shard}: {record['blob_bytes'] / (1 << 30):.3f} GiB "
              f"tensors={len(record['entries'])} http={source.status_counts}", flush=True)

    doc = dm.manifest(model=model, repo=spec["hf_repo"],
                      revision=spec["hf_rev"], shards=shards)
    if doc["tensor_count"] != len(names):
        raise RuntimeError(f"tensor count {doc['tensor_count']} != {len(names)}")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"manifest: {doc['tensor_count']} tensors, "
          f"{doc['total_blob_bytes']} bytes in {len(shards)} shards", flush=True)
    print(f"http_status={source.status_counts} stats={source.stats}", flush=True)
    return manifest_path


def peak_rss_kb() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dsv4-flash", choices=sorted(config.MODELS))
    parser.add_argument("--repo-root", type=Path, required=True,
                        help="clone root that contains dee.cpp/")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--s3", action="store_true",
                        help="sync blobs + manifest to the dense/<model>/ prefix")
    args = parser.parse_args(argv)
    build(args.model, args.repo_root.resolve(), args.out_dir.resolve())
    print(f"peak_rss_kb={peak_rss_kb()}", flush=True)
    if args.s3:
        subprocess.run(["aws", "s3", "sync", str(args.out_dir),
                        config.s3_prefix("dense", args.model),
                        "--exclude", "work/*", "--only-show-errors"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
