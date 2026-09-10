#!/usr/bin/env python3
"""Aggregate full-sweep CSV -> DEE4_CODEC_RESULTS.csv + console table."""
import csv
import statistics
from collections import defaultdict

SRC = "dee.cpp/experiments/dee4_lossless/scan-live-full/scan_out.csv"
DST = "research/dee4-lossless/DEE4_CODEC_RESULTS.csv"

rows = list(csv.DictReader(open(SRC)))
print("rows:", len(rows))


def pct(s, q):
    s = sorted(s)
    return s[min(len(s) - 1, int(q * len(s)))]


agg = defaultdict(list)
tp_c = defaultdict(list)
tp_d = defaultdict(list)
ver = defaultdict(int)
for r in rows:
    if r["error"] or r["codec"] in ("zeros",):
        continue
    o = int(r["orig"])
    v = float(r["cbytes"]) if r["cbytes"] else float(r["est_bytes"])
    k = (r["region"], r["codec"])
    agg[k].append(v / o)
    if r["c_ms"]:
        tp_c[k].append(o / 2**20 / (float(r["c_ms"]) / 1000))
    if r["d_ms"]:
        tp_d[k].append(o / 2**20 / (float(r["d_ms"]) / 1000))
    if r["verified"] == "True":
        ver[k] += 1

codecs = ["lz4_fast", "lz4_hc", "zstd_1", "zstd_3", "zstd_10",
          "nib_split_zstd3", "order0", "order1", "rans"]
regions = ["w1w", "w3w", "w2w", "w1s", "w3s", "w2s", "whole"]
out = []
for region in regions:
    for codec in codecs:
        v = agg.get((region, codec))
        if not v:
            continue
        n = len(v)
        row = {
            "region": region, "codec": codec, "n": n,
            "mean_ratio": round(statistics.fmean(v), 4),
            "median_ratio": round(statistics.median(v), 4),
            "p95_ratio": round(pct(v, 0.95), 4),
            "worst_ratio": round(max(v), 4),
            "best_ratio": round(min(v), 4),
            "c_MB_s_med": round(statistics.median(tp_c[(region, codec)])
                                if tp_c[(region, codec)] else 0),
            "d_MB_s_med": round(statistics.median(tp_d[(region, codec)])
                                if tp_d[(region, codec)] else 0),
            "verified_n": ver[(region, codec)],
            "independent": "yes",
            "streaming": "yes" if codec.startswith(
                ("lz4", "zstd", "nib")) else "n/a-bound",
            "random_access": "yes",
        }
        out.append(row)
        print(f"{region:6s} {codec:16s} mean={row['mean_ratio']:.4f} "
              f"p95={row['p95_ratio']:.4f} worst={row['worst_ratio']:.4f} "
              f"c={row['c_MB_s_med']}MB/s d={row['d_MB_s_med']}MB/s "
              f"n={n} verified={row['verified_n']}")

with open(DST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("wrote", DST)
