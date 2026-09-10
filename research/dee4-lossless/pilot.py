#!/usr/bin/env python3
"""Local pilot: 8 real experts via HF ranges, full codec tier."""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import HFBackend, reconstruct, sha256_hex, load_integrity
from scan import scan_record

BASE = Path(__file__).resolve().parent / "inputs"
wm = json.loads((BASE / "model.safetensors.index.json").read_text())
wm = wm["weight_map"]
integ = load_integrity(BASE / "dee4-integrity.jsonl")

PILOT = [(0, 113), (5, 135), (13, 125), (21, 153),
         (29, 129), (35, 118), (42, 124), (0, 0)]

be = HFBackend(wm)
rows = []
t_all = time.time()
for li, ei in PILOT:
    t0 = time.time()
    rec = reconstruct(li, ei, be)
    ft = time.time() - t0
    got = sha256_hex(rec)
    want = integ[(li, ei)]["record_sha256"]
    match = "EXACT" if got == want else "MISMATCH!!"
    print(f"({li},{ei}): fetch {ft:.1f}s integrity={match}", flush=True)
    assert got == want, (li, ei)
    t0 = time.time()
    r = scan_record(li, ei, rec, "full")
    print(f"({li},{ei}): scanned {len(r)} rows in {time.time()-t0:.1f}s",
          flush=True)
    rows.extend(r)
with open(Path(__file__).resolve().parent / "pilot_out.csv",
           "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"TOTAL {time.time()-t_all:.0f}s, {len(rows)} rows -> pilot_out.csv")
