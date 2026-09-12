#!/usr/bin/env python3
"""One-off Kaggle log viewer (local use)."""
import json
import sys

recs = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(len(recs), "log records")
tail = recs[-int(sys.argv[2]) :] if len(sys.argv) > 2 else recs
for r in tail:
    data = r.get("data", "")
    print(f"[{r.get('time', 0):.0f}][{r.get('stream_name')}] {data[:220]}")
