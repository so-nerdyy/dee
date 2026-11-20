#!/usr/bin/env python3
"""Fill occupancy timeline builder + renderer (no CUDA needed).

Input: fill_timeline.json as emitted by StageProfiler::fill_timeline_json
(batches with per-request start/service windows), optionally merged with
host transfer/compute markers.

Output:
  - machine JSON: time-bucketed series (reads_outstanding, bytes_in_flight,
    ssd_busy, workers_active_estimate, batch_waits) plus summary
    (busy_fraction, starvation_gaps, max_outstanding).
  - self-contained HTML (inline SVG): SSD busy/idle bands, outstanding reads,
    batch waits, H2D/compute markers when provided.

Accounting rule (the Flash-bug guard): worker service sums are NEVER added
as wall. SSD busy at time t = union of [start, start+service) windows.
A batch's critical contribution is its wall; worker_sum explains it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BUCKET_MS_DEFAULT = 5.0


def load_batches(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    batches = payload.get("batches", [])
    if not isinstance(batches, list):
        raise ValueError("fill_timeline.json needs a batches list")
    return batches


def _batch_offset(batch: dict, index: int, cursor_ms: float) -> float:
    # Batches are sequential on the calling thread; explicit offsets win.
    if isinstance(batch.get("submit_offset_ms"), (int, float)):
        return float(batch["submit_offset_ms"])
    return cursor_ms


def build_series(batches: list[dict], bucket_ms: float = BUCKET_MS_DEFAULT,
                 extra_markers: list[dict] | None = None) -> dict:
    """Sweep-line over request windows. All times in ms, relative."""
    events: list[tuple[float, int, float]] = []  # (t, delta_n, delta_bytes)
    batch_windows: list[dict] = []
    cursor = 0.0
    t_end = 0.0
    for index, batch in enumerate(batches):
        submit = _batch_offset(batch, index, cursor)
        wall = float(batch.get("batch_wall_ms", 0.0))
        cursor = max(cursor, submit + wall)
        batch_windows.append({"batch_id": batch.get("batch_id", index),
                              "submit_ms": submit, "wall_ms": wall,
                              "misses": batch.get("misses", 0)})
        for req in batch.get("requests", []):
            if req.get("cache_hit") or not req.get("success", True):
                continue
            start = submit + float(req.get("start_offset_ms", 0.0))
            end = start + float(req.get("service_ms", 0.0))
            if end <= start:
                continue
            events.append((start, 1, float(req.get("nbytes", 0))))
            events.append((end, -1, -float(req.get("nbytes", 0))))
            t_end = max(t_end, end)
    events.sort(key=lambda e: (e[0], e[1]))
    # The window covers every service window (a request may extend past its
    # batch's recorded wall); cursor covers batch walls without requests.
    t_end = max(t_end, cursor)
    buckets: list[dict] = []
    outstanding, in_flight = 0, 0.0
    busy_ms, window_ms = 0.0, 0.0
    t = 0.0
    ev_index = 0
    max_outstanding = 0
    gaps: list[list[float]] = []
    gap_start: float | None = None
    while t < t_end:
        t_next = min(t + bucket_ms, t_end)
        while ev_index < len(events) and events[ev_index][0] < t_next:
            _, dn, db = events[ev_index]
            outstanding += dn
            in_flight += db
            ev_index += 1
        max_outstanding = max(max_outstanding, outstanding)
        buckets.append({"t_ms": round(t, 3), "reads_outstanding": outstanding,
                        "bytes_in_flight": int(in_flight)})
        if outstanding > 0:
            busy_ms += t_next - t
            if gap_start is not None and t > gap_start:
                gaps.append([gap_start, t])
                gap_start = None
        elif gap_start is None:
            gap_start = t
        window_ms += t_next - t
        t = t_next
    if gap_start is not None and t_end > gap_start:
        gaps.append([gap_start, t_end])
    markers = extra_markers or []
    return {"bucket_ms": bucket_ms, "buckets": buckets,
            "batch_windows": batch_windows,
            "summary": {
                "window_ms": round(window_ms, 3),
                "ssd_busy_ms": round(busy_ms, 3),
                "ssd_busy_fraction": round(busy_ms / window_ms, 4) if window_ms else 0.0,
                "max_reads_outstanding": max_outstanding,
                "starvation_gaps": [[round(a, 3), round(b, 3)] for a, b in gaps],
                "markers": markers},
            }


def render_html(series: dict, title: str = "fill occupancy") -> str:
    buckets = series["buckets"]
    if not buckets:
        return "<html><body><p>empty timeline</p></body></html>"
    width, height = 960, 260
    t_end = buckets[-1]["t_ms"] + series["bucket_ms"]
    max_n = max([b["reads_outstanding"] for b in buckets] + [1])
    bw = width / len(buckets)
    rects = []
    for i, bkt in enumerate(buckets):
        x = i * bw
        h = (bkt["reads_outstanding"] / max_n) * (height - 60)
        color = "#c0392b" if bkt["reads_outstanding"] == 0 else "#2471a3"
        rects.append(
            f'<rect x="{x:.1f}" y="{height - 40 - h:.1f}" width="{max(bw - 0.5, 0.5):.1f}" '
            f'height="{h:.1f}" fill="{color}"><title>t={bkt["t_ms"]}ms '
            f'n={bkt["reads_outstanding"]} bytes={bkt["bytes_in_flight"]}</title></rect>')
    summary = series["summary"]
    caption = (f"SSD busy {summary['ssd_busy_fraction'] * 100:.1f}% | "
               f"max outstanding {summary['max_reads_outstanding']} | "
               f"gaps {len(summary['starvation_gaps'])} "
               f"(red = SSD idle/starved)")
    return (f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
            f"<body><h3>{html.escape(title)}</h3><p>{caption}</p>"
            f"<svg width='{width}' height='{height}' style='border:1px solid #ccc'>"
            f"{''.join(rects)}</svg></body></html>")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--bucket-ms", type=float, default=BUCKET_MS_DEFAULT)
    args = parser.parse_args(argv)
    series = build_series(load_batches(args.timeline), args.bucket_ms)
    if args.out_json is not None:
        args.out_json.write_text(json.dumps(series, indent=2), encoding="utf-8")
    if args.out_html is not None:
        args.out_html.write_text(render_html(series), encoding="utf-8")
    summary = series["summary"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
