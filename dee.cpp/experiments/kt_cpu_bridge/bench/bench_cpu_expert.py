"""Microbench driver: measure CPU expert ms -> JSON for cost_model.

Does NOT claim full-model TPS. Measures the portable reference executor
(single expert, single row) and optionally the torch KT-emulated path, plus a
memcpy-based H2D proxy (pinned/ pageable as available). GPU exec must be
measured on the campaign host with dee's own T4 kernels (KT's Triton path is
SM_86/89/120-only and NOT used here); pass --t-gpu-ms manually or via JSON.

Example:
  python bench/bench_cpu_expert.py --hidden 64 --inter 32 --repeats 50
  python bench/bench_cpu_expert.py --hidden 128 --inter 64 --repeats 20 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from kt_cpu_bridge.codec import REALISTIC_SCALE_BYTES  # noqa: E402
from kt_cpu_bridge.reference import expert_forward_reference, kt_emulated_forward  # noqa: E402


def _tensors(hidden: int, inter: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    def rnd(shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    def scl(shape):
        return torch.randint(REALISTIC_SCALE_BYTES[0], REALISTIC_SCALE_BYTES[1],
                             shape, dtype=torch.int64, generator=g).to(torch.uint8)
    return (rnd((inter, hidden // 2)), scl((inter, hidden // 32)),
            rnd((hidden, inter // 2)), scl((hidden, inter // 32)),
            rnd((inter, hidden // 2)), scl((inter, hidden // 32)))


def _bench(fn, repeats: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"min_ms": ts[0], "p50_ms": ts[len(ts) // 2], "max_ms": ts[-1],
            "mean_ms": sum(ts) / len(ts), "repeats": repeats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--inter", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-gpu-ms", type=float, default=None)
    ap.add_argument("--t-h2d-ms", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    assert args.hidden % 32 == 0 and args.inter % 32 == 0
    p1, s1, p2, s2, p3, s3 = _tensors(args.hidden, args.inter, args.seed)
    torch.manual_seed(args.seed)
    x = torch.randn(1, args.hidden)
    # correctness gate before timing
    y = expert_forward_reference(x, p1, s1, p2, s2, p3, s3)
    assert torch.isfinite(y).all()
    ref = _bench(lambda: expert_forward_reference(x, p1, s1, p2, s2, p3, s3), args.repeats)
    kte = _bench(lambda: kt_emulated_forward(x, p1, s1, p2, s2, p3, s3), args.repeats)
    packed_bytes = int(3 * (args.inter * args.hidden / 2 + args.inter * args.hidden / 32))
    # H2D proxy: host memcpy of the packed record (lower bound; real H2D on host GPU may differ)
    blob = torch.randint(0, 256, (packed_bytes,), dtype=torch.uint8).numpy()
    dst = bytearray(packed_bytes)
    def _copy():
        dst[:] = blob.tobytes()
    h2d = _bench(_copy, max(10, args.repeats // 2))
    rec = {
        "hidden": args.hidden, "inter": args.inter,
        "cpu_ref_ms_per_expert": ref, "cpu_kt_emulated_ms_per_expert": kte,
        "h2d_memcpy_proxy_ms_per_expert": h2d,
        "packed_bytes_per_expert": packed_bytes,
        "t_gpu_ms_per_expert": args.t_gpu_ms, "t_h2d_ms_per_expert": args.t_h2d_ms,
        "note": "portable-torch timing only; measure t_gpu with dee T4 kernels on campaign host",
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
