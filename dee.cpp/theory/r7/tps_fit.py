"""Fit a tokens/sec model to the r7 campaign (stdlib only).

Per GPU and batch, decode step time is modelled as
    T_step = t_fixed(GPU, b) + cold_reads_per_step * c_read(GPU)
t_fixed covers compute, host-to-GPU copies and launch overhead; c_read is the
effective cost of one 12.75 MiB record read from the store tier. Both are
solved per GPU from its h16/h64 pair (two equations, two unknowns), then a
leave-one-GPU-out check predicts each GPU from the others using only its
published memory bandwidth for t_fixed and the median c_read.

Usage: python theory/r7/tps_fit.py [runs_dir]   (from dee.cpp/)
"""
import json, math, statistics, sys
from pathlib import Path

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
REC_MIB = 13369344 / 2**20
# Published memory bandwidth, GB/s (vendor datasheets).
MEM_BW = {"L4": 300, "A10": 600, "L40S": 864, "RTXPRO6000": 1792, "2xL4": 300}
TAG = {"L4": "l4", "A10": "a10", "L40S": "l40s", "RTXPRO6000": "rtx", "2xL4": "2xl4"}


def row(gpu, b, host):
    sc = json.loads((RUNS / f"p6m-r7-{TAG[gpu]}-b{b}-h{host}-nu" / "scorecard.json").read_text())
    m = sc["metrics"]
    tps = m["decode_tps"]["measured"]
    cold_tok = m["cold_records_per_tok"]["measured"]
    return {"tps": tps, "t_step": b / tps, "cold_step": cold_tok * b,
            "hit": m["host_hit_rate"]["measured"], "b_ssd": sc["measured_b_ssd_gib_s"]}


def solve(gpu, b):
    lo, hi = row(gpu, b, 16), row(gpu, b, 64)
    c = (lo["t_step"] - hi["t_step"]) / (lo["cold_step"] - hi["cold_step"])
    t0 = hi["t_step"] - hi["cold_step"] * c
    return {"gpu": gpu, "b": b, "t_fixed_s": t0, "c_read_ms": c * 1e3,
            "eff_read_gib_s": REC_MIB / 1024 / c if c > 0 else float("inf"),
            "fio_b_ssd": lo["b_ssd"], "rows": {"h16": lo, "h64": hi}}


def loo(fits):
    out = []
    singles = [f for f in fits if f["gpu"] != "2xL4"]
    for f in singles:
        rest = [g for g in singles if g is not f]
        xs = [math.log(MEM_BW[g["gpu"]]) for g in rest]
        ys = [math.log(g["t_fixed_s"]) for g in rest]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        t0 = math.exp(my + k * (math.log(MEM_BW[f["gpu"]]) - mx))
        c = statistics.median(g["c_read_ms"] for g in rest) / 1e3
        for h in ("h16", "h64"):
            r = f["rows"][h]
            pred = f["b"] / (t0 + r["cold_step"] * c)
            out.append({"gpu": f["gpu"], "b": f["b"], "host": h, "measured": r["tps"],
                        "predicted": pred, "err_pct": 100 * (pred / r["tps"] - 1)})
    return out


def main():
    report = {"model": "T_step = t_fixed(GPU,b) + cold_reads_per_step * c_read(GPU)", "fits": [], "loo": []}
    # b=1 is not fitted: h64 ran after h16 in the same container (warmer page
    # cache) and read slower on 4 of 5 GPUs, so its pair gives a negative slope.
    for b in (8,):
        fits = [solve(g, b) for g in MEM_BW]
        report["fits"] += fits
        report["loo"] += loo(fits)
    for f in report["fits"]:
        print(f"b={f['b']} {f['gpu']:<11} t_fixed {f['t_fixed_s']:6.3f} s  c_read {f['c_read_ms']:6.2f} ms/rec"
              f"  eff {f['eff_read_gib_s']:6.2f} GiB/s  (fio {f['fio_b_ssd']:.2f})")
    for l in report["loo"]:
        print(f"LOO b={l['b']} {l['gpu']:<11} {l['host']}  measured {l['measured']:.3f}  "
              f"predicted {l['predicted']:.3f}  err {l['err_pct']:+.0f}%")
    errs = [abs(l["err_pct"]) for l in report["loo"]]
    report["loo_median_abs_err_pct"] = statistics.median(errs)
    print(f"median |err| {statistics.median(errs):.0f}%  max {max(errs):.0f}%")
    (Path(__file__).parent / "tps_fit.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
