#!/usr/bin/env python3
"""DEE VRAM/RAM expert-budget estimator (derived arithmetic only).

This tool performs *derived memory arithmetic* from user-supplied budgets and
expert record sizes. It measures nothing and predicts no throughput.

Canon DSV4-Flash-0731 geometry (from
``dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/official-source/config.json``
and ``T4_FEASIBILITY.md`` / ``P2.3_FP4_VRAM_ANALYSIS.md``):

- hidden_size=4096, moe_intermediate_size=2048, n_routed_experts=256,
  num_hidden_layers=43, num_experts_per_tok=6, expert_dtype=fp4.
- One routed expert holds gate+up+down = 3 * 4096 * 2048 = 25,165,824 params.
- MXFP4 packed record (verbatim e2m1 bytes + e8m0 scales): 13,369,344 bytes
  (12.75 MiB). FP16/BF16 expanded: 50,331,648 bytes (48.0 MiB).
  INT8: 25,952,256 bytes (24.75 MiB).

Claim discipline: outputs are DERIVED arithmetic. STQ/IQ2-style hypothetical
inputs are approximate/research-only and carry NO quality claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

# --- Canon geometry -------------------------------------------------------
DSV4_FLASH_0731 = {
    "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    "architecture": "DeepseekV4ForCausalLM",
    "hidden_size": 4096,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_hidden_layers": 43,
    # dee follows T4_FEASIBILITY.md and bounds topk across all 43 layers.
    # Some layers may be dense/hash in practice; moe_layers is therefore an
    # explicit, overridable assumption, not a measured fact.
    "moe_layers_default": 43,
    "num_experts_per_tok": 6,
    "params_per_routed_expert": 3 * 4096 * 2048,  # 25,165,824
    "total_checkpoint_bytes": 166878536440,
}

EXPERT_RECORD_BYTES = {
    # Verbatim packed MXFP4 (e2m1fn nibbles + e8m0 scales), per P2.3 doc.
    "mxfp4": 13369344,
    "mxfp4_packed": 13369344,
    # Expanded execution formats (weights only, incl. scales where noted).
    "fp16": 50331648,
    "bf16": 50331648,
    "int8": 25952256,
}

# Codecs that are hypothetical / research-only: allowed for arithmetic but
# always flagged approximate with no quality claim.
APPROXIMATE_CODECS = ("stq", "iq2", "iq2_xxs", "iq1", "custom-bpb")

DEFAULT_DENSE_RESERVED_BYTES = 8840000000  # ~8.84 GB compressed dense, T4_FEASIBILITY


def parse_bytes(value: str) -> int:
    """Parse '16GiB', '1536MiB', '8589934592', '12.75MiB' into bytes."""
    text = value.strip().lower().replace(" ", "").replace("_", "")
    multipliers = {
        "gib": GIB, "gb": 1000 ** 3, "g": 1000 ** 3,
        "mib": MIB, "mb": 1000 ** 2, "m": 1000 ** 2,
        "kib": 1024, "kb": 1000, "k": 1000,
        "b": 1, "": 1,
    }
    for suffix in sorted(multipliers, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            number = text[: -len(suffix)]
            return int(float(number) * multipliers[suffix])
        if not suffix:
            return int(float(text) * multipliers[suffix])
    raise ValueError(f"unparseable byte quantity: {value!r}")


def format_bytes(num: int) -> str:
    if num >= GIB:
        return f"{num / GIB:.2f} GiB"
    if num >= MIB:
        return f"{num / MIB:.2f} MiB"
    if num >= 1024:
        return f"{num / 1024:.1f} KiB"
    return f"{num} B"


def resolve_expert_bytes(args) -> tuple[int, bool, str]:
    """Return (expert_bytes, is_approximate, explanation)."""
    if args.expert_bytes is not None:
        return parse_bytes(args.expert_bytes), False, "user-supplied --expert-bytes"
    if args.expert_mib is not None:
        return int(float(args.expert_mib) * MIB), False, "user-supplied --expert-mib"
    if args.bits_per_param is not None:
        params = args.expert_params or DSV4_FLASH_0731["params_per_routed_expert"]
        overhead = parse_bytes(args.scale_overhead) if args.scale_overhead else 786432
        total = int(params * float(args.bits_per_param) / 8.0) + overhead
        return total, True, (
            f"approximate/research-only: {args.bits_per_param} bits/param x "
            f"{params} params + {overhead} B scale overhead; NO quality claim"
        )
    codec = (args.codec or "mxfp4").lower()
    if codec in EXPERT_RECORD_BYTES:
        return EXPERT_RECORD_BYTES[codec], False, f"canon codec record: {codec}"
    if codec in APPROXIMATE_CODECS or codec.startswith("iq") or codec.startswith("stq"):
        raise ValueError(
            f"codec {codec!r} is hypothetical/research-only: supply --bits-per-param "
            "(and optionally --expert-params/--scale-overhead) so the "
            "approximation is explicit. No quality claim is made."
        )
    raise ValueError(f"unknown codec {codec!r}; use --expert-bytes for arbitrary sizes")


def compute_budget(
    vram_total_bytes: int,
    reserved_bytes: int,
    expert_bytes: int,
    ram_cache_bytes: int,
    total_experts: int,
    topk: int,
    moe_layers: int,
) -> dict:
    if vram_total_bytes <= 0:
        raise ValueError("vram_total must be > 0")
    if expert_bytes <= 0:
        raise ValueError("expert record size must be > 0")
    if reserved_bytes < 0 or ram_cache_bytes < 0:
        raise ValueError("reserved/ram budgets must be >= 0")
    vram_for_experts = vram_total_bytes - reserved_bytes
    vram_slots = max(0, vram_for_experts // expert_bytes)
    ram_slots = max(0, ram_cache_bytes // expert_bytes)
    cached_slots = vram_slots + ram_slots
    # Upper-bound working set: distinct topk experts at every MoE layer at once.
    # Impossible for real routes; valid worst case (T4_FEASIBILITY.md).
    worst_case_bytes = topk * moe_layers * expert_bytes
    return {
        "vram_total_bytes": vram_total_bytes,
        "reserved_bytes": reserved_bytes,
        "vram_for_experts_bytes": vram_for_experts,
        "expert_bytes": expert_bytes,
        "ram_cache_bytes": ram_cache_bytes,
        "vram_slots": int(vram_slots),
        "ram_slots": int(ram_slots),
        "cached_slots": int(cached_slots),
        "total_experts": int(total_experts),
        "cache_coverage_fraction": (cached_slots / total_experts) if total_experts else 0.0,
        "vram_headroom_bytes": vram_for_experts - vram_slots * expert_bytes,
        "ram_headroom_bytes": ram_cache_bytes - ram_slots * expert_bytes,
        "worst_case_working_set_bytes": worst_case_bytes,
        "worst_case_fits_vram": worst_case_bytes <= vram_for_experts,
        "worst_case_fits_vram_plus_ram": worst_case_bytes <= vram_for_experts + ram_cache_bytes,
        "full_model_fits_cached": cached_slots >= total_experts,
        "feasible_bounded": reserved_bytes < vram_total_bytes and expert_bytes <= vram_for_experts + ram_cache_bytes,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vram-total", default="16GiB",
                        help="total VRAM, e.g. 16GiB (default) or bytes")
    parser.add_argument("--reserved", default=None,
                        help="dense/state/workspace bytes, e.g. 8.84GiB; "
                             "default is the T4_FEASIBILITY dense assumption")
    parser.add_argument("--dense-gib", type=float, default=None,
                        help="shorthand for dense reserved GiB (overrides --reserved)")
    parser.add_argument("--codec", default="mxfp4",
                        help="mxfp4 | fp16 | bf16 | int8 | arbitrary via --expert-bytes")
    parser.add_argument("--expert-bytes", default=None, help="arbitrary expert record size")
    parser.add_argument("--expert-mib", type=float, default=None, help="arbitrary expert MiB")
    parser.add_argument("--bits-per-param", type=float, default=None,
                        help="hypothetical codec density (approximate/research-only)")
    parser.add_argument("--expert-params", type=int, default=None)
    parser.add_argument("--scale-overhead", default=None)
    parser.add_argument("--ram-cache", default="24GiB", help="host RAM expert cache budget")
    parser.add_argument("--total-experts", type=int,
                        default=DSV4_FLASH_0731["n_routed_experts"] * DSV4_FLASH_0731["moe_layers_default"],
                        help="default 256*43=11008 (assumption; override if layers differ)")
    parser.add_argument("--topk", type=int, default=DSV4_FLASH_0731["num_experts_per_tok"])
    parser.add_argument("--moe-layers", type=int, default=DSV4_FLASH_0731["moe_layers_default"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    vram_total = parse_bytes(args.vram_total)
    if args.dense_gib is not None:
        reserved = int(args.dense_gib * GIB)
        reserved_note = f"--dense-gib {args.dense_gib}"
    elif args.reserved is not None:
        reserved = parse_bytes(args.reserved)
        reserved_note = f"--reserved {args.reserved}"
    else:
        reserved = DEFAULT_DENSE_RESERVED_BYTES
        reserved_note = "T4_FEASIBILITY dense assumption ~8.84 GB (override with --reserved)"
    ram_cache = parse_bytes(args.ram_cache)

    try:
        expert_bytes, approximate, expert_note = resolve_expert_bytes(args)
    except ValueError as exc:
        print(f"memory_budget: {exc}", file=sys.stderr)
        return 2

    try:
        result = compute_budget(vram_total, reserved, expert_bytes, ram_cache,
                                args.total_experts, args.topk, args.moe_layers)
    except ValueError as exc:
        print(f"memory_budget: {exc}", file=sys.stderr)
        return 2

    report = {
        "tool": "memory_budget",
        "version": TOOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_tier": "DERIVED",
        "claim_note": "Derived memory arithmetic only. Not a measurement; not a throughput prediction.",
        "model": DSV4_FLASH_0731,
        "inputs": {
            "vram_total_bytes": vram_total,
            "reserved_bytes": reserved,
            "reserved_note": reserved_note,
            "ram_cache_bytes": ram_cache,
            "codec": args.codec,
            "expert_note": expert_note,
            "approximate_research_only": approximate,
            "total_experts": args.total_experts,
            "topk": args.topk,
            "moe_layers": args.moe_layers,
        },
        "derived": result,
        "derived_human": {
            "vram_total": format_bytes(vram_total),
            "reserved": format_bytes(reserved),
            "expert_record": format_bytes(expert_bytes),
            "ram_cache": format_bytes(ram_cache),
            "worst_case_working_set": format_bytes(result["worst_case_working_set_bytes"]),
        },
    }
    if approximate:
        report["research_only_warning"] = (
            "Hypothetical codec arithmetic (e.g. STQ/IQ2). Approximate and "
            "research-only. Makes NO quality/correctness claim."
        )

    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    if not args.json_only:
        print_human(report)
    else:
        print(text)
    return 0


def print_human(report: dict) -> None:
    d = report["derived"]
    h = report["derived_human"]
    print("dee memory budget (DERIVED arithmetic — not measured, not throughput)")
    print(f"  model: {report['model']['model_id']} rev {report['model']['revision'][:12]}")
    print(f"  VRAM total: {h['vram_total']}  reserved: {h['reserved']}  "
          f"for experts: {format_bytes(d['vram_for_experts_bytes'])}")
    print(f"  expert record: {h['expert_record']} ({d['expert_bytes']} B)  [{report['inputs']['expert_note']}]")
    print(f"  VRAM slots: {d['vram_slots']}  RAM slots: {d['ram_slots']}  "
          f"cached: {d['cached_slots']}/{d['total_experts']} "
          f"({100.0 * d['cache_coverage_fraction']:.1f}%)")
    print(f"  headroom: VRAM {format_bytes(d['vram_headroom_bytes'])}, "
          f"RAM {format_bytes(d['ram_headroom_bytes'])}")
    print(f"  worst-case working set (topk x layers, upper bound): {h['worst_case_working_set']}  "
          f"fits VRAM: {d['worst_case_fits_vram']}  fits VRAM+RAM: {d['worst_case_fits_vram_plus_ram']}")
    print(f"  full model cached: {d['full_model_fits_cached']}  "
          f"bounded feasible: {d['feasible_bounded']}")
    if report["inputs"]["approximate_research_only"]:
        print("  WARNING: approximate/research-only codec — NO quality claim.")


if __name__ == "__main__":
    raise SystemExit(main())
