# dee consumer-harness benchmark schema (v1)

Common machine/runtime record for future comparison among dee, KTransformers,
FreeToken, llama.cpp hot-expert approaches, and MoE-Infinity where
reproducible. No competitor is implemented or run by this harness; the schema
only fixes the fields so later runs are comparable.

Every record carries a `claim_tier` per section: `measured` (observed on the
named host), `derived` (arithmetic from measured inputs + stated assumptions),
`theoretical_ceiling` (upper bound, not a runtime expectation), or `unknown`.
A result that names a GPU model (5070 Ti / 4090 / 5090 / similar) is only
valid with a `host.json` from `tools/qualify_host.py` on that actual host.

## Schema (JSON)

```json
{
  "schema": "dee.benchmark/v1",
  "claim_tier": "measured",
  "model": {
    "identifier": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    "architecture": "DeepseekV4ForCausalLM",
    "quantization_codec": "mxfp4-packed-e2m1fn+e8m0 | fp16 | int8 | ...",
    "mode": "exact | approximate",
    "mode_note": "approximate runs MUST state codec + calibration; no quality claim"
  },
  "hardware": {
    "cpu": "model string",
    "cores_physical": 0,
    "threads_logical": 0,
    "isa": {"avx2": true, "avx512f": false, "bf16": "unknown", "amx": "unknown"},
    "ram_total_bytes": 0,
    "gpu": [{"name": "...", "count": 1, "compute_capability": "12.0",
             "vram_total_bytes": 0, "driver": "...", "pcie": "gen5 x16"}],
    "ssd": {"path": "...", "filesystem": "ext4|ntfs|...",
            "seq_read_bps": 0, "rand_read_bps": 0},
    "host_qualification": "path/to/host.json"
  },
  "workload": {
    "prompt": "exact prompt text or dataset pointer",
    "prompt_tokens": 0,
    "generated_tokens": 0,
    "context_length": 0,
    "batch_size": 1
  },
  "performance": {
    "ttft_ms": 0.0,
    "prefill_tokens_per_s": 0.0,
    "decode_tokens_per_s": 0.0,
    "wall_time_s": 0.0,
    "per_token_latency_ms": {"p50": 0.0, "p90": 0.0, "p99": 0.0},
    "tier": "measured",
    "note": "end-to-end only; raw bandwidth alone is NOT a tok/s claim"
  },
  "memory": {
    "peak_ram_bytes": 0,
    "peak_vram_bytes": 0,
    "model_disk_bytes": 166878536440,
    "vram_expert_slots": 0,
    "ram_expert_slots": 0
  },
  "io": {
    "ssd_gb_per_token": 0.0,
    "h2d_gb_per_token": 0.0,
    "d2h_gb_per_token": 0.0,
    "read_throughput_bps": 0,
    "cache_hit_rates": {"vram": 0.0, "ram": 0.0}
  },
  "correctness": {
    "output_text": "...",
    "token_ids": [0],
    "exact_match": true,
    "reference_hash": "sha256:...",
    "evidence_path": "benchmark_reports/..."
  },
  "runtimes_compared": {
    "dee": {"commit": "...", "config": "..."},
    "ktransformers": null,
    "freetoken": null,
    "llama_cpp": null,
    "moe_infinity": null
  }
}
```

## Field rules

- MODEL: exact identifier + revision always. `mode: exact` means bit-exact vs
  the reference path; anything else is `approximate` with codec/calibration
  stated and no quality claim.
- HARDWARE: from `tools/qualify_host.py --output host.json`. Copy, do not
  retype. `host_qualification` points at the artifact.
- WORKLOAD: prompt text (or content hash + pointer), token counts, context,
  batch. One record per (runtime, workload) pair; no averaging across prompts.
- PERFORMANCE: measured end-to-end only. `theoretical_ceiling` values (e.g.
  bytes/bandwidth) go in a separate `roofline` object if needed, never in
  `decode_tokens_per_s`.
- MEMORY: peak RSS/VRAM measured by the runner, plus disk footprint and the
  slot counts from `tools/memory_budget.py` (derived).
- I/O: per-token GB measured by profiler counters, not estimated from hit
  rates. `cache_hit_rates` are measured telemetry.
- CORRECTNESS: output text + token IDs + exact-match vs the pinned reference;
  `reference_hash` / `evidence_path` make it re-checkable.
- Competitor slots default to `null` (not run). A non-null entry MUST include
  its own commit/config/evidence; "trivial" runs only.

## Claim discipline (normative)

Never record as `measured`: future DSV4 tok/s from bandwidth alone, 20 TPS
feasibility, native FP4 execution from 4-bit weights, or any named-GPU result
without that host's `host.json`. Violations fail review.
