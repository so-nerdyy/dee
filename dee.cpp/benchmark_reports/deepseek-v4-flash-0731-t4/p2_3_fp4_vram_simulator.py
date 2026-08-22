#!/usr/bin/env python3
"""P2.3: Packed FP4 VRAM cache capacity simulator.

Analyzes the v15 expert access trace (2,509 cache loads, 104 hits, 4.1% hit rate
with 3.5 GiB FP16 cache) and projects the hit rate improvement from storing
packed FP4 (12.75 MiB/expert) instead of expanded FP16 (48.00 MiB/expert).

The 3.76× capacity increase moves from ~74 to ~281 resident experts at 3.5 GiB.
"""

import json, math

# Expert sizes
FP16_EXPERT_MIB = 48.00   # expanded gate/up/down as FP16
FP4_EXPERT_MIB  = 12.75   # packed I8 + E8M0 scales

# From v15 evidence
GPU_CACHE_LOADS  = 2509
GPU_CACHE_HITS   = 104
GPU_CACHE_BUDGET_GIB = 3.5

FP16_RESIDENT = int(GPU_CACHE_BUDGET_GIB * 1024 / FP16_EXPERT_MIB)
FP4_RESIDENT  = int(GPU_CACHE_BUDGET_GIB * 1024 / FP4_EXPERT_MIB)

print(f"=== P2.3 Packed FP4 VRAM Cache Simulator ===")
print(f"Cache budget: {GPU_CACHE_BUDGET_GIB} GiB")
print(f"FP16 experts resident: {FP16_RESIDENT}  ({FP16_EXPERT_MIB:.1f} MiB each)")
print(f"FP4  experts resident: {FP4_RESIDENT}  ({FP4_EXPERT_MIB:.1f} MiB each)")
print(f"Capacity increase: {FP4_RESIDENT/FP16_RESIDENT:.1f}×")
print()

# ── Access pattern analysis (from v15 summary) ───────────────────────
# Without a full trace, we model three scenarios:
#   1. Uniform random: hit rate ∝ resident count / total expert population
#   2. Power-law (Pareto): few hot experts dominate
#   3. Actual v15: 104/2509 = 4.1%

TOTAL_ROUTED_EXPERTS = 43 * 256  # 11,008 experts across 43 layers

print("--- Hit Rate Projections ---")

# Scenario 1: Uniform random
fp16_hit_uniform = FP16_RESIDENT / TOTAL_ROUTED_EXPERTS
fp4_hit_uniform  = FP4_RESIDENT  / TOTAL_ROUTED_EXPERTS
print(f"Uniform random: FP16 {fp16_hit_uniform*100:.1f}% → FP4 {fp4_hit_uniform*100:.1f}%")

# Scenario 2: Measured v15
fp16_hit_measured = GPU_CACHE_HITS / GPU_CACHE_LOADS
# Scale proportionally (optimistic: assumes perfect admission, uniform reuse)
fp4_hit_scaled = min(fp16_hit_measured * (FP4_RESIDENT / FP16_RESIDENT), 1.0)
print(f"Measured v15:   FP16 {fp16_hit_measured*100:.1f}% → FP4 {fp4_hit_scaled*100:.1f}% (linear scale)")

# Scenario 3: Conservative (diminishing returns, sqrt scaling)
fp4_hit_sqrt = min(fp16_hit_measured * math.sqrt(FP4_RESIDENT / FP16_RESIDENT), 1.0)
print(f"Conservative:   FP16 {fp16_hit_measured*100:.1f}% → FP4 {fp4_hit_sqrt*100:.1f}% (sqrt scale)")

print()

# ── Bytes-per-token projection ──────────────────────────────────────
EXPERTS_PER_TOKEN = 43 * 6  # 258 expert requests per token
COLD_BYTES_FP4 = FP4_EXPERT_MIB * (1 << 20)

print("--- H2D Bytes per Token ---")
for scenario_name, hit_rate in [("v15 FP16", fp16_hit_measured),
                                  ("FP4 linear", fp4_hit_scaled),
                                  ("FP4 sqrt",   fp4_hit_sqrt)]:
    cold_loads = EXPERTS_PER_TOKEN * (1 - hit_rate)
    h2d_gib = cold_loads * COLD_BYTES_FP4 / (1 << 30)
    print(f"  {scenario_name}: {hit_rate*100:.1f}% hit, {cold_loads:.0f} cold loads, {h2d_gib:.1f} GiB/token")

print()

# ── Sweep cache budgets ──────────────────────────────────────────────
print("--- Cache Budget Sweep ---")
print(f"{'Budget GiB':>10} {'FP16 experts':>13} {'FP4 experts':>13} {'FP16 hit%':>10} {'FP4 hit%':>10} {'FP4 H2D GiB':>12}")
for budget_gib in [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 5.0, 7.0, 10.0]:
    fp16_n = int(budget_gib * 1024 / FP16_EXPERT_MIB)
    fp4_n  = int(budget_gib * 1024 / FP4_EXPERT_MIB)
    fp16_h = fp16_hit_measured * (fp16_n / FP16_RESIDENT) if FP16_RESIDENT > 0 else 0
    fp4_h  = fp4_hit_scaled * (fp4_n / FP4_RESIDENT) if FP4_RESIDENT > 0 else 0
    fp4_h2d = EXPERTS_PER_TOKEN * (1 - min(fp4_h, 1.0)) * COLD_BYTES_FP4 / (1 << 30)
    print(f"{budget_gib:>10.2f} {fp16_n:>13d} {fp4_n:>13d} {fp16_h*100:>9.1f}% {fp4_h*100:>9.1f}% {fp4_h2d:>11.1f}")

print()

# ── Roofline: storage MB/s → tok/s with FP4 cache ────────────────────
print("--- Storage Roofline with FP4 Cache ---")
for storage_mbps in [13, 50, 126, 500, 1000, 3000, 5000]:
    for scenario_name, hit_rate in [("FP4 sqrt", fp4_hit_sqrt), ("FP4 linear", fp4_hit_scaled)]:
        cold_gib = EXPERTS_PER_TOKEN * (1 - hit_rate) * COLD_BYTES_FP4 / (1 << 30)
        storage_s = cold_gib * 1024 / storage_mbps
        # GPU compute: ~26ms per expert per layer, but 2 GPUs share 43 layers
        # Total GPU time per token: 43 layers × 6 experts/layer × 26ms / 2 GPUs ≈ 3.35s
        gpu_compute_s = 43 * 6 * 0.026 / 2
        tok_s = 1.0 / (storage_s + gpu_compute_s)
        print(f"  {storage_mbps:>5} MB/s  {scenario_name}: {cold_gib:.1f} GiB cold, {storage_s:.1f}s I/O + {gpu_compute_s:.1f}s GPU = {tok_s:.3f} tok/s")

print("\n=== Summary ===")
print(f"Primary bottleneck: storage I/O ({GPU_CACHE_BUDGET_GIB} GiB VRAM with ~4% hit rate)")
print(f"FP4 cache improves capacity {FP4_RESIDENT/FP16_RESIDENT:.1f}× (to ~{FP4_RESIDENT} experts)")
print(f"Projected hit rate: {fp4_hit_scaled*100:.1f}% (linear) to {fp4_hit_sqrt*100:.1f}% (conservative)")
print(f"Still storage-bound even with perfect VRAM cache: need P2.2 repack + P2.5 fused kernel + P2.8 prefetch")