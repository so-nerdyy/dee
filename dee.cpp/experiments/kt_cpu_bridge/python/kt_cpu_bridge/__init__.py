"""kt_cpu_bridge prototype package (isolated, NOT production)."""
from .codec import compact_scales_e8, decode_e8m0, dequantize_weight, unpack_fp4
from .cost_model import MicrobenchSample, SplitPlan, plan_split, plan_table
from .reference import error_metrics, expert_forward_reference, kt_emulated_forward

__all__ = [
    "decode_e8m0", "unpack_fp4", "dequantize_weight", "compact_scales_e8",
    "expert_forward_reference", "kt_emulated_forward", "error_metrics",
    "plan_split", "plan_table", "SplitPlan", "MicrobenchSample",
]
