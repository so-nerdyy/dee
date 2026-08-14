# dee.cpp/pydee/__init__.py
"""pydee: a pybind11 binding for dee.cpp's MoE expert engine.

Used by Hugging Face Transformers' Qwen3_5MoeForCausalLM / similar models to
replace ONLY the routed-expert execution path with the streaming dee.cpp
backend. Tokenizer, embeddings, attention (full and linear), RMSNorm, KV
cache, residual, LM head, and sampling all stay on the dense HF path; the
adapter (pydee.adapter) replaces the routed MoE forward with a call into
pydee.Engine.moe_forward_experts.

This module is loaded lazily so the import succeeds even if pydee has not
been compiled yet (the canonical import path is::
    from pydee.adapter import DeeMoERuntime
or, for direct access)
    import pydee
    e = pydee.Engine()
).
"""

try:
    from .pydee_core import (  # noqa: F401  (created by setup.py build_ext --inplace)
        Engine,
        EngineConfig,
        DeviceCacheDType,
        WeightTransferDType,
        _trace_alloc_selftest,
        _trace_alloc_stats,
    )
except ImportError:
    # pydee_core not built yet; expose stubs so static imports work.
    Engine = None
    EngineConfig = None
    DeviceCacheDType = None
    WeightTransferDType = None
    _trace_alloc_selftest = None
    _trace_alloc_stats = None


def configure(shard_path: str,
              num_experts: int,
              num_layers: int,
              hidden: int,
              inter: int,
              use_cuda: bool = False,
              transfer_dtype: str = "bf16",
              cache_dtype: str = "fp32",
              topk: int = 8,
              budget_bytes: int = 0,
              swiglu_limit: float = 0.0) -> "EngineConfig":
    """Convenience: build the EngineConfig the adapter expects."""
    import pydee
    if EngineConfig is None:
        raise RuntimeError(
            "pydee compiled binding not importable. "
            "Build first: cmake --build build && (cd pydee && python3 setup.py build_ext --inplace)"
        )
    cfg = EngineConfig()
    cfg.shard_path = shard_path
    cfg.oracle_path = ""  # caller-owned routing (HF model)
    cfg.num_experts = num_experts
    cfg.num_layers = num_layers
    cfg.hidden = hidden
    cfg.inter = inter
    cfg.use_cuda = use_cuda
    cfg.transfer_dtype = {
        "bf16": pydee.WeightTransferDType.Bf16,
        "int8": pydee.WeightTransferDType.Int8,
        "int4": pydee.WeightTransferDType.Int4,
        "fp4": pydee.WeightTransferDType.Fp4E2m1,
    }[transfer_dtype]
    cfg.cache_dtype = {
        "fp32": pydee.DeviceCacheDType.Fp32,
        "fp16": pydee.DeviceCacheDType.Fp16,
    }[cache_dtype]
    cfg.topk = topk
    cfg.budget_bytes = budget_bytes
    cfg.swiglu_limit = swiglu_limit
    return cfg


def new_engine(cfg) -> "Engine":
    if Engine is None:
        raise RuntimeError("pydee compiled binding not importable; build first.")
    engine = Engine()
    if not engine.init(cfg):
        raise RuntimeError("dee.cpp Engine::init failed (see stderr)")
    return engine
