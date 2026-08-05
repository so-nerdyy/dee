"""DS10 local ladder tests: full-model execution on synthetic weights.

Covers the DS10 staged ladder locally (no checkpoint, no GPU):

1. Model-level config parse (official config.json -> ModelConfig).
2. Full-model prefill + decode: every layer variant (hash ratio 0/4, score
   ratio 128/0) executes, states carry, outputs finite, deterministic rerun.
3. Hash-layer routing: expert ids come from the learned tid2eid table.
4. Greedy generation: in-vocab tokens, deterministic, cold==warm equality.
5. Ratio-128 compressor focus: prefill + decode compression paths (reference
   vs cache-fp16 candidate).
6. Ratio-0 layer (pure sliding-window) executes.
7. Memory plan is computed and bounded.
8. Tensor-source layer-dense name resolution is ratio-aware.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.deepseek_v4_cache import DeepSeekExpertCache, DeepSeekExpertLoader
from scripts.deepseek_v4_model import (
    CommittedHeaderSource,
    DeepseekV4Model,
    DictTensorSource,
    ExpertProvider,
    ModelConfig,
    coverage_audit_report,
    coverage_audit_passes,
    model_config_from_official,
    static_memory_plan,
)
from scripts.deepseek_v4_layer_reference import (
    DeepseekV4Layer,
    LayerConfig,
    make_synthetic_hash_layer_weights,
    make_synthetic_layer_weights,
)
from scripts.deepseek_v4_layer_candidate import make_candidate_layer
from scripts.deepseek_v4_support import (
    layer_dense_tensor_names,
)

REPORTS = Path(__file__).resolve().parents[1] / "benchmark_reports" \
    / "deepseek-v4-flash-0731-t4"

# Small synthetic model: layers 0-1 hash (ratios 0, 4), layers 2-3 score
# (ratios 128, 0) -- every official layer variant.
CFG = ModelConfig(
    vocab_size=32,
    dim=64,
    moe_inter_dim=32,
    n_layers=4,
    n_hash_layers=2,
    n_heads=4,
    n_routed=16,
    n_shared=1,
    topk=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    q_lora_rank=32,
    head_dim=128,
    rope_head_dim=64,
    o_groups=2,
    o_lora_rank=32,
    window_size=8,
    original_seq_len=0,
    rope_theta=10000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=8,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    norm_eps=1e-6,
    compress_rope_theta=160000.0,
    max_seq_len=64,
    compress_ratios=(0, 4, 128, 0),
)


def _flatten_layer(w: dict, layer: int, hash_layer: bool,
                   ratio: int = 4) -> dict[str, torch.Tensor]:
    """Flatten the synthetic nested layer weight dict to official names."""
    p = f"layers.{layer}"
    raw: dict[str, torch.Tensor] = {}
    for key in ("attn_norm.weight", "ffn_norm.weight"):
        raw[f"{p}.{key}"] = w[key].to(torch.bfloat16)
    for kind in ("attn", "ffn"):
        for part in ("fn", "scale", "base"):
            raw[f"{p}.hc_{kind}_{part}"] = w[f"hc_{kind}_{part}"]
    a = w["attn"]
    raw[f"{p}.attn.q_norm.weight"] = a["q_norm.weight"].to(torch.bfloat16)
    raw[f"{p}.attn.kv_norm.weight"] = a["kv_norm.weight"].to(torch.bfloat16)
    raw[f"{p}.attn.wq_a.weight"] = a["wq_a.weight"]
    raw[f"{p}.attn.wq_b.weight"] = a["wq_b.weight"]
    raw[f"{p}.attn.wkv.weight"] = a["wkv.weight"]
    raw[f"{p}.attn.wo_a.weight"] = a["wo_a.weight"]
    raw[f"{p}.attn.wo_b.weight"] = a["wo_b.weight"]
    raw[f"{p}.attn.attn_sink"] = a["attn_sink"]
    if "compressor" in a:
        raw[f"{p}.attn.compressor.wkv.weight"] = a["compressor"]["wkv.weight"]
        raw[f"{p}.attn.compressor.wgate.weight"] = a["compressor"]["wgate.weight"]
        raw[f"{p}.attn.compressor.norm.weight"] = (
            a["compressor"]["norm.weight"].to(torch.bfloat16))
        raw[f"{p}.attn.compressor.ape"] = a["compressor"]["ape"]
    if "indexer" in a:
        raw[f"{p}.attn.indexer.wq_b.weight"] = a["indexer"]["wq_b.weight"]
        raw[f"{p}.attn.indexer.weights_proj.weight"] = (
            a["indexer"]["weights_proj.weight"].to(torch.bfloat16))
        ic = a["indexer"]["compressor"]
        raw[f"{p}.attn.indexer.compressor.wkv.weight"] = ic["wkv.weight"]
        raw[f"{p}.attn.indexer.compressor.wgate.weight"] = ic["wgate.weight"]
        raw[f"{p}.attn.indexer.compressor.norm.weight"] = (
            ic["norm.weight"].to(torch.bfloat16))
        raw[f"{p}.attn.indexer.compressor.ape"] = ic["ape"]
    raw[f"{p}.ffn.gate.weight"] = w["ffn"]["gate_w"].to(torch.bfloat16)
    if hash_layer:
        raw[f"{p}.ffn.gate.tid2eid"] = w["ffn"]["tid2eid"]
    else:
        raw[f"{p}.ffn.gate.bias"] = w["ffn"]["gate_b"]
    # Scale-tensor placeholders for every F8-capable dense weight the official
    # name list expects (the synthetic weights are F32, so the scales are
    # never read -- they only satisfy tensor-resolution coverage).
    for name in layer_dense_tensor_names(
            layer, hash_layer=hash_layer, compress_ratio=ratio):
        if name.endswith(".scale"):
            weight_name = name[:-len(".scale")] + ".weight"
            wt = raw.get(weight_name)
            if wt is None or wt.ndim != 2:
                continue
            out, in_ = wt.shape
            raw[name] = torch.ones(
                ((out + 127) // 128, (in_ + 127) // 128),
                dtype=torch.float8_e8m0fnu)
    return raw


def _make_synthetic_model(cfg: ModelConfig, *, seed: int = 0,
                          n_experts: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    d, hcm = cfg.dim, cfg.hc_mult
    tensors: dict[str, torch.Tensor] = {}
    tensors["embed.weight"] = (
        torch.randn(cfg.vocab_size, d, generator=g) * 0.1).to(torch.bfloat16)
    tensors["head.weight"] = (
        torch.randn(cfg.vocab_size, d, generator=g) * 0.1).to(torch.bfloat16)
    tensors["norm.weight"] = (
        torch.randn(d, generator=g) * 0.1 + 1.0).to(torch.bfloat16)
    tensors["hc_head_fn"] = torch.randn(hcm, hcm * d, generator=g) * 0.1
    tensors["hc_head_base"] = torch.randn(hcm, generator=g) * 0.1
    tensors["hc_head_scale"] = torch.randn(1, generator=g) * 0.1
    for layer in range(cfg.n_layers):
        lcfg = cfg.layer_config(layer)
        if lcfg.hash_layer:
            w, routed_raw, shared_raw = make_synthetic_hash_layer_weights(
                lcfg, seed=seed + layer, n_experts=n_experts,
                vocab_size=cfg.vocab_size)
        else:
            w, routed_raw, shared_raw = make_synthetic_layer_weights(
                lcfg, seed=seed + layer, n_experts=n_experts)
        tensors.update(_flatten_layer(w, layer, lcfg.hash_layer,
                                      ratio=lcfg.compress_ratio))
        for eid, t in routed_raw.items():
            for k, v in t.items():
                tensors[f"layers.{layer}.ffn.experts.{eid}.{k}"] = v
        for k, v in shared_raw.items():
            tensors[f"layers.{layer}.ffn.shared_experts.{k}"] = v
    return tensors


def _build_cpu_candidate(cfg: ModelConfig, source: DictTensorSource,
                         *, dense_dtype: torch.dtype = torch.float16,
                         budget_bytes: int = 1 << 30):
    cache = DeepSeekExpertCache(budget_bytes, device="cpu")
    loader = DeepSeekExpertLoader(cache)
    provider = ExpertProvider(source)
    model = DeepseekV4Model.build_candidate(
        cfg, source, device0="cpu", device1="cpu", cache0=cache, loader0=loader,
        cache1=cache, loader1=loader, provider=provider,
        dense_dtype=dense_dtype, embed_head_dtype=torch.bfloat16, split=2)
    return model, cache, provider


def test_model_config_from_official() -> None:
    path = REPORTS / "official-source" / "inference" / "config.json"
    if not path.is_file():
        return  # asset not checked out in this environment
    cfg = model_config_from_official(path)
    assert cfg.n_layers == 43
    assert cfg.n_hash_layers == 3
    assert len(cfg.compress_ratios) == 43
    assert cfg.vocab_size == 129280
    assert cfg.dim == 4096
    # per-layer configs mirror the ratios and the hash flag
    assert cfg.layer_config(0).hash_layer is True
    assert cfg.layer_config(2).hash_layer is True
    assert cfg.layer_config(3).hash_layer is False
    assert cfg.layer_config(20).compress_ratio == 4


def test_layer_dense_names_ratio_aware() -> None:
    names4 = layer_dense_tensor_names(20, compress_ratio=4)
    names128 = layer_dense_tensor_names(20, compress_ratio=128)
    names0 = layer_dense_tensor_names(20, compress_ratio=0)
    assert len(names4) == 34
    assert len(names128) == 27
    assert len(names0) == 23
    assert any("indexer" in n for n in names4)
    assert not any("indexer" in n for n in names128)
    assert not any("compressor" in n for n in names0)
    # hash layers swap gate.bias for gate.tid2eid
    hnames = layer_dense_tensor_names(0, hash_layer=True, compress_ratio=0)
    assert "layers.0.ffn.gate.tid2eid" in hnames
    assert "layers.0.ffn.gate.bias" not in hnames


def test_model_full_prefill_decode() -> None:
    cfg = CFG
    source = DictTensorSource(_make_synthetic_model(cfg, seed=11))
    model, cache, provider = _build_cpu_candidate(cfg, source)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8)).long()
    captures: dict[int, dict] = {}
    logits = model.forward(input_ids, 0, captures=captures)
    assert logits.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    # every layer executed: layer-0 capture has router artifacts
    assert "expert_ids" in captures[0]
    assert "moe_out" in captures[3]
    # decode steps carry state: the layer-0 (window) kv cache slot 0 is
    # rewritten at start_pos=8 (8 %% 8 == 0) with a new key/value
    buf_pre = model.state_buffers([0])[0]["attn_kv_cache"].clone()
    logits1 = model.forward(torch.randint(0, cfg.vocab_size, (1, 1)).long(),
                            start_pos=8)
    buf_post = model.state_buffers([0])[0]["attn_kv_cache"]
    assert logits1.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits1).all()
    assert not torch.equal(buf_pre, buf_post), "decode must carry state"
    # determinism: identical inputs -> identical logits (fresh model)
    model2, _, _ = _build_cpu_candidate(cfg, source)
    l1 = model.forward(input_ids, 0)
    l2 = model2.forward(input_ids, 0)
    assert torch.equal(l1, l2)


def test_model_hash_layer_routing() -> None:
    cfg = CFG
    source = DictTensorSource(_make_synthetic_model(cfg, seed=3))
    model, _, _ = _build_cpu_candidate(cfg, source)
    input_ids = torch.tensor([[0, 5, 9, 2, 17, 23, 7, 13]]).long()
    tid2eid = source.get_tensor("layers.0.ffn.gate.tid2eid")
    captures: dict[int, dict] = {}
    model.forward(input_ids, 0, captures=captures)
    ids = captures[0]["expert_ids"]  # [8, topk]
    expected = tid2eid[input_ids[0]]  # [8, topk]
    assert torch.equal(ids, expected), "hash layer must route via tid2eid"


def test_model_greedy_generation_deterministic_cold_warm() -> None:
    cfg = CFG
    source = DictTensorSource(_make_synthetic_model(cfg, seed=5))
    model, cache, provider = _build_cpu_candidate(cfg, source)
    input_ids = torch.tensor([[3, 7, 11]]).long()
    trace: dict = {}
    toks = model.generate(input_ids, max_new_tokens=5, trace=trace)
    assert len(toks) == 5
    assert all(0 <= t < cfg.vocab_size for t in toks)
    assert all(trace[f"token_{i}"]["logits_finite"] for i in range(5))
    # deterministic rerun on a fresh model
    model2, _, _ = _build_cpu_candidate(cfg, source)
    toks2 = model2.generate(input_ids, max_new_tokens=5)
    assert toks == toks2
    # cold == warm: reset + rerun through the SAME model/cache -> same tokens
    model.reset_state()
    toks_warm = model.generate(input_ids, max_new_tokens=5)
    assert toks == toks_warm
    # Dynamic routed experts must not accumulate expanded FP16 host payloads.
    # Only the provider's bounded compact per-layer LRU survives an eviction.
    pstats = provider.stats()
    assert pstats["raw_entries"] <= cfg.n_layers * 8
    assert pstats["raw_bytes"] > 0
    assert all(not model.layer(i).ffn_fn.fp16_payloads
               for i in range(cfg.n_layers))


def test_ratio128_compressor_reference_vs_candidate() -> None:
    lcfg = LayerConfig(
        hidden=64, n_heads=4, head_dim=128, rope_head_dim=64,
        q_lora_rank=32, o_lora_rank=32, o_groups=2, window_size=8,
        compress_ratio=128, index_n_heads=2, index_head_dim=128,
        index_topk=8, n_routed=16, topk=2, route_scale=1.5,
        swiglu_limit=10.0, norm_eps=1e-6, hc_mult=2, hc_sinkhorn_iters=20,
        hc_eps=1e-6, max_seq_len=300)
    torch.manual_seed(0)
    w, routed_raw, shared_raw = make_synthetic_layer_weights(
        lcfg, seed=9, n_experts=8)
    ref = DeepseekV4Layer(lcfg, w, device="cpu", max_batch=1)
    cache = DeepSeekExpertCache(1 << 30, device="cpu")
    loader = DeepSeekExpertLoader(cache)
    from scripts.deepseek_v4_layer_candidate import (
        build_fp16_payloads, build_shared_fp16_payload)
    cand = make_candidate_layer(
        lcfg, w, device="cpu", max_batch=1, cache=cache, loader=loader,
        layer_id=3, fp16_payloads=build_fp16_payloads(routed_raw),
        shared_payload=build_shared_fp16_payload(shared_raw))
    x = torch.randn(1, 140, lcfg.hc_mult, lcfg.hidden,
                    dtype=torch.bfloat16)
    # prefill 140 (one compress event at 128) + decode 141 (no compress) and
    # 255 (should_compress at (255+1) % 128 == 0)
    rc, cc = {}, {}
    o_ref = ref.forward(x, 0, capture=rc)
    o_cand = cand.forward(x, 0, capture=cc)
    assert o_ref.shape == o_cand.shape == x.shape
    assert torch.isfinite(o_cand).all()
    assert torch.equal(rc["attn_window_idxs"], cc["attn_window_idxs"])
    assert torch.equal(rc["attn_compress_idxs"], cc["attn_compress_idxs"])
    # ratio-128 compress ran at prefill (one compressed slot written)
    assert "kv_compressed" in rc and "kv_compressed" in cc
    for start_pos in (141, 255):
        xs = x[:, :1]
        o_ref = ref.forward(xs, start_pos, capture=rc)
        o_cand = cand.forward(xs, start_pos, capture=cc)
        assert torch.isfinite(o_cand).all()
        assert torch.equal(rc["expert_ids"], cc["expert_ids"])
        ref_sig = ref.state_signature()
        cand_sig = cand.state_signature()
        for key in ref_sig:
            assert ref_sig[key] == cand_sig[key], f"state {key} at {start_pos}"


def test_model_memory_plan_bounded() -> None:
    cfg = CFG
    source = DictTensorSource(_make_synthetic_model(cfg, seed=7))
    model, _, _ = _build_cpu_candidate(cfg, source)
    plan = model.per_gpu_memory_plan({"cpu": 1 << 20})
    for dev, row in plan.items():
        assert row["dense_bytes"] > 0
        assert row["total_estimate_bytes"] > 0
        assert row["total_estimate_bytes"] < (14 << 30)


def test_coverage_audit_real_headers_ratio_aware() -> None:
    """The DS10.1 full-model coverage audit must use the REAL per-layer
    compress_ratios from the official config (layer 0 is ratio 0 -> no
    compressor tensors).  A default all-4 audit would demand
    layers.0.attn.compressor.ape and fail -- the exact v1 remote bug."""
    headers = REPORTS / "shard-headers"
    cfg_path = REPORTS / "official-source" / "inference" / "config.json"
    if not headers.is_dir() or not cfg_path.is_file():
        return  # committed assets not present in this environment
    cfg = model_config_from_official(cfg_path)
    src = CommittedHeaderSource(headers)
    audit = coverage_audit_report(
        src, n_layers=cfg.n_layers, n_hash_layers=cfg.n_hash_layers,
        compress_ratios=cfg.compress_ratios)
    assert audit["all_resolved"] is True
    assert coverage_audit_passes(audit, n_layers=cfg.n_layers) is True
    assert audit["tensor_count"] == 72317
    assert len(audit["layers"]) == 43
    assert audit["layers"][0]["hash_layer"] is True
    assert audit["layers"][2]["hash_layer"] is True
    assert audit["layers"][3]["hash_layer"] is False
    # dense_tensors includes the 6 shared-expert tensors (audit resolves
    # dense + shared together): ratio-4 layer = 34 dense + 6 shared = 40;
    # ratio-0 layer = 23 dense + 6 shared = 29.
    assert audit["layers"][20]["dense_tensors"] == 40
    assert audit["layers"][0]["dense_tensors"] == 29
    assert audit["layers"][0]["shared_tensors"] == 6
    assert audit["layers"][0]["routed_expert_tensors"] == 1536

    wrong_key_regression = dict(audit)
    wrong_key_regression.pop("all_resolved")
    wrong_key_regression["ok"] = True
    assert coverage_audit_passes(
        wrong_key_regression, n_layers=cfg.n_layers) is False


def test_static_memory_plan_from_real_headers_bounded() -> None:
    """The DS10.2 per-GPU resident-bytes plan, computed from the COMMITTED
    official shard headers (identity only -- zero checkpoint bytes).

    This is the bring-up memory envelope the harness checks on Kaggle.
    The official model must fit under the 12-13 GiB bring-up ceiling per
    T4; the plan here is the same computation the remote v1 run gates on.
    """
    headers = REPORTS / "shard-headers"
    cfg_path = REPORTS / "official-source" / "inference" / "config.json"
    if not headers.is_dir() or not cfg_path.is_file():
        return  # committed assets not present in this environment
    cfg = model_config_from_official(cfg_path)
    src = CommittedHeaderSource(headers)
    assert len(src.tensor_names()) == 72317
    plan = static_memory_plan(cfg, src, budgets0=2 << 30, budgets1=2 << 30)
    d0, d1 = plan["devices"]["cuda:0"], plan["devices"]["cuda:1"]
    assert d0["total_estimate_gib"] < 12.0
    assert d1["total_estimate_gib"] < 12.0
    assert d0["total_estimate_gib"] > 5.0
    assert d1["total_estimate_gib"] > 5.0
    assert d0["dense_bytes"] > 0 and d1["dense_bytes"] > 0
    assert plan["split"] == 22  # (43 + 1) // 2


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="requires CUDA (v2 device-cat regression)")
def test_ratio128_reference_forward_cuda_no_device_cat_crash() -> None:
    """Regression for the DS10 v2 remote failure: ratio-128 layers use
    get_compress_topk_idxs which built on CPU, so the torch.cat with the
    cuda window_idxs crashed ("Expected all tensors to be on the same
    device").  The reference layer must move compress_idxs to x.device.
    """
    dev = "cuda:0"
    lcfg = LayerConfig(
        hidden=64, n_heads=4, head_dim=128, rope_head_dim=64,
        q_lora_rank=32, o_lora_rank=32, o_groups=2, window_size=8,
        compress_ratio=128, index_n_heads=2, index_head_dim=128,
        index_topk=8, n_routed=16, topk=2, route_scale=1.5,
        swiglu_limit=10.0, norm_eps=1e-6, hc_mult=2, hc_sinkhorn_iters=20,
        hc_eps=1e-6, max_seq_len=300)
    torch.manual_seed(0)
    w, routed_raw, shared_raw = make_synthetic_layer_weights(
        lcfg, seed=9, n_experts=8)
    w = {k: (v.to(dev) if isinstance(v, torch.Tensor) else
             {kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict)
             else v) for k, v in w.items()}
    from scripts.deepseek_v4_model import _cast_floats, _move_to_device
    w = _move_to_device(_cast_floats(w, torch.float16), dev)
    ref = DeepseekV4Layer(lcfg, w, device=dev, max_batch=1)
    x = torch.randn(1, 140, lcfg.hc_mult, lcfg.hidden,
                    dtype=torch.bfloat16, device=dev)
    o = ref.forward(x, 0)  # prefill 140 exercises the ratio-128 compress cat
    assert o.shape == x.shape
    assert bool(torch.isfinite(o).all())


def test_model_state_sentinels_preserved() -> None:
    """Compressor score_state keeps -inf sentinels after a short prefill:
    after 8 tokens at ratio 4 only a few slots are written; the untouched
    slots must remain exactly -inf (structural sentinel discipline)."""
    cfg = CFG
    source = DictTensorSource(_make_synthetic_model(cfg, seed=13))
    model, _, _ = _build_cpu_candidate(cfg, source)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8)).long()
    model.forward(input_ids, 0)
    bufs = model.state_buffers([1])[1]  # layer 1 is ratio 4
    ss = bufs["compressor_score_state"]
    assert bool((ss == float("-inf")).any()), "unwritten slots must stay -inf"
    assert bool((ss != float("-inf")).any()), "written slots must be finite"
    iss = bufs["indexer_compressor_score_state"]
    assert bool((iss == float("-inf")).any())
    assert bool((iss != float("-inf")).any())
