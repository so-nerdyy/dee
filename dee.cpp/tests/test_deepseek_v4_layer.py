"""DS9 staged ladder tests for the official DeepSeek-V4 layer port.

Covers the DS9.9 validation ladder locally (no checkpoint needed):

1. RMSNorm vs manual fp32 computation.
2. YaRN freqs_cis vs a naive recomputation.
3. FWHT vs naive Sylvester-Hadamard matrix multiply (incl. orthonormality
   with the official scale).
4. e8m0 ceil-to-power-of-two scales vs reference bit formulas.
5. e2m1 grid rounding boundaries (RNE).
6. FP8 in-place act-quant round trip.
7. FP4 in-place act-quant round trip.
8. sparse_attn vs a manual index-gather softmax with sink-in-denominator.
9. window / compress index matrices vs brute force.
10. hc_split_sinkhorn vs a manual loop port.
11. Full layer: reference (fp32 experts) vs candidate (fp16 cache backend)
    on the same input -- route agreement, exact window/compress indices,
    bounded category error, identical state signatures across prefill +
    decode, and DS9 final-layer gate.
12. Layer tensor resolution (support) + weight assembly from raw tensors.
"""

from __future__ import annotations

import torch

from scripts.deepseek_v4_layer_common import (
    act_quant_inplace,
    apply_rotary_emb,
    fast_log2_ceil,
    fast_pow2,
    fast_round_scale,
    fp4_act_quant_inplace,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    hadamard_transform,
    hc_split_sinkhorn,
    precompute_freqs_cis,
    rms_norm,
    round_e2m1_grid,
    sparse_attn,
)
from scripts.deepseek_v4_layer_reference import (
    DeepseekV4Layer,
    LayerConfig,
    build_layer_weights_from_tensors,
    make_synthetic_layer_weights,
)
from scripts.deepseek_v4_cache import DeepSeekExpertCache, DeepSeekExpertLoader
from scripts.deepseek_v4_layer_candidate import (
    build_fp16_payloads,
    build_shared_fp16_payload,
    make_candidate_layer,
)
from scripts.deepseek_v4_contract import (
    DS9_TOLERANCE_EXPERT,
    DS9_TOLERANCE_FINAL,
    compute_ds8_metrics,
)
from scripts.deepseek_v4_support import (
    layer_dense_tensor_names,
    resolve_layer_dense_tensors,
    scale_for_weight,
    shared_expert_tensor_names,
)

# Small synthetic config whose head dims keep every official quantization
# block exact (nope = 128 - 64 = 64 -> one act-quant block of 64; fp4 blocks
# of 32 divide 128).
CFG = LayerConfig(
    hidden=64,
    n_heads=4,
    head_dim=128,
    rope_head_dim=64,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=2,
    window_size=8,
    compress_ratio=4,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=8,
    n_routed=16,
    topk=2,
    route_scale=1.5,
    swiglu_limit=10.0,
    norm_eps=1e-6,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_seq_len=32,
)


# ---------------------------------------------------------------------------
# 1-2. norm + YaRN freqs
# ---------------------------------------------------------------------------


def test_rms_norm_matches_manual() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randn(16)
    eps = 1e-6
    out = rms_norm(x, weight, eps)
    manual = (x.float() / torch.sqrt(x.float().square().mean(-1, keepdim=True) + eps)) * weight
    assert out.dtype == torch.bfloat16
    # bf16 output rounding dominates: ~2^-8 relative
    assert torch.allclose(out.float(), manual, atol=5e-3, rtol=1e-2)


def test_freqs_cis_matches_naive() -> None:
    dim, seqlen = 8, 16
    base, factor = 10000.0, 16.0
    freqs = precompute_freqs_cis(dim, seqlen, 0, base, factor, 32, 1)
    angles = torch.outer(torch.arange(seqlen).float(),
                         1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
    naive = torch.view_as_complex(torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1))
    assert torch.allclose(freqs, naive, atol=1e-6)


def test_rotary_roundtrip() -> None:
    torch.manual_seed(0)
    dim, seqlen = 8, 4
    freqs = precompute_freqs_cis(dim, seqlen, 0, 10000.0, 16.0, 32, 1)
    x = torch.randn(2, seqlen, 4, dim, dtype=torch.float32)
    y = x.clone()
    apply_rotary_emb(y, freqs)
    apply_rotary_emb(y, freqs, inverse=True)
    assert torch.allclose(y, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. FWHT
# ---------------------------------------------------------------------------


def test_hadamard_matches_naive() -> None:
    n = 4
    h = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                      [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float32)
    x = torch.randn(3, n)
    y = hadamard_transform(x, 1.0)
    assert torch.allclose(y.float(), x @ h.t(), atol=1e-6)
    # official scale makes it an orthogonal transform: H^2 = n*I, so applying
    # the same (1/\/n) scale twice recovers the input exactly
    z = hadamard_transform(x, n ** -0.5)
    back = hadamard_transform(z, n ** -0.5)
    assert torch.allclose(back.float(), x, atol=1e-5)
    # bf16 round trip keeps dtype
    assert hadamard_transform(x.to(torch.bfloat16), n ** -0.5).dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 4-5. scale/rounding helpers
# ---------------------------------------------------------------------------


def test_fast_round_scale_values() -> None:
    # scale = 2^ceil(log2(x / 448)) -- the ceil-to-next-power-of-two rule
    x = torch.tensor([1.0, 1.5, 2.0, 0.75, 0.5, 448.0, 1e-4])
    scales = fast_round_scale(x, 1.0 / 448.0)
    expected = torch.tensor([2 ** -8, 2 ** -8, 2 ** -7, 2 ** -9, 2 ** -9,
                             1.0, 2 ** -22], dtype=torch.float32)
    assert torch.equal(scales, expected)


def test_fast_log2_ceil_pow2() -> None:
    x = torch.tensor([1.0, 1.5, 2.0, 3.999, 4.0, 0.5, 0.75])
    assert fast_log2_ceil(x).tolist() == [0, 1, 1, 2, 2, -1, 0]
    assert fast_pow2(torch.tensor([0, 1, 2, -1, -22])).tolist() == [1.0, 2.0, 4.0, 0.5, 2 ** -22]


def test_round_e2m1_grid_boundaries() -> None:
    a = torch.tensor([0.0, 0.24, 0.25, 0.26, 0.74, 0.75, 1.24, 1.25, 1.75,
                      2.49, 2.5, 3.49, 3.5, 4.99, 5.0, 5.5, 6.0, 6.5])
    got = round_e2m1_grid(a)
    expected = torch.tensor([0, 0, 0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0,
                             2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0])
    assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# 6-7. quant round trips
# ---------------------------------------------------------------------------


def test_act_quant_inplace_roundtrip() -> None:
    torch.manual_seed(0)
    x = (torch.randn(2, 128) * 0.1).to(torch.bfloat16)
    y = x.clone()
    act_quant_inplace(y, 128, "ue8m0")
    assert y.dtype == torch.bfloat16
    rel = (y.float() - x.float()).abs() / (x.float().abs() + 1e-3)
    assert rel.mean() < 0.05
    # zeros stay zero
    z = torch.zeros(1, 128, dtype=torch.bfloat16)
    act_quant_inplace(z, 128, "ue8m0")
    assert torch.equal(z, torch.zeros_like(z))


def test_fp4_act_quant_inplace_grid() -> None:
    # exactly-representable grid values round trip exactly (32-wide row)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
                         0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
                        dtype=torch.bfloat16)
    y = grid.clone()
    fp4_act_quant_inplace(y, 32)
    assert torch.allclose(y.float(), grid.float(), atol=1e-3)


# ---------------------------------------------------------------------------
# 8. sparse attention
# ---------------------------------------------------------------------------


def test_sparse_attn_matches_manual() -> None:
    torch.manual_seed(0)
    b, m, h, d, k = 1, 4, 2, 8, 6
    q = torch.randn(b, m, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, 10, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.tensor([[[0, 1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5, 6],
                         [2, 3, 4, 5, 6, 7],
                         [0, 2, 4, 6, 8, 9]]]).expand(b, m, -1)
    scale = d ** -0.5
    o = sparse_attn(q, kv, sink, idx, scale)
    manual = []
    for bb in range(b):
        for mm in range(m):
            for hh in range(h):
                scores = (q[bb, mm, hh].float() @ kv[bb, idx[bb, mm]].float().t()) * scale
                mx = scores.max()
                e = torch.exp(scores - mx)
                denom = e.sum() + torch.exp(sink[hh] - mx)
                out = (e @ kv[bb, idx[bb, mm]].float()) / denom
                manual.append(out)
    manual = torch.stack(manual).view(b, m, h, d)
    # o is cast to q.dtype (bf16) like the official model writes its output
    # buffer, while the manual reference computes in fp32 -- allow the bf16
    # output-rounding (~2^-8 relative) in the comparison.
    assert torch.allclose(o.float(), manual, atol=3e-3, rtol=1e-2)


def test_sparse_attn_invalid_mask() -> None:
    torch.manual_seed(0)
    q = torch.randn(1, 2, 2, 8, dtype=torch.bfloat16)
    kv = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    sink = torch.randn(2)
    idx = torch.tensor([[[0, 1, -1, -1], [0, 1, 2, 3]]])
    o = sparse_attn(q, kv, sink, idx, 8 ** -0.5)
    assert torch.isfinite(o).all()


# ---------------------------------------------------------------------------
# 9. index matrices
# ---------------------------------------------------------------------------


def test_window_idx_prefill_causal() -> None:
    idx = get_window_topk_idxs(8, 1, 8, 0)
    assert idx.shape == (1, 8, 8)
    # official: matrix = (base - win + 1).clamp(0) + arange(win), then mask
    # positions > base with -1 -> token t attends only to 0..t at prefill
    for t in range(8):
        row = idx[0, t].tolist()
        assert row[:t + 1] == list(range(t + 1))
        assert row[t + 1:] == [-1] * (7 - t)


def test_window_idx_decode_rolling() -> None:
    idx = get_window_topk_idxs(8, 1, 1, 11)
    assert idx.shape == (1, 1, 8)
    assert torch.equal(idx[0, 0], torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))


def test_compress_idx_prefill() -> None:
    idx = get_compress_topk_idxs(4, 1, 8, 0, 8)
    assert idx.shape == (1, 8, 2)
    # token t may only attend to compressed positions < (t+1)//4
    assert int(idx[0, 0, 0]) == -1
    assert int(idx[0, 7, 0]) == 8  # 8 + 0
    assert int(idx[0, 7, 1]) == 9  # 8 + 1


# ---------------------------------------------------------------------------
# 10. hc split sinkhorn
# ---------------------------------------------------------------------------


def test_hc_split_sinkhorn_matches_manual() -> None:
    torch.manual_seed(0)
    hc, iters, eps = 2, 20, 1e-6
    mix = (2 + hc) * hc
    mixes = torch.randn(5, mix)
    scale = torch.randn(3)
    base = torch.randn(mix)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)

    pre_m = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post_m = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb_m = mixes[:, 2 * hc:].unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc:].view(hc, hc)
    comb_m = comb_m.softmax(-1) + eps
    comb_m = comb_m / (comb_m.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb_m = comb_m / (comb_m.sum(-1, keepdim=True) + eps)
        comb_m = comb_m / (comb_m.sum(-2, keepdim=True) + eps)
    assert torch.allclose(pre, pre_m, atol=1e-6)
    assert torch.allclose(post, post_m, atol=1e-6)
    assert torch.allclose(comb, comb_m, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. full layer: reference vs candidate
# ---------------------------------------------------------------------------


def _make_candidate(cfg, w, routed_raw, shared_raw):
    cache = DeepSeekExpertCache(1 << 30, device="cpu")
    loader = DeepSeekExpertLoader(cache)
    fp16 = build_fp16_payloads(routed_raw)
    shared_payload = build_shared_fp16_payload(shared_raw)
    return make_candidate_layer(cfg, w, device="cpu", max_batch=1,
                                cache=cache, loader=loader, layer_id=20,
                                fp16_payloads=fp16, shared_payload=shared_payload)


def _run_sequence(layer, x_prefill, steps):
    """Prefill (start_pos=0) + single-token decode steps."""
    outputs, sigs, caps = [], [], []
    for start_pos in steps:
        x_step = x_prefill if start_pos == 0 else x_prefill[:, :1]
        cap: dict = {}
        out = layer.forward(x_step, start_pos, capture=cap)
        assert out.shape == x_step.shape
        outputs.append(out)
        sigs.append(layer.state_signature())
        caps.append(cap)
    return outputs, sigs, caps


def test_full_layer_reference_vs_candidate() -> None:
    torch.manual_seed(0)
    cfg = CFG
    w, routed_raw, shared_raw = make_synthetic_layer_weights(cfg, seed=3, n_experts=8)
    ref_layer = DeepseekV4Layer(cfg, w, device="cpu", max_batch=1)
    cand_layer = _make_candidate(cfg, w, routed_raw, shared_raw)
    x = torch.randn(1, 8, cfg.hc_mult, cfg.hidden, dtype=torch.bfloat16)

    steps = [0, 8, 11, 12]
    ref_out, ref_sigs, ref_caps = _run_sequence(ref_layer, x, steps)
    cand_out, cand_sigs, cand_caps = _run_sequence(cand_layer, x, steps)

    for i, start_pos in enumerate(steps):
        rc, cc = ref_caps[i], cand_caps[i]
        assert torch.equal(rc["attn_window_idxs"], cc["attn_window_idxs"]), f"window idxs step {start_pos}"
        assert torch.equal(rc["attn_compress_idxs"], cc["attn_compress_idxs"]), f"compress idxs step {start_pos}"
        assert torch.equal(rc["expert_ids"], cc["expert_ids"]), f"router ids step {start_pos}"
        assert torch.allclose(rc["router_scores"], cc["router_scores"], atol=1e-4), f"router scores step {start_pos}"
        # moe_out/shared_out are the DS9 expert categories gated by the remote
        # harness -- lock their parity locally (both pre-cast fp32 [b,s,d]).
        # The synthetic scale is O(1e3-1e4), so use the scale-invariant subset
        # of the expert tolerance, consistent with the final-layer gate below.
        exp_tol = DS9_TOLERANCE_EXPERT
        for cat in ("moe_out", "shared_out"):
            m = compute_ds8_metrics(rc[cat], cc[cat])
            assert m["cosine_similarity"] >= exp_tol["cosine_similarity"], f"{cat} cosine step {start_pos}"
            assert m["normalized_rmse"] <= exp_tol["normalized_rmse"], f"{cat} norm-rmse step {start_pos}"
            assert m["non_near_zero"]["p99_rel_error"] <= exp_tol["p99_rel_error"], f"{cat} p99-rel step {start_pos}"
        metrics = compute_ds8_metrics(ref_out[i], cand_out[i])
        # The synthetic fixture's random FP4/FP8 weights produce layer outputs
        # of magnitude ~1e3-1e4, so the predeclared DS9 absolute-error gates
        # (calibrated for the REAL checkpoint's O(1-10) hidden states) are not
        # meaningful here -- gate the scale-invariant metrics instead.  The
        # absolute contract stays enforced verbatim in the remote DS9 harness.
        tol = DS9_TOLERANCE_FINAL
        nnz = metrics["non_near_zero"]
        assert metrics["cosine_similarity"] >= tol["cosine_similarity"], \
            f"cosine step {start_pos}: {metrics}"
        assert metrics["normalized_rmse"] <= tol["normalized_rmse"], \
            f"norm-rmse step {start_pos}: {metrics}"
        assert metrics["output_norm_rel_error"] <= tol["output_norm_rel_error"], \
            f"out-norm step {start_pos}: {metrics}"
        assert nnz["mean_rel_error"] <= tol["mean_rel_error"], \
            f"mean-rel step {start_pos}: {metrics}"
        assert nnz["p99_rel_error"] <= tol["p99_rel_error"], \
            f"p99-rel step {start_pos}: {metrics}"
        assert metrics["excluded"]["fraction"] <= tol["max_excluded_fraction"], \
            f"excluded-fraction step {start_pos}: {metrics}"
        for key in ref_sigs[i]:
            assert ref_sigs[i][key] == cand_sigs[i][key], f"state {key} step {start_pos}"
    assert ref_out[-1].dtype == torch.bfloat16


def test_full_layer_attention_boundaries_close() -> None:
    torch.manual_seed(1)
    cfg = CFG
    w, routed_raw, shared_raw = make_synthetic_layer_weights(cfg, seed=5, n_experts=8)
    ref_layer = DeepseekV4Layer(cfg, w, device="cpu", max_batch=1)
    cand_layer = _make_candidate(cfg, w, routed_raw, shared_raw)
    x = torch.randn(1, 8, cfg.hc_mult, cfg.hidden, dtype=torch.bfloat16)
    rc, cc = {}, {}
    ref_layer.forward(x, 0, capture=rc)
    cand_layer.forward(x, 0, capture=cc)
    for cat in ("qr", "q", "kv", "kv_compressed", "attn_o", "attn_out"):
        assert torch.allclose(rc[cat].float(), cc[cat].float(), atol=1e-2), cat
    assert torch.allclose(rc["indexer_scores"].float(), cc["indexer_scores"].float(),
                          atol=1e-2), "indexer_scores"


# ---------------------------------------------------------------------------
# 12. tensor resolution + weight assembly
# ---------------------------------------------------------------------------


def _fake_ledger_rows() -> dict[str, dict]:
    full = layer_dense_tensor_names(20) + shared_expert_tensor_names(20)
    rows = {}
    for name in full:
        scale = scale_for_weight(name) if name.endswith(".weight") else None
        rows[name] = {
            "tensor_name": name,
            "source_shard": "model-00022-of-00048.safetensors",
            "scale_tensor": scale if scale in full else None,
        }
    return rows


def test_resolve_layer_dense_tensors() -> None:
    rows = _fake_ledger_rows()
    resolved = resolve_layer_dense_tensors(rows, 20)
    assert len(resolved) == 40  # 34 dense + 6 shared
    shards = {row["source_shard"] for row in resolved.values()}
    assert shards == {"model-00022-of-00048.safetensors"}
    for name, row in resolved.items():
        if row["scale_tensor"] is not None:
            assert row["scale_tensor"] in resolved


def _flatten_w_to_raw(w, layer: int = 20) -> dict:
    """Flatten the synthetic nested weight dict back to official names."""
    p = f"layers.{layer}"
    raw = {}
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
    for sub, names in (("compressor", ("wkv", "wgate")),):
        for proj in names:
            raw[f"{p}.attn.{sub}.{proj}.weight"] = a[sub][f"{proj}.weight"]
        raw[f"{p}.attn.{sub}.norm.weight"] = a[sub]["norm.weight"].to(torch.bfloat16)
        raw[f"{p}.attn.{sub}.ape"] = a[sub]["ape"]
    raw[f"{p}.attn.indexer.wq_b.weight"] = a["indexer"]["wq_b.weight"]
    raw[f"{p}.attn.indexer.weights_proj.weight"] = a["indexer"]["weights_proj.weight"].to(torch.bfloat16)
    for proj in ("wkv", "wgate"):
        raw[f"{p}.attn.indexer.compressor.{proj}.weight"] = a["indexer"]["compressor"][f"{proj}.weight"]
    raw[f"{p}.attn.indexer.compressor.norm.weight"] = a["indexer"]["compressor"]["norm.weight"].to(torch.bfloat16)
    raw[f"{p}.attn.indexer.compressor.ape"] = a["indexer"]["compressor"]["ape"]
    raw[f"{p}.ffn.gate.weight"] = w["ffn"]["gate_w"].to(torch.bfloat16)
    raw[f"{p}.ffn.gate.bias"] = w["ffn"]["gate_b"]
    return raw


def test_build_layer_weights_from_tensors() -> None:
    torch.manual_seed(0)
    cfg = CFG
    w, _, _ = make_synthetic_layer_weights(cfg, seed=7, n_experts=8)
    raw = _flatten_w_to_raw(w, layer=20)
    # exercise the fp8 dequant path on wq_a: replace with fp8 + block scale
    wq_a = w["attn"]["wq_a.weight"]
    raw["layers.20.attn.wq_a.weight"] = wq_a.to(torch.float8_e4m3fn)
    raw["layers.20.attn.wq_a.scale"] = torch.ones(
        ((wq_a.shape[0] + 127) // 128, (wq_a.shape[1] + 127) // 128),
        dtype=torch.float8_e8m0fnu)
    built = build_layer_weights_from_tensors(raw, layer=20)
    assert built["attn_norm.weight"].dtype == torch.float32
    assert built["attn"]["wq_a.weight"].dtype == torch.float32
    assert built["ffn"]["gate_w"].dtype == torch.float32
    assert list(built["attn"]["compressor"].keys()) == ["wkv.weight", "wgate.weight",
                                                        "norm.weight", "ape"]
