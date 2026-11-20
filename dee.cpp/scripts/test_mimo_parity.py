"""Layer-level parity: dee adapter vs upstream MiMoV2 reference.

Loads the pinned ``modeling_mimo_v2.py`` (vendored under
``modal/phase6/ref/``) + real checkpoint tensors over HF byte ranges, then
compares on REAL layer-1 (SWA + MoE) weights:
  * router: expert ids + weights (noaux_tc sigmoid+bias) — ids EXACT,
    weights within fp tolerance
  * MoE combine: dee's (raw*w).sum(dim=1) vs reference per-expert
    index_add semantics on deterministic pseudo-experts
  * attention: full layer output mine vs reference — bf16 tolerance

Run:  python scripts/test_mimo_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modal" / "phase6"))

from ref import configuration_mimo_v2  # noqa: E402
from ref import modeling_mimo_v2 as ref  # noqa: E402

from scripts import mimo_v2_model as mine  # noqa: E402

REPO = mine.OFFICIAL_REPOSITORY
REV = mine.OFFICIAL_REVISION
HEADERS = ROOT / "tools" / "phase3" / "headers" / "mimo-flash"
CFG_PATH = ROOT / "modal" / "phase6" / "ref" / "config.json"

LAYER = 1   # SWA + MoE — the interesting layer type
SEQ = 5


class DictEngine:
    """Engine stub: deterministic pseudo-expert outputs."""

    def __init__(self, expert_fn):
        self.expert_fn = expert_fn

    def moe_forward_batch(self, layer, h_in, ids):
        h = torch.from_numpy(np.asarray(h_in))
        out = torch.zeros(h.shape[0], ids.shape[1], h.shape[1],
                          dtype=torch.float32)
        for t in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                out[t, k] = self.expert_fn(int(ids[t, k]), h[t].float())
        return out.numpy()


def main() -> int:
    if not CFG_PATH.exists():
        import urllib.request
        urllib.request.urlretrieve(
            f"https://huggingface.co/{REPO}/resolve/{REV}/config.json",
            CFG_PATH)
    mcfg = mine.mimo_config_from_official(CFG_PATH)
    ref_cfg = configuration_mimo_v2.MiMoV2Config(
        **json.loads(CFG_PATH.read_text()))
    source = mine.RemoteShardSource(HEADERS, repository=REPO, revision=REV)

    torch.manual_seed(0)
    x = (torch.randn(1, SEQ, mcfg.hidden) * 0.5).to(torch.bfloat16)

    # --- router parity -----------------------------------------------------
    p = f"model.layers.{LAYER}."
    gate_w = source.get_tensor(p + "mlp.gate.weight").float()
    gate_b = source.get_tensor(
        p + "mlp.gate.e_score_correction_bias").float()

    ref_gate = ref.MiMoV2MoEGate(ref_cfg)
    ref_gate.weight = torch.nn.Parameter(gate_w)
    ref_gate.e_score_correction_bias = torch.nn.Parameter(gate_b)
    ref_gate.eval()
    with torch.no_grad():
        ref_idx, ref_w = ref_gate(x)

    xf = x.reshape(-1, mcfg.hidden).float()
    chosen = torch.sigmoid(xf @ gate_w.T) + gate_b
    my_idx = torch.topk(chosen, mcfg.topk, dim=-1, sorted=False)[1]
    my_w = torch.sigmoid(xf @ gate_w.T).gather(1, my_idx)
    my_w = my_w / (my_w.sum(dim=-1, keepdim=True) + 1e-20)

    same_ids = torch.equal(ref_idx.sort(-1)[0], my_idx.sort(-1)[0])
    wdiff = (ref_w.sort(-1)[0] - my_w.sort(-1)[0]).abs().max().item()
    print(f"router: ids_exact={same_ids} max_w_diff={wdiff:.3e}")
    assert same_ids, "router ids differ"
    assert wdiff < 1e-6

    # --- MoE combine parity ------------------------------------------------
    probe = torch.randn(mcfg.n_routed, mcfg.hidden)

    def expert_fn(eid: int, hv: torch.Tensor) -> torch.Tensor:
        return hv * probe[eid].mean().float()

    t = xf.shape[0]
    ref_final = torch.zeros(t, mcfg.hidden)
    for ti in range(t):
        for k in range(mcfg.topk):
            eid = int(ref_idx[ti, k])
            ref_final[ti] += (expert_fn(eid, xf[ti])
                              * ref_w[ti, k].float())

    eng = DictEngine(expert_fn)
    my_raw = torch.from_numpy(
        eng.moe_forward_batch(0, xf.numpy(), ref_idx.numpy()))
    my_final = (my_raw * ref_w.float()[..., None]).sum(dim=1)
    d = (ref_final - my_final).abs().max().item()
    print(f"moe combine: max_diff={d:.3e}")
    assert d < 1e-5

    # --- attention parity --------------------------------------------------
    ref_attn = ref.MiMoV2Attention(ref_cfg, is_swa=True, layer_idx=LAYER,
                                   projection_layout="fused_qkv")
    qkv_w = source.get_tensor(p + "self_attn.qkv_proj.weight")
    qkv_si = source.get_tensor(p + "self_attn.qkv_proj.weight_scale_inv")
    o_w = source.get_tensor(p + "self_attn.o_proj.weight")
    sink = source.get_tensor(p + "self_attn.attention_sink_bias")
    qkv_deq = mine.dequantize_fp8_e4m3_block(qkv_w, qkv_si).to(torch.bfloat16)
    ref_attn = ref_attn.to(torch.bfloat16)
    with torch.no_grad():
        ref_attn.qkv_proj.weight.copy_(qkv_deq)
        ref_attn.o_proj.weight.copy_(o_w.to(torch.bfloat16))
        ref_attn.attention_sink_bias.copy_(sink.to(torch.bfloat16))
    ref_attn.eval()

    pos_ids = torch.arange(SEQ).unsqueeze(0)
    rope_mod = ref.MiMoV2RotaryEmbedding(ref_cfg, is_swa=True)
    cos, sin = rope_mod(x, pos_ids)
    # transformers' mask helpers return None for a trivially-unmasked call
    # (eager then runs NON-causal) — build the sliding-window causal mask
    # explicitly: query i attends keys j iff j <= i and i - j < window.
    W = mcfg.sliding_window
    qi = torch.arange(SEQ)[:, None]
    kj = torch.arange(SEQ)[None, :]
    attn_mask = torch.zeros(1, 1, SEQ, SEQ, dtype=torch.bfloat16)
    attn_mask.masked_fill_(
        ((kj > qi) | (qi - kj >= W))[None, None], float("-inf"))
    with torch.no_grad():
        ref_full = ref_attn(x, (cos, sin), attn_mask)[0]

    w = {"qkv": qkv_deq, "o": o_w.to(torch.bfloat16),
         "sink": sink.float()}
    my_attn_layer = mine.MiMoV2Layer(
        mcfg, LAYER, w, engine=None, device="cpu", is_moe=False,
        is_swa=True, geo=mcfg.swa_attn,
        rope=mine.RotaryCache(mcfg.swa_attn.rope_dim,
                              mcfg.swa_attn.rope_theta, 1024, "cpu",
                              torch.bfloat16),
        diagnostics=False)
    my_out = my_attn_layer._attention(x, 0)

    diff = (ref_full.float() - my_out.float()).abs()
    print(f"attention: ref={tuple(ref_full.shape)} "
          f"mine={tuple(my_out.shape)} "
          f"max_diff={diff.max().item():.4f} "
          f"mean={diff.mean().item():.5f}")
    assert diff.max().item() < 0.05, "attention drift"
    print("PARITY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
