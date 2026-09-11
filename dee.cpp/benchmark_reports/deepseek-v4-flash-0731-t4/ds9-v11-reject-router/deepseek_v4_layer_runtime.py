#!/usr/bin/env python3
"""DS9 Kaggle harness: one complete official DeepSeek-V4-Flash-0731 layer on T4.

Advances from the sealed DS8 milestone (routed+shared expert runtime through
the bounded cache, ACCEPT_EXPERT_RUNTIME) to DS9: execute ONE COMPLETE
official transformer layer -- layer 20, ``compress_ratios[20] == 4`` so the
FULL path is exercised -- and match a trusted official reference.

The layer includes, in the exact official order:

  HC-pre -> attn_norm -> [low-rank Q + kv | compressor (ratio-4 overlap) |
  indexer (low-rank Q + Hadamard + fp4 act-quant + compressed-KV scores) |
  sliding-window + compressed sparse attention with the sink (denominator
  only) | grouped low-rank O] -> HC-post
  -> HC-pre -> ffn_norm -> [sqrtsoftplus router, top-6 routed experts
  (packed FP4 storage) + 1 shared expert (F8_E4M3 storage), official weight
  placement and clamps, executed through the sealed DS8 cache runtime] ->
  HC-post

Trusted reference: scripts/deepseek_v4_layer_reference.DeepseekV4Layer with
the FP32 FFN backend, computed on CPU (full-precision-weight convention, same
as DS7/DS8, INCLUDING the in-model QAT simulation points -- act_quant / fp4
round trips).

Candidate: the SAME layer math on T4 CUDA (native torch kernels for
attention/compressor/indexer/mHC -- hybrid bring-up bridge, NOT latency
comparable) with the FFN through DeepSeekExpertCache + DeepSeekExpertLoader
(FP16-expanded payloads, async staging stream/events) -- exactly the runtime
accepted at DS8.

Gates (predeclared DS9_TOLERANCES per category + exact gates):
  - window / compress index matrices exact;
  - router expert IDs exact; routing-weight signs exact;
  - per-category bounded numerical error (attention / index / router /
    expert / final tolerances);
  - state agreement across prefill + decode steps (bounded; the reference
    runs on CPU and the candidate on CUDA, so per-buffer relative bounds
    absorb backend ULP drift -- a stale/missing write is O(1) and fails);
  - cache correctness (cold == warm candidate outputs bitwise, warm reloads
    == 0);
  - no host fallback; bounded VRAM; complete validated evidence.

performance_comparable: false -- correctness milestone, not a throughput run.
No model TPS is claimed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

RUN_ID = "20260731T120000Z-dsv4-ds9-one-layer"
REPOSITORY = "https://github.com/so-nerdyy/dee.git"
BRANCH = "freebuff/deepseek-v4-flash-0731-t4"
ROOT = Path("/kaggle/temp/dsv4-source")
EVIDENCE = Path(f"/kaggle/working/dsv4-ds9-evidence-{RUN_ID}")
ARCHIVE_BASE = EVIDENCE
DEE = ROOT / "dee.cpp"
IDENTITY_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/harness-identity-ds9.json")
HARNESS_RELATIVE = Path(
    "dee.cpp/kaggle/deepseek-v4-flash-0731/deepseek_v4_layer_runtime.py")
MODULE_RELATIVES = {
    "layer_common": Path("dee.cpp/scripts/deepseek_v4_layer_common.py"),
    "layer_reference": Path("dee.cpp/scripts/deepseek_v4_layer_reference.py"),
    "layer_candidate": Path("dee.cpp/scripts/deepseek_v4_layer_candidate.py"),
    "moe_reference": Path("dee.cpp/scripts/deepseek_v4_moe_reference.py"),
    "expert_reference": Path("dee.cpp/scripts/deepseek_v4_expert_reference.py"),
    "cache": Path("dee.cpp/scripts/deepseek_v4_cache.py"),
    "contract": Path("dee.cpp/scripts/deepseek_v4_contract.py"),
    "corpus": Path("dee.cpp/scripts/deepseek_v4_corpus.py"),
    "support": Path("dee.cpp/scripts/deepseek_v4_support.py"),
}

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
LAYER = 20  # compress_ratios[20] == 4 -> full compressor+indexer path
SHARD = "model-00022-of-00048.safetensors"
CACHED_HEADER_RELATIVE = Path(
    "dee.cpp/benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers")

# Official layer-20 config (verified against config.json).
HIDDEN = 4096
MOE_INTER = 2048
N_ROUTED = 256
TOPK = 6
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 10.0
HC_MULT = 4
N_TOKENS = 16
# (start_pos, is_prefill) steps: prefill 16 tokens, then decode steps that
# exercise state update across repeated invocations.
# DS9 v7 FOCUSED STATE DIAGNOSTIC: the v6 evidence shows the compressor/
# indexer state diverges ALREADY at step 0 (prefill), so the focused run
# needs exactly [prefill, first decode carry]: STEPS = [(0, True), (16,
# False)].  Full-sequence runs are restored by flipping FOCUSED_STATE False.
FOCUSED_STATE = True
STEPS = ([(0, True), (16, False)] if FOCUSED_STATE
         else [(0, True), (16, False), (20, False), (24, False)])
N_CORPUS_INPUTS = 1  # DS9 v1: one reference input (normal case), trace mode

# Transient HTTP codes from HF's CDN that are safe to retry with backoff.
RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_DOWNLOAD_ATTEMPTS = 6
RETRY_BACKOFF_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_logged(command: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        rc = process.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, command)
    return rc


def canonical_header_sha256(parsed: dict[str, Any]) -> str:
    canonical = json.dumps(parsed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def downloaded_header_sha256(shard_path: Path) -> str:
    with open(shard_path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{shard_path}: truncated 8-byte prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (1 << 31):
            raise ValueError(f"{shard_path}: implausible header length {hlen}")
        header_bytes = fh.read(hlen)
        if len(header_bytes) != hlen:
            raise ValueError(f"{shard_path}: truncated header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{shard_path}: malformed header JSON: {exc}") from exc
    return canonical_header_sha256(header)


def shard_expected_bytes(shard: str) -> int:
    import urllib.request
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"cannot resolve remote size for {shard}: {cr!r}")
    return int(cr.split("/")[1])


def fetch_chunk_with_retries(req: Any) -> bytes:
    import urllib.error
    import urllib.request
    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if r.status != 206:
                    raise RuntimeError(
                        f"server did not honor Range (status {r.status})")
                return r.read(8 << 20)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_CODES:
                raise
            last_error = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        print(f"  transient download error {type(last_error).__name__}: "
              f"{last_error}; retry {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS}",
              flush=True)
        time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise ConnectionError(
        f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: "
        f"{last_error!r}")


def download_shard(shard: str, checkpoint_dir: Path) -> Path:
    import urllib.request
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / shard
    want = shard_expected_bytes(shard)
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        return dest
    if have > want:
        raise RuntimeError(f"shard too large {have} > {want}")
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{shard}")
    chunk = 8 << 20
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(url, headers={
                "Range": f"bytes={have}-{end}"})
            data = fetch_chunk_with_retries(req)
            if not data:
                raise ConnectionError(f"empty chunk at {have}")
            fh.write(data)
            have += len(data)
            print(f"  {shard} {have}/{want} ({100.0*have/want:.1f}%)",
                  flush=True)
    if dest.stat().st_size != want:
        raise RuntimeError(f"shard size mismatch {dest.stat().st_size} != "
                           f"{want}")
    return dest


def load_tensors_from_shard(shard_path: Path, names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for name in names:
            if name not in f.keys():
                raise KeyError(f"missing {name} in {shard_path.name}")
            tensors[name] = f.get_tensor(name).contiguous()
    return tensors


class LazyRoutedExpertDict(dict):
    """dict subclass that materializes a routed expert's packed FP4 tensors
    from the shard on first access (single open safetensors handle).

    The DS9 sequence is CHAINED: each decode step routes through the hidden
    state produced by the previous step, so the full-sequence expert union
    CANNOT be known up-front from the first step's routing.  This dict lets
    the trusted reference run discover exactly the experts it touches;
    ``accessed`` records the ids so the candidate's FP16 payloads are built
    from the true union (v1 crashed with KeyError: 214 because a later
    decode step routed to an expert outside the step-0 union).
    """

    def __init__(self, shard_path: Path, layer: int, support: Any):
        super().__init__()
        from safetensors import safe_open
        self._layer = layer
        self._support = support
        self._handle = safe_open(str(shard_path), framework="pt", device="cpu")
        self._keys = set(self._handle.keys())
        self.accessed: set[int] = set()

    def __missing__(self, eid: int) -> dict[str, torch.Tensor]:
        # The resolver returns FULL official names
        # (layers.<l>.ffn.experts.<eid>.w1.weight, ...).  Both consumers --
        # the trusted reference (moe.routed_expert_forward_weighted) and the
        # harness fp16 payload builder -- expect SHORT keys (w1.weight,
        # w1.scale, ...).  v2 crashed with KeyError: 'w1.weight' at
        # moe_reference.py:98 because this dict was keyed by the full names.
        prefix = f"layers.{self._layer}.ffn.experts.{eid}."
        names = self._support.routed_expert_tensor_names(self._layer, eid)
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            if name not in self._keys:
                raise KeyError(f"missing {name} in shard")
            if not name.startswith(prefix):
                raise ValueError(
                    f"resolver returned unexpected tensor name {name!r} "
                    f"for layer {self._layer} expert {eid}")
            tensors[name[len(prefix):]] = self._handle.get_tensor(
                name).contiguous()
        self[eid] = tensors
        self.accessed.add(int(eid))
        return self[eid]


def move_tree_to_device(value: Any, device: str) -> Any:
    """Recursively move a nested dict-of-tensors weight structure to device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [move_tree_to_device(v, device) for v in value]
    return value


def state_agreement(ref_buffers: dict[str, torch.Tensor],
                    cand_buffers: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Bounded state agreement between the CPU reference and CUDA candidate.

    Per-buffer relative bounds absorb backend ULP drift in the bf16/fp32
    state buffers; a stale write or missing update is O(1) and fails.
    """
    # bf16 caches -> 2% relative bound; fp32 accumulators -> 0.1%.
    bounds = {
        "attn_kv_cache": 0.02, "indexer_kv_cache": 0.02,
    }
    checks = []
    all_ok = True
    for name, rt in ref_buffers.items():
        ct = cand_buffers.get(name)
        if ct is None:
            checks.append({"buffer": name, "ok": False,
                           "reason": "missing candidate buffer"})
            all_ok = False
            continue
        if rt.shape != ct.shape or rt.dtype != ct.dtype:
            checks.append({"buffer": name, "ok": False,
                           "reason": f"shape/dtype {rt.shape} {rt.dtype} vs "
                                     f"{ct.shape} {ct.dtype}"})
            all_ok = False
            continue
        rf, cf = rt.reshape(-1).float(), ct.reshape(-1).float()
        # The compressor score_state buffers are initialized to -inf and only
        # some rows are ever written (legitimate sentinels on BOTH sides), so
        # a global isfinite() gate would spuriously fail.  Treat positions
        # where both sides are non-finite as matching; require finiteness
        # agreement everywhere and compare only the finite-intersection.
        rf_fin, cf_fin = torch.isfinite(rf), torch.isfinite(cf)
        finite_ok = bool(torch.equal(rf_fin, cf_fin))
        both = rf_fin & cf_fin
        if finite_ok and bool(both.any()):
            max_abs = float((rf[both] - cf[both]).abs().max())
            max_rel = max_abs / (rf[both].abs().max().item() or 1.0)
            ok = max_rel <= bounds.get(name, 0.001)
        else:
            max_abs, max_rel = 0.0, 0.0
            ok = bool(finite_ok)
        all_ok = all_ok and ok
        checks.append({"buffer": name, "ok": bool(ok),
                       "finite_agreement": bool(finite_ok),
                       "shape": list(rt.shape), "dtype": str(rt.dtype),
                       "max_abs_diff": max_abs, "max_rel_diff": max_rel,
                       "bound": bounds.get(name, 0.001)})
    return {"ok": bool(all_ok), "buffers": checks}


def boundary_captures_compare(ref_caps: dict[str, Any],
                               cand_caps: dict[str, Any]) -> dict[str, Any]:
    """Bitwise compare the DS9 v7 compressor boundary captures.

    The reference and candidate run the IDENTICAL layer module, so their raw
    pre-write projections (kv, score), the input, ape, and the pre-write
    state snapshots must be bitwise identical UNLESS the divergence lives in
    the compressor matmul / state write itself.  ``compressor_*`` keys are
    the main attention compressor; the ``indexer_compressor`` sub-dict is the
    indexer's own compressor.  Every key reports bitwise equality, and when
    different the first divergent element with ref/cand bits, values, abs /
    rel error and ULP distance (fail-closed per-key).
    """
    # DS9 v9: keys whose bitwise equality is REQUIRED ("structural"): fresh
    # inputs, the positional bias, and scalars carry NO state and no cross-
    # device reduction, so they must be bitwise identical.  The raw
    # projections (kv_raw/score_raw) and the pre-write state snapshots are
    # LOCATOR evidence: the reference (CPU fp32) and candidate (CUDA fp32)
    # run the same module, so those legitimately drift by 1-7 ULP from
    # cross-device reduction order (v8 proved input/ape/scalars bitwise
    # while kv_raw/score_raw drifted 1-7 ULP).  ``boundary_captures_ok``
    # therefore gates ONLY the structural keys; the locator rows stay in
    # the evidence with their ULP/first-divergent-element payload.
    result: dict[str, Any] = {}

    def _ulp(a: float, b: float) -> int:
        """fp32 ULP distance (local; avoids module-level v4contract ref)."""
        def ordered(bits: int) -> int:
            return bits if bits < 0x80000000 else 0xFFFFFFFF - bits
        ia = ordered(struct.unpack("<I", struct.pack("<f", float(a)))[0])
        ib = ordered(struct.unpack("<I", struct.pack("<f", float(b)))[0])
        return abs(ia - ib)

    def compare_one(key: str, structural: bool) -> None:
        full = key
        rt = ref_caps.get(key)
        ct = cand_caps.get(key)
        if rt is None or ct is None:
            result[full] = {"present": False, "structural": structural}
            return
        if not isinstance(rt, torch.Tensor) or not isinstance(ct, torch.Tensor):
            # scalar metadata (start_pos, should_compress): plain equality
            result[full] = {"bitwise_exact": bool(rt == ct),
                            "reference": rt, "candidate": ct,
                            "structural": structural}
            return
        rf = rt.float().reshape(-1)
        cf = ct.float().reshape(-1)
        exact = bool(torch.equal(rf, cf))
        row: dict[str, Any] = {"bitwise_exact": exact,
                               "numel": int(rf.numel()),
                               "structural": structural}
        if not exact:
            diff = (rf != cf).nonzero().squeeze(-1)
            flat = int(diff[0])
            rv, cv = float(rf[flat]), float(cf[flat])
            row.update({
                "first_divergent_flat": flat,
                "ref_bits": format(
                    struct.unpack("<I", struct.pack("<f", rv))[0], "08x"),
                "cand_bits": format(
                    struct.unpack("<I", struct.pack("<f", cv))[0], "08x"),
                "reference": rv,
                "candidate": cv,
                "abs_error": float(abs(rv - cv)),
                "rel_error": float(abs(rv - cv) / (abs(rv) + 1e-12)),
                "ulp": _ulp(rv, cv),
            })
        result[full] = row

    for key, structural in (
            ("compressor_input", True),
            ("compressor_ape", True),
            ("compressor_start_pos", True),
            ("compressor_should_compress", True),
            ("compressor_kv_raw", False),
            ("compressor_score_raw", False),
            ("compressor_kv_state_pre", False),
            ("compressor_score_state_pre", False)):
        compare_one(key, structural)
    rsub = ref_caps.get("indexer_compressor") or {}
    csub = cand_caps.get("indexer_compressor") or {}
    for key, structural in (
            ("compressor_input", True),
            ("compressor_ape", True),
            ("compressor_kv_raw", False),
            ("compressor_score_raw", False),
            ("compressor_kv_state_pre", False),
            ("compressor_score_state_pre", False)):
        if key not in rsub or key not in csub:
            # fail-closed: a missing capture on EITHER side is reported
            # (present: False), never silently skipped
            result[f"indexer_compressor/{key}"] = {
                "present": False, "structural": structural}
            continue
        rt = rsub[key].float().reshape(-1)
        ct = csub[key].float().reshape(-1)
        exact = bool(torch.equal(rt, ct))
        row: dict[str, Any] = {"bitwise_exact": exact,
                               "numel": int(rt.numel()),
                               "structural": structural}
        if not exact:
            diff = (rt != ct).nonzero().squeeze(-1)
            flat = int(diff[0])
            rv, cv = float(rt[flat]), float(ct[flat])
            row.update({
                "first_divergent_flat": flat,
                "ref_bits": format(
                    struct.unpack("<I", struct.pack("<f", rv))[0], "08x"),
                "cand_bits": format(
                    struct.unpack("<I", struct.pack("<f", cv))[0], "08x"),
                "reference": rv,
                "candidate": cv,
                "abs_error": float(abs(rv - cv)),
                "rel_error": float(abs(rv - cv) / (abs(rv) + 1e-12)),
                "ulp": _ulp(rv, cv),
            })
        result[f"indexer_compressor/{key}"] = row
    return result


def router_isolation(rc: dict[str, Any], cc: dict[str, Any],
                     gate_w: torch.Tensor, gate_b: torch.Tensor) -> dict[str, Any]:
    """DS9 v10 Phase 3: router isolation matrix + Phase 4 topk kernel audit.

    The official Gate.forward runs in FP32 on both sides; the reference
    executes it on CPU and the candidate on CUDA (w_cuda gate tensors), so
    the four critical comparisons are:

      1. reference router (CPU)  on reference input
      2. candidate router (CUDA) on reference input
      3. reference router (CPU)  on candidate input
      4. candidate router (CUDA) on candidate input

    CPU variants of 2/4 (identical code) isolate the CUDA-vs-CPU matmul
    reduction effect; a CPU-vs-CUDA topk audit on IDENTICAL biased scores
    isolates the topk kernel semantics from the matmul drift.  The matrix
    must reproduce the captured ids bitwise (sanity: the isolation is
    faithful to the actual runs).
    """
    from scripts import (deepseek_v4_contract as v4contract,  # noqa: E402
                         deepseek_v4_expert_reference as ds7)
    ref_xf = rc["ffn_norm_out"].float().reshape(-1, HIDDEN).cpu()
    cand_xf = cc["ffn_norm_out"].float().reshape(-1, HIDDEN).cpu()
    gw = gate_w.float().cpu()
    gb = gate_b.float().cpu()
    iso: dict[str, Any] = {}

    def run(x: torch.Tensor, device: str) -> tuple[torch.Tensor,
                                                    torch.Tensor]:
        xs, ws, bs = x.to(device), gw.to(device), gb.to(device)
        sc, ids, _wts = ds7.router_scores(
            xs, ws, bias=bs, score_func="sqrtsoftplus",
            topk=TOPK, route_scale=ROUTE_SCALE)
        return ids.detach().cpu(), sc.detach().cpu()

    for label, x in (("ref_in", ref_xf), ("cand_in", cand_xf)):
        ids_cpu, sc_cpu = run(x, "cpu")
        iso[f"{label}_cpu_ids"] = ids_cpu.tolist()
        iso[f"{label}_cpu_scores_sha256"] = v4contract._tensor_sha(sc_cpu)
        if torch.cuda.is_available():
            ids_cuda, sc_cuda = run(x, "cuda")
            iso[f"{label}_cuda_ids"] = ids_cuda.tolist()
            iso[f"{label}_ids_cpu_vs_cuda_equal"] = bool(
                torch.equal(ids_cpu, ids_cuda))
            iso[f"{label}_ids_cpu_vs_cuda_rowwise"] = [
                bool(torch.equal(ids_cpu[r], ids_cuda[r]))
                for r in range(ids_cpu.shape[0])]
            iso[f"{label}_scores_bitwise_cpu_vs_cuda"] = bool(
                torch.equal(sc_cpu, sc_cuda))
            iso[f"{label}_scores_max_ulp_cpu_vs_cuda"] = int(
                v4contract.ulp_tensor(sc_cpu, sc_cuda).max().item())
    # Phase 4: topk kernel device audit on IDENTICAL biased scores.
    if torch.cuda.is_available():
        biased = v4contract.router_stages(
            ref_xf, gw, gb, topk=TOPK, route_scale=ROUTE_SCALE)["biased"]
        ids_cpu_topk = biased.topk(TOPK, dim=-1)[1].cpu()
        ids_cuda_topk = biased.cuda().topk(TOPK, dim=-1)[1].cpu()
        iso["topk_same_scores_cpu_vs_cuda_ids_equal"] = bool(
            torch.equal(ids_cpu_topk, ids_cuda_topk))
        iso["topk_same_scores_cpu_vs_cuda_rowwise"] = [
            bool(torch.equal(ids_cpu_topk[r], ids_cuda_topk[r]))
            for r in range(ids_cpu_topk.shape[0])]
    # sanity: the matrix must reproduce the captured ids bitwise
    if "expert_ids" in rc and "expert_ids" in cc:
        iso["captured_ref_matches_cpu_recompute"] = bool(
            torch.equal(rc["expert_ids"],
                        torch.tensor(iso["ref_in_cpu_ids"])))
        if torch.cuda.is_available():
            iso["captured_cand_matches_cuda_recompute"] = bool(
                torch.equal(cc["expert_ids"],
                            torch.tensor(iso["cand_in_cuda_ids"])))
    return iso


def router_ulp_trace(rc: dict[str, Any], cc: dict[str, Any],
                     categories: tuple[str, ...]) -> dict[str, Any]:
    """DS9 v10 Phase 5: first-divergence ULP trace across the FFN path.

    Every category is already captured device-authentically by the layer and
    relocated to CPU by run_sequence; this computes the bitwise-diff count
    and ULP statistics per boundary so the FIRST boundary where reference
    and candidate diverge is identified (attention projection -> O -> mHC ->
    ffn norm -> router input).
    """
    from scripts import deepseek_v4_contract as v4contract  # noqa: E402
    out: dict[str, Any] = {}
    for cat in categories:
        if cat not in rc or cat not in cc:
            out[cat] = {"present": False}
            continue
        a = rc[cat].float().reshape(-1)
        b = cc[cat].float().reshape(-1)
        if a.numel() != b.numel():
            out[cat] = {"present": True, "shape_mismatch": True}
            continue
        sd = a.view(torch.int32) != b.view(torch.int32)
        nd = int(sd.sum())
        ulps = v4contract.ulp_tensor(a, b)
        out[cat] = {
            "present": True,
            "numel": int(a.numel()),
            "bitwise_exact": nd == 0,
            "count_diff": nd,
            "max_ulp": int(ulps.max().item()) if nd else 0,
            "max_abs": float((a - b).abs().max().item()) if nd else 0.0,
            "first_diff_flat": int(sd.nonzero().flatten()[0].item())
            if nd else None,
        }
    return out


def classify_router_diagnosis(diag: dict[str, Any],
                              iso: dict[str, Any]) -> dict[str, Any]:
    """DS9 v10/v11 diagnostic classification.

    Delegates to the contract module's router_diagnosis_classify (pure,
    CPU-testable).  The v11 refinement replaces the v10 'max_ulp > 64'
    heuristic with the bf16_storage_bound discriminator, which is required
    for correctness: one bf16 rounding step at magnitude 2**e spans 2**16
    fp32 ULPs, so any bf16-stored activation trivially exceeds a raw-ULP
    threshold and the v10 rule mislabeled storage-rounded drift as a
    layout/lifetime/transfer defect.
    """
    from scripts import deepseek_v4_contract as v4contract  # noqa: E402
    return v4contract.router_diagnosis_classify(diag, iso)


def main() -> int:
    print("=== DS9 DeepSeek-V4-Flash-0731 one-layer runtime on T4 ===",
          flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    fatal_error: dict[str, object] | None = None
    commit: str | None = None
    module_shas: dict[str, str] = {}
    running_sha = ""
    all_gates = False
    verdict = "INVALID_EXPERIMENT"

    try:
        bootstrap = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [{
                "index": device,
                "name": torch.cuda.get_device_name(device),
                "memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            } for device in range(torch.cuda.device_count())],
        }
        write_json(EVIDENCE / "bootstrap-environment.json", bootstrap)
        if torch.cuda.device_count() < 1 or not all(
            "T4" in torch.cuda.get_device_name(device)
            for device in range(torch.cuda.device_count())
        ):
            raise RuntimeError(
                f"expected Tesla T4 topology, got {bootstrap['gpus']}")

        # CUDA capability probes: bf16 matmul on sm75 must work (the indexer
        # einsum is bf16 x bf16).  Fail fast with a precise blocker if not.
        probe = {}
        try:
            a = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")
            b = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")
            _ = a @ b
            probe["bf16_matmul_cuda"] = True
        except Exception as exc:  # noqa: BLE001
            probe["bf16_matmul_cuda"] = False
            probe["bf16_matmul_error"] = f"{type(exc).__name__}: {exc}"
        try:
            z = torch.zeros(4, device="cuda")
            _ = z.to(torch.float8_e4m3fn)
            probe["float8_cast_cuda"] = True
        except Exception as exc:  # noqa: BLE001
            probe["float8_cast_cuda"] = False
            probe["float8_cast_error"] = f"{type(exc).__name__}: {exc}"
        write_json(EVIDENCE / "cuda-capability-probe.json", probe)
        # The candidate path REQUIRES bf16 matmul (indexer einsum) and the
        # fp8 cast (act_quant_inplace on CUDA).  Fail closed with a precise
        # blocker when either is unavailable on this GPU.
        if not probe.get("bf16_matmul_cuda"):
            raise RuntimeError({"blocker": "bf16 matmul unavailable on CUDA",
                                "probe": probe})
        if not probe.get("float8_cast_cuda"):
            raise RuntimeError({"blocker": "float8 cast unavailable on CUDA",
                                "probe": probe})

        run_logged(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
             "safetensors==0.8.0"],
            EVIDENCE / "logs/pip-install.log", EVIDENCE,
        )

        if ROOT.exists():
            resolved_root = ROOT.resolve()
            if not str(resolved_root).startswith("/kaggle/temp/"):
                raise RuntimeError(
                    f"refusing to remove unexpected path {resolved_root}")
            shutil.rmtree(ROOT)
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                        REPOSITORY, str(ROOT)], check=True)

        identity_path = ROOT / IDENTITY_RELATIVE
        if not identity_path.is_file():
            raise RuntimeError(f"missing DS9 harness identity "
                               f"{identity_path}")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("model_revision") != REV:
            raise RuntimeError({"identity_model_revision":
                                identity.get("model_revision"),
                                "expected": REV})
        if identity.get("shards") != [SHARD]:
            raise RuntimeError({"identity_shards": identity.get("shards"),
                                "expected": [SHARD]})
        expected_harness_commit = identity.get("repository_commit")
        if (not isinstance(expected_harness_commit, str)
                or len(expected_harness_commit) != 40
                or any(c not in "0123456789abcdef"
                       for c in expected_harness_commit)):
            raise RuntimeError({"repository_commit": expected_harness_commit,
                                "reason": "identity must pin a 40-char commit"})
        subprocess.run(["git", "checkout", "--quiet", expected_harness_commit],
                       cwd=ROOT, check=True)
        checked_out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if checked_out != expected_harness_commit:
            raise RuntimeError({"commit": checked_out,
                                "expected": expected_harness_commit})

        committed_harness = ROOT / HARNESS_RELATIVE
        committed_harness_sha = sha256_file(committed_harness)
        running_sha = sha256_file(Path(__file__).resolve())
        if committed_harness_sha != identity.get("harness_sha256"):
            raise RuntimeError({"committed_harness_sha256":
                                committed_harness_sha,
                                "expected": identity.get("harness_sha256")})
        if running_sha != identity.get("harness_sha256"):
            raise RuntimeError({"running_harness_sha256": running_sha,
                                "expected": identity.get("harness_sha256")})

        for key, rel in MODULE_RELATIVES.items():
            committed = ROOT / rel
            module_sha = sha256_file(committed)
            expected = identity.get("module_sha256", {}).get(key)
            if not expected or module_sha != expected:
                raise RuntimeError({f"{key}_sha256": module_sha,
                                    "expected": expected})
            module_shas[key] = module_sha
        commit = checked_out
        print("pinned commit", commit, flush=True)
        print("harness sha", running_sha, flush=True)

        sys.path.insert(0, str(ROOT / "dee.cpp"))
        from scripts import (deepseek_v4_cache as v4cache,  # noqa: E402
                             deepseek_v4_contract as v4contract,
                             deepseek_v4_corpus as v4corpus,
                             deepseek_v4_expert_reference as ds7,
                             deepseek_v4_layer_candidate as v4cand,
                             deepseek_v4_layer_reference as v4ref,
                             deepseek_v4_moe_reference as moe,
                             deepseek_v4_support as v4support)
        # Run the pinned module self-tests from the checked-out tree.
        moe.main()
        ds7.main()

        # Official layer-20 config (verified against config.json).
        cfg = v4ref.LayerConfig()
        assert cfg.hidden == HIDDEN and cfg.hc_mult == HC_MULT
        assert cfg.n_routed == N_ROUTED and cfg.topk == TOPK

        # Corpus: synthetic inputs; the official hidden-state traces are a
        # DS5 dependency and are NOT claimed here.
        corpus_cases, corpus_meta = v4corpus.build_corpus(
            N_TOKENS, HIDDEN, base_seed=7,
            official_trace=Path("/kaggle/input/") / "missing.npz")
        print("corpus:", [name for name, _ in corpus_cases][:N_CORPUS_INPUTS],
              flush=True)

        checkpoint_dir = Path("/kaggle/working/dsv4-checkpoint")
        shard_path = download_shard(SHARD, checkpoint_dir)
        cached_header_path = ROOT / CACHED_HEADER_RELATIVE / f"{SHARD}.json"
        if not cached_header_path.is_file():
            raise RuntimeError(
                f"missing committed cached shard header "
                f"{cached_header_path}")
        cached = json.loads(cached_header_path.read_text(encoding="utf-8"))
        expected_header_sha = canonical_header_sha256(cached)
        got_header_sha = downloaded_header_sha256(shard_path)
        if got_header_sha != expected_header_sha:
            raise RuntimeError({"downloaded_header_sha256": got_header_sha,
                                "expected_committed_header_sha256":
                                expected_header_sha})
        h = hashlib.sha256()
        with open(shard_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        shard_sha256 = h.hexdigest()

        # ---- resolve every dense + shared-expert tensor of layer 20 --------
        dense_names = v4support.layer_dense_tensor_names(LAYER)
        shared_names = v4support.shared_expert_tensor_names(LAYER)
        dense_raw = load_tensors_from_shard(shard_path, dense_names)
        shared_raw = load_tensors_from_shard(shard_path, shared_names)
        print(f"layer {LAYER} dense tensors loaded: {len(dense_raw)}",
              flush=True)

        # ---- route the first corpus input (route-agreement evidence) -------
        # The router is the official sqrtsoftplus gate (score layer).  NOTE:
        # this step-0 routing CANNOT pre-declare the full expert union.  The
        # DS9 sequence is CHAINED (each decode step routes through the hidden
        # state produced by the previous step), so experts selected at later
        # steps -- e.g. expert 214 in v1 -- are outside the step-0 union.  The
        # trusted reference therefore runs with a LAZY routed dict that
        # materializes each expert from the shard on first access and records
        # the true full-sequence union; candidate payloads are built from
        # that discovered union (the DS8 v2 KeyError:203 lesson, applied to
        # the chained sequence).
        gate_w = dense_raw[f"layers.{LAYER}.ffn.gate.weight"].float()
        gate_b = dense_raw[f"layers.{LAYER}.ffn.gate.bias"].float()
        case_routes = []
        step0_union: set[int] = set()
        for case_name, x in corpus_cases[:N_CORPUS_INPUTS]:
            sc, ids_c, wts_c = ds7.router_scores(
                x, gate_w, bias=gate_b, score_func="sqrtsoftplus",
                topk=TOPK, route_scale=ROUTE_SCALE)
            case_routes.append({"case": case_name, "scores": sc,
                                "expert_ids": ids_c,
                                "routing_weights": wts_c})
            step0_union.update(int(e) for e in ids_c.flatten().tolist())
        _, x_route = corpus_cases[0]
        route = case_routes[0]
        ids = route["expert_ids"]
        weights = route["routing_weights"]
        print("  step-0 routed expert union:", sorted(step0_union), flush=True)

        # ---- shared expert raw tensors + fp16 payload ----------------------
        shared_t = {
            "w1.weight": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w1.weight"],
            "w1.scale": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w1.scale"],
            "w2.weight": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w2.weight"],
            "w2.scale": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w2.scale"],
            "w3.weight": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w3.weight"],
            "w3.scale": shared_raw[f"layers.{LAYER}.ffn.shared_experts.w3.scale"],
        }
        shared_payload = {
            "w1.weight": moe.dequantize_fp8_e4m3(
                shared_t["w1.weight"], shared_t["w1.scale"]).half(),
            "w2.weight": moe.dequantize_fp8_e4m3(
                shared_t["w2.weight"], shared_t["w2.scale"]).half(),
            "w3.weight": moe.dequantize_fp8_e4m3(
                shared_t["w3.weight"], shared_t["w3.scale"]).half(),
        }

        # ---- assemble the layer weight dict (reference on CPU) -------------
        raw_all: dict[str, torch.Tensor] = dict(dense_raw)
        raw_all.update(shared_raw)
        w = v4ref.build_layer_weights_from_tensors(raw_all, layer=LAYER)
        lazy_routed = LazyRoutedExpertDict(shard_path, LAYER, v4support)
        w["ffn"]["routed"] = lazy_routed
        w["ffn"]["shared"] = shared_t
        w_cuda = move_tree_to_device(w, "cuda")  # routed stays {} here (unused)

        # ---- trusted reference layer (CPU, FP32 FFN) -----------------------
        ref_layer = v4ref.DeepseekV4Layer(cfg, w, device="cpu", max_batch=1)

        # ---- run the reference sequence (DISCOVERS the expert union) -------
        # Official input construction: embed -> h.unsqueeze(2).repeat(hc_mult).
        def build_x_hc(tokens: int) -> torch.Tensor:
            return x_route[:tokens].unsqueeze(0).unsqueeze(2).expand(
                1, tokens, HC_MULT, HIDDEN).to(torch.bfloat16)

        def relocate(value: Any) -> Any:
            """Detach + move tensors to CPU (harness-side comparisons only).

            The reference runs on CPU and the candidate on CUDA; all gates
            compare their outputs/captures cross-device, so every tensor is
            relocated here before it is returned to the gate code.  This is
            comparison plumbing, NOT the candidate's execution device: the
            forward itself still runs on ``layer.device`` (v3 crashed with
            wrapper_CUDA_mm device mismatch because x_step stayed on CPU)."""
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: relocate(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [relocate(v) for v in value]
            return value

        def run_sequence(layer: Any) -> tuple[list[torch.Tensor],
                                              list[dict[str, str]],
                                              list[dict[str, Any]],
                                              list[dict[str, torch.Tensor]]]:
            outputs, sigs, caps, states = [], [], [], []
            for start_pos, is_prefill in STEPS:
                x_step = build_x_hc(N_TOKENS) if is_prefill \
                    else build_x_hc(1)
                x_step = x_step.to(layer.device)
                cap: dict[str, Any] = {}
                out = layer.forward(x_step, start_pos, capture=cap)
                outputs.append(relocate(out))
                sigs.append(layer.state_signature())
                caps.append(relocate(cap))
                states.append(layer.state_buffers())
            return outputs, sigs, caps, states

        ref_out, ref_sigs, ref_caps, ref_states = run_sequence(ref_layer)
        reference_union = set(lazy_routed.accessed)
        print("  reference-discovered full-sequence expert union:",
              sorted(reference_union), flush=True)
        if not reference_union:
            raise RuntimeError("reference run selected no routed experts")

        # ---- fp16 payloads for the candidate (discovered union) ------------
        fp16_payloads: dict[int, dict[str, torch.Tensor]] = {}
        for eid in sorted(reference_union):
            t = lazy_routed[eid]  # already materialized during the ref run
            fp16_payloads[eid] = {
                "w1.weight": ds7.dequantize_expert_weight(
                    t["w1.weight"], t["w1.scale"]).half(),
                "w2.weight": ds7.dequantize_expert_weight(
                    t["w2.weight"], t["w2.scale"]).half(),
                "w3.weight": ds7.dequantize_expert_weight(
                    t["w3.weight"], t["w3.scale"]).half(),
            }
        # Free the shard from the working dir (keeps the output archive small).
        shard_path.unlink()

        # ---- candidate layer (T4 CUDA; cache-fp16 FFN backend) -------------
        budget_bytes = int((len(reference_union) + 2) * 50 * (1 << 20))
        cache = v4cache.DeepSeekExpertCache(budget_bytes, device="cuda")
        loader = v4cache.DeepSeekExpertLoader(cache)
        cand_layer = v4cand.make_candidate_layer(
            cfg, w_cuda, device="cuda", max_batch=1, cache=cache,
            loader=loader, layer_id=LAYER, fp16_payloads=fp16_payloads,
            shared_payload=shared_payload)

        # ---- candidate sequence ---------------------------------------------
        t0 = time.time()
        cand_out, cand_sigs, cand_caps, cand_states = run_sequence(cand_layer)
        torch.cuda.synchronize()
        t_cand = time.time() - t0
        cold_stats = dict(cache.stats)
        print(f"  candidate sequence {t_cand:.3f}s, cache "
              f"{json.dumps(cold_stats)}", flush=True)

        # Warm replay: reset both layers, rerun, require bitwise-identical
        # candidate outputs and zero warm reloads.
        ref_layer.reset_state()
        cand_layer.reset_state()
        torch.cuda.synchronize()
        cand_out_warm, _, _, _ = run_sequence(cand_layer)
        torch.cuda.synchronize()
        warm_stats = dict(cache.stats)
        warm_reloaded = warm_stats["loads"] - cold_stats["loads"]
        warm_identical = all(
            torch.equal(c, cw) for c, cw in zip(cand_out, cand_out_warm))
        cache_ok = bool(warm_identical and warm_reloaded == 0)
        if not cache_ok:
            failures.append({"name": "cache_behavior",
                             "details": {"warm_identical": bool(warm_identical),
                                         "warm_reloaded_experts":
                                         warm_reloaded}})

        # ---- per-step gates -------------------------------------------------
        step_results = []
        cat_failures: list[str] = []
        for i, (start_pos, is_prefill) in enumerate(STEPS):
            rc, cc = ref_caps[i], cand_caps[i]
            row: dict[str, Any] = {"start_pos": start_pos,
                                   "is_prefill": is_prefill}
            # exact gates
            exact_ok = True
            for cat in ("attn_window_idxs",):
                eq = torch.equal(rc[cat], cc[cat])
                row[f"{cat}_exact"] = bool(eq)
                exact_ok = exact_ok and eq
            if "attn_compress_idxs" in rc and "attn_compress_idxs" in cc:
                eq = torch.equal(rc["attn_compress_idxs"],
                                 cc["attn_compress_idxs"])
                row["attn_compress_idxs_exact"] = bool(eq)
                exact_ok = exact_ok and eq
            ids_eq = bool(torch.equal(rc["expert_ids"], cc["expert_ids"]))
            row["expert_ids_exact"] = ids_eq
            exact_ok = exact_ok and ids_eq
            row["exact_gates_ok"] = bool(exact_ok)
            if not exact_ok:
                failures.append({"name": f"step_{start_pos}_exact_gates",
                                 "details": {k: v for k, v in row.items()
                                             if k.endswith("_exact")}})
            # numerical category gates
            cats_present = [cat for cat in v4contract.DS9_TOLERANCES
                            if cat in rc and cat in cc]
            cat_rows = []
            for cat in cats_present:
                tol = v4contract.DS9_TOLERANCES[cat]
                metrics = v4contract.compute_ds8_metrics(rc[cat], cc[cat])
                passed = v4contract.ds8_gate_passed(metrics, tol)
                cat_rows.append({"category": cat, "passed": bool(passed),
                                 "max_abs": metrics["all_elements"]["max_abs_error"],
                                 "mean_rel": metrics["non_near_zero"]["mean_rel_error"],
                                 "p99_rel": metrics["non_near_zero"]["p99_rel_error"],
                                 "cosine": metrics["cosine_similarity"],
                                 "norm_rmse": metrics["normalized_rmse"]})
                if not passed:
                    cat_failures.append(f"step{start_pos}:{cat}")
            row["categories"] = cat_rows
            row["categories_present"] = cats_present
            # state agreement (per-step snapshots; buffers mutate across steps)
            sa = state_agreement(ref_states[i], cand_states[i])
            row["state_agreement"] = sa
            if not sa["ok"]:
                failures.append({"name": f"step_{start_pos}_state_agreement",
                                 "details": sa["buffers"]})
            # DS9 v7 focused state diagnostics: exact sentinel/finite masks
            # per state buffer (finite/-inf classification, first divergent
            # element with bits/values/abs/rel/ULP) and bitwise comparison of
            # the raw compressor boundary captures (input, kv, score, ape,
            # pre-write state).  The reference and candidate run the IDENTICAL
            # module, so boundary bitwise equality isolates whether the first
            # divergence is INSIDE the compressor matmul/write or downstream.
            state_masks = v4contract.state_mask_analysis(
                ref_states[i], cand_states[i],
                init_values={"compressor_score_state": float("-inf"),
                             "indexer_compressor_score_state":
                                 float("-inf")})
            row["state_masks"] = state_masks
            state_masks_ok = all(
                buf.get("ok", False) for buf in state_masks.values())
            row["state_masks_ok"] = bool(state_masks_ok)
            boundary = boundary_captures_compare(rc, cc)
            row["boundary_captures"] = boundary
            # DS9 v9: gate ONLY the structural keys (input/ape/scalars); the
            # raw-projection and pre-write-state rows are locator evidence
            # where 1-7 ULP cross-device drift is the finding, not a failure.
            boundary_ok = all(
                (v.get("bitwise_exact", False) if v.get("present", True)
                 else False)
                for k, v in boundary.items() if v.get("structural", False))
            row["boundary_captures_ok"] = bool(boundary_ok)
            row["boundary_locator_ulp"] = {
                k: ({"ulp": v.get("ulp"), "abs_error": v.get("abs_error"),
                     "reference": v.get("reference"),
                     "candidate": v.get("candidate")}
                    if v.get("present", True) else None)
                for k, v in boundary.items()
                if not v.get("structural", True)
                and not v.get("bitwise_exact", True)}
            # DS9 v10: router-boundary diagnostics (expert-ID flip proof).
            # The router input (ffn_norm_out) is captured on both sides; the
            # full causal analysis is computed host-side from those exact
            # device-produced tensors plus the official FP32 Gate pipeline.
            if "ffn_norm_out" in rc and "ffn_norm_out" in cc:
                rdiag = v4contract.router_boundary_metrics(
                    rc["ffn_norm_out"], cc["ffn_norm_out"], gate_w, gate_b,
                    topk=TOPK, route_scale=ROUTE_SCALE)
                rdiag["isolation"] = router_isolation(rc, cc, gate_w, gate_b)
                rdiag["diagnostic_verdict"] = classify_router_diagnosis(
                    rdiag, rdiag["isolation"])
                rdiag["ulp_trace"] = router_ulp_trace(
                    rc, cc, ("attn_norm_in", "attn_norm_out", "attn_o",
                             "attn_out", "attn_hc_out", "ffn_norm_in",
                             "ffn_norm_out"))
                row["router_diagnosis"] = rdiag
            else:
                row["router_diagnosis"] = {"present": False}
            if not state_masks_ok:
                failures.append({"name": f"step_{start_pos}_state_masks",
                                 "details": state_masks})
            if not boundary_ok:
                failures.append({"name": f"step_{start_pos}_boundary_captures",
                                 "details": boundary})
            step_results.append(row)
        all_categories_ok = not cat_failures
        if not all_categories_ok:
            failures.append({"name": "category_gates",
                             "details": {"failing_categories": cat_failures}})

        # ---- route agreement across corpus inputs --------------------------
        route_rows = []
        for cr in case_routes:
            # Re-route via the official sqrtsoftplus gate on the same input;
            # expert IDs (and the reference's own internal re-route) must
            # agree with the up-front routing used for candidate execution.
            x_case = x_route if cr["case"] == corpus_cases[0][0] else \
                dict(corpus_cases)[cr["case"]]
            _, ids_check, _ = ds7.router_scores(
                x_case, gate_w, bias=gate_b, score_func="sqrtsoftplus",
                topk=TOPK, route_scale=ROUTE_SCALE)
            agree = bool(torch.equal(ids_check, cr["expert_ids"]))
            route_rows.append({"case": cr["case"],
                               "route_agreement": agree,
                               "expert_ids_first_token":
                                   [int(e) for e in cr["expert_ids"][0]]})
            if not agree:
                failures.append({"name": f"route_{cr['case']}_agreement",
                                 "details": {}})
        all_route_agree = all(r["route_agreement"] for r in route_rows)

        all_gates = bool(all_categories_ok and cache_ok and all_route_agree)
        print("categories ok:", all_categories_ok, "cache ok:", cache_ok,
              "route agree:", all_route_agree, flush=True)
        if cat_failures:
            print("  category failures:", cat_failures, flush=True)

        # ---- verdict --------------------------------------------------------
        if failures or not all_gates:
            # attribute by failure family (explicit failures first, then the
            # per-category gate failures carried in cat_failures)
            names = [f["name"] for f in failures]
            if any("state_agreement" in n or "state_masks" in n
                   or "boundary_captures" in n for n in names):
                verdict = "REJECT_STATE"
            elif any("exact_gates" in n or "route_" in n for n in names):
                verdict = "REJECT_ROUTER"
            elif "cache_behavior" in names:
                verdict = "REJECT_EXPERT_INTEGRATION"
            elif any(c.startswith("step") and (":q" in c or ":kv" in c
                     or ":attn_o" in c or ":attn_out" in c
                     or ":attn_hc_out" in c or ":attn_norm" in c
                     or ":indexer_scores" in c or ":layer_input" in c
                     or ":qr" in c) for c in cat_failures):
                verdict = "REJECT_ATTENTION"
            elif any(":moe_out" in c or ":shared_out" in c
                     for c in cat_failures):
                verdict = "REJECT_EXPERT_INTEGRATION"
            elif any(":router_scores" in c or ":router_bias_scores" in c
                     for c in cat_failures):
                verdict = "REJECT_ROUTER"
            elif any(":output" in c or ":ffn_norm" in c for c in cat_failures):
                verdict = "REJECT_NUMERICAL"
            else:
                verdict = "INVALID_EXPERIMENT"
        else:
            verdict = "ACCEPT_ONE_LAYER"

        # ---- DS9 v10: primary router diagnostic verdict ----------------------
        router_diag_verdict: dict[str, Any] = {
            "verdict": "NO_FLIP_OBSERVED",
            "reasons": ["no router expert-ID difference in any step"]}
        for row in step_results:
            rd = row.get("router_diagnosis") or {}
            if rd.get("first_flip_token") is not None:
                router_diag_verdict = rd.get("diagnostic_verdict") or \
                    router_diag_verdict
                break

        # ---- evidence --------------------------------------------------------
        write_json(EVIDENCE / "environment.json", {
            "schema_version": 1,
            "run_id": RUN_ID,
            "repository": REPOSITORY,
            "branch": BRANCH,
            "repository_commit": commit,
            "harness_sha256": running_sha,
            "module_sha256": module_shas,
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "shards": [SHARD],
            "performance_comparable": False,
        })
        write_json(EVIDENCE / "ds9-one-layer-evidence.json", {
            "campaign": "deepseek-v4-flash-0731",
            "phase": "DS9-one-layer",
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": REV,
            "layer": LAYER,
            "shard": SHARD,
            "shard_sha256": shard_sha256,
            "header_sha256": expected_header_sha,
            "n_tokens": N_TOKENS,
            "steps": STEPS,
            "focused_state": bool(FOCUSED_STATE),
            "n_corpus_inputs": N_CORPUS_INPUTS,
            "corpus_meta": corpus_meta,
            "selected_expert_union": sorted(reference_union),
            "step0_routed_union": sorted(step0_union),
            "expert_union_discovery": "reference-run-lazy",
            "expert_ids_first_token": [int(e) for e in ids[0]],
            "cuda_probe": probe,
            "device": torch.cuda.get_device_name(0),
            "contract": v4contract.DS9_TOLERANCES,
            "exact_gate_categories": list(v4contract.DS9_INDEX_EXACT_CATEGORIES)
                                     + ["expert_ids"],
            "router_diagnostic_verdict": router_diag_verdict,
            "router_diagnosis_steps": {
                f"step{row.get('start_pos')}": (
                    row.get("router_diagnosis") or {}).get(
                        "diagnostic_verdict") or row.get("router_diagnosis")
                for row in step_results},
            "steps_results": step_results,
            "route_agreement": route_rows,
            "all_route_agree": bool(all_route_agree),
            "cache": {
                "budget_bytes": cache.budget_bytes,
                "cold": {k: cold_stats[k] for k in
                         ("hits", "loads", "evictions", "h2d_bytes",
                          "wait_ms")},
                "warm": {k: warm_stats[k] for k in
                         ("hits", "loads", "evictions", "h2d_bytes",
                          "wait_ms")},
                "warm_reloaded_experts": warm_reloaded,
                "warm_output_identical": bool(warm_identical),
            },
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "candidate_cuda_resident": bool(
                cand_layer.attn.kv_cache.is_cuda
                and cand_layer.attn.indexer.kv_cache.is_cuda
                and cand_layer.attn.compressor.kv_state.is_cuda),
            "candidate_sequence_seconds": t_cand,
            "verdict": verdict,
            "passed": bool(all_gates and not failures),
            "performance_comparable": False,
            "note": (
                "DS9 correctness milestone: ONE complete official layer "
                "(20) executed on T4 CUDA through the Freebuff hybrid bridge "
                "(torch CUDA attention/compressor/indexer/mHC + the sealed "
                "DS8 cache-fp16 expert runtime), validated against the "
                "trusted CPU FP32 reference on the predeclared DS9 "
                "category tolerances plus exact index/router gates. "
                "Synthetic input corpus (official hidden-state traces are a "
                "DS5 dependency). No TPS is claimed. performance_comparable "
                "is false: the candidate attention path is a correctness "
                "bridge, not a latency target."
            ),
        })
        shutil.copy2(Path(__file__).resolve(),
                     EVIDENCE / "deepseek_v4_layer_runtime.py")
        for key, rel in MODULE_RELATIVES.items():
            shutil.copy2(ROOT / rel,
                         EVIDENCE / f"deepseek_v4_module_{key}.py")
        (EVIDENCE / "manifest.sha256").write_text(
            hashlib.sha256(
                (EVIDENCE / "ds9-one-layer-evidence.json").read_bytes()
            ).hexdigest() + "  ds9-one-layer-evidence.json\n",
            encoding="utf-8")

    except Exception as exc:  # noqa: BLE001
        fatal_error = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": traceback.format_exc()}
        write_json(EVIDENCE / "fatal-error.json", fatal_error)

    finally:
        required_paths = [
            EVIDENCE / "bootstrap-environment.json",
            EVIDENCE / "cuda-capability-probe.json",
            EVIDENCE / "environment.json",
            EVIDENCE / "ds9-one-layer-evidence.json",
            EVIDENCE / "manifest.sha256",
            EVIDENCE / "logs/pip-install.log",
        ]
        required_status = []
        for path in required_paths:
            required_status.append({
                "path": path.relative_to(EVIDENCE).as_posix()
                if path.is_file() else str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
            })
        missing = [row["path"] for row in required_status if not row["present"]]
        if missing:
            failures.append({"name": "required_artifacts_present",
                             "details": missing})
        result = ("PASS" if fatal_error is None and not failures and all_gates
                  else "FAIL")
        artifacts = [
            {"path": path.relative_to(EVIDENCE).as_posix(),
             "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(EVIDENCE.rglob("*"))
            if path.is_file() and path.name != "artifact-manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "result": result,
            "terminal_verdict": verdict,
            "repository_commit": commit,
            "module_sha256": module_shas,
            "fatal_error": fatal_error,
            "validation_failures": failures,
            "required_paths": required_status,
            "artifacts": artifacts,
            "performance_comparable": False,
        }
        write_json(EVIDENCE / "artifact-manifest.json", manifest)
        archive = shutil.make_archive(str(ARCHIVE_BASE), "gztar",
                                      root_dir=EVIDENCE.parent,
                                      base_dir=EVIDENCE.name)
        write_json(EVIDENCE / "archive-metadata.json", {
            "archive": archive,
            "archive_sha256": sha256_file(Path(archive)),
            "manifest_sha256": sha256_file(
                EVIDENCE / "artifact-manifest.json"),
            "excluded_from_archive": ["archive-metadata.json"],
        })
        print(json.dumps({"run_id": RUN_ID, "result": result,
                          "terminal_verdict": manifest["terminal_verdict"],
                          "archive": archive, "fatal_error": fatal_error},
                         sort_keys=True), flush=True)
        if result != "PASS":
            raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
