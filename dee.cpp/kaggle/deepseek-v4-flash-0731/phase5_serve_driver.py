"""Phase-5 W2 — dee-serve v0 cohort driver (model-agnostic core).

Owns the serving semantics that sit above ``DeepseekV4Model.generate_cohort``:

- Cohort construction: tokenize K prompts and LEFT-PAD every row to the
  cohort's common length L* (the scalar-start_pos contract requires equal
  positions; per-row positions are P5b scope).  Padding is deterministic
  and recorded in the artifacts — the exactness invariant is
  ``cohort row == sequential run of the identical padded input``.
- Execution: one ``model.generate_cohort`` call per cohort, wiring
  per-row token checkpoints and the K-row route journal.
- Telemetry: per-row token SHAs, per-forward timings, engine/cache
  counter snapshots, and expert-dedup statistics (unique experts staged
  per (step, layer) vs. the K*topk request sum — the direct measure of
  cross-request sharing).
- Artifacts: per-row result JSON + checkpoint JSONL, cohort summary,
  SHA256 manifest — same evidence discipline as Phase 4.

Local use: ``python phase5_serve_driver.py --selftest`` exercises the
driver end-to-end on the synthetic CPU model.  The Kaggle runner
(``deepseek_v4_native_generate.py`` cohort mode) drives it with the real
engines, route journal, and checkpoint sinks attached.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_ids_sha(tokens: list[int]) -> str:
    """Canonical token-stream hash, same convention as the Phase-4
    integrity payload: sha256 over compact-json serialization.
    Comparable to committed token-sha-manifest.json values."""
    return _sha256_bytes(
        json.dumps([int(t) for t in tokens],
                   separators=(",", ":")).encode("utf-8"))


@dataclass
class CohortSpec:
    """One lockstep cohort: prompt indices into the driver's prompt set."""
    cohort_id: int
    prompt_indices: list[int]


@dataclass
class RowResult:
    row: int
    prompt_index: int
    prompt_text: str
    pad_tokens: int
    prompt_len: int          # tokenized length INCLUDING pad
    token_ids: list[int]
    token_ids_sha256: str
    decoded_text: str = ""


@dataclass
class CohortResult:
    cohort_id: int
    k: int
    prompt_len: int          # L* — the shared padded length
    rows: list[RowResult]
    wall_seconds: float
    decode_timings_ms: list[float]
    route_journal: Optional[dict[str, Any]] = None
    dedup: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, Any] = field(default_factory=dict)


def build_cohort_ids(tokenize: Callable[[str], list[int]],
                     prompts: list[str],
                     pad_token_id: int = 0,
                     min_len: int = 0
                     ) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Tokenize + left-pad a cohort to a common length.

    ``min_len`` raises the pad target above the group max — the session
    driver's ``pad_to: "max"`` resolves it to the workload-global token
    length so EVERY arm (including K=1 reference groups) sees the same
    padded inputs and the exactness gate compares like inputs.

    Returns (ids[K, L*], pad_meta[K]).  Left-padding is the only pad side
    compatible with causal decode, and determinism is the only property
    the exactness gate needs — the sequential reference sees the same
    padded inputs.
    """
    tokenized = [list(tokenize(t)) for t in prompts]
    lstar = max([min_len] + [len(t) for t in tokenized])
    ids, meta = [], []
    for t in tokenized:
        pad = lstar - len(t)
        ids.append([pad_token_id] * pad + t)
        meta.append({"pad_tokens": pad, "raw_prompt_len": len(t)})
    return ids, meta


def dedup_stats_from_journal(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-request expert sharing, from route-journal records.

    Each record's ``expert_ids_rank_order`` is the [K, topk] matrix for
    one (forward_step, layer).  Per record: ``sum`` counts every
    request-slot (K*topk); ``unique`` counts the distinct experts the
    engine had to stage — their ratio is the dedup factor.
    """
    total_slots = 0
    total_unique = 0
    for rec in records:
        matrix = rec.get("expert_ids_rank_order") or rec.get("expert_ids")
        if not matrix:
            continue
        flat = [int(e) for row in matrix for e in row]
        total_slots += len(flat)
        total_unique += len(set(flat))
    return {
        "request_slots": total_slots,
        "unique_experts_staged": total_unique,
        "dedup_ratio": (round(total_slots / total_unique, 4)
                        if total_unique else None),
        "records": len(records),
    }


class ServeDriver:
    """Run lockstep cohorts through a model with per-row evidence.

    ``model`` must expose ``generate_cohort(input_ids[K, L], n, ...)``
    and ``reset_state()``.  ``tokenize``/``decode`` adapt the tokenizer.
    Optional hooks plug into the host environment's evidence machinery:

    - ``make_row_checkpoint(cohort_id, row)`` -> callable(step, token)
      or None; invoked per row so the caller can fan the cohort's
      per-step token list out into per-row checkpoint files.
    - ``on_cohort_counters(cohort_id)`` -> dict snapshot of engine/cache
      counters taken after the cohort (host pack, store reads, VRAM
      counters) for shared-cache telemetry.
    """

    def __init__(self, model: Any,
                 tokenize: Callable[[str], list[int]],
                 decode: Callable[[list[int]], str],
                 out_dir: Path,
                 run_id: str,
                 pad_token_id: int = 0,
                 make_row_checkpoint: Optional[Callable[[int, int], Any]] = None,
                 on_cohort_step: Optional[Callable[[int, int, list[int]], None]] = None,
                 post_layer_hook: Optional[Callable[[int], None]] = None,
                 on_cohort_counters: Optional[Callable[[int], dict]] = None):
        self.model = model
        self.tokenize = tokenize
        self.decode = decode
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.pad_token_id = pad_token_id
        self.make_row_checkpoint = make_row_checkpoint
        # on_cohort_step(cohort_id, step, toks[K]) — runs BEFORE the row
        # hooks so the caller can bind a once-per-forward journal link and
        # embed it in every row's checkpoint record.
        self.on_cohort_step = on_cohort_step
        self.post_layer_hook = post_layer_hook
        self.on_cohort_counters = on_cohort_counters
        self.results: list[CohortResult] = []

    def run_cohort(self, prompt_texts: list[str], cohort_id: int,
                   prompt_indices: list[int],
                   max_new_tokens: int,
                   eos_id: int = -1,
                   prebuilt: Optional[tuple[list[list[int]],
                                          list[dict[str, Any]]]] = None
                   ) -> CohortResult:
        import torch
        k = len(prompt_texts)
        if k < 1:
            raise ValueError("cohort requires K >= 1 prompts")
        if prebuilt is not None:
            ids_list, pad_meta = prebuilt
            if len(ids_list) != k or len({len(r) for r in ids_list}) != 1:
                raise ValueError(
                    "prebuilt ids must be [K, L*] with equal row lengths")
        else:
            ids_list, pad_meta = build_cohort_ids(
                self.tokenize, prompt_texts, self.pad_token_id)
        lstar = len(ids_list[0])
        input_ids = torch.tensor(ids_list, dtype=torch.long)

        row_hooks = None
        if self.make_row_checkpoint is not None:
            row_hooks = [self.make_row_checkpoint(cohort_id, r)
                         for r in range(k)]

        def _step_hook(step: int, toks: list[int]) -> None:
            if self.on_cohort_step is not None:
                self.on_cohort_step(cohort_id, step, [int(t) for t in toks])
            if row_hooks:
                for r, hk in enumerate(row_hooks):
                    if hk is not None:
                        hk(step, int(toks[r]))

        decode_ms: list[float] = []
        self.model.reset_state()
        t0 = time.monotonic()
        # forward() moves ids to device0 itself; CPU tensor is fine.
        streams = self.model.generate_cohort(
            input_ids, max_new_tokens, eos_id=eos_id,
            decode_timings_ms=decode_ms,
            post_step_hook=_step_hook,
            post_layer_hook=self.post_layer_hook)
        wall = time.monotonic() - t0

        if len(streams) != k:
            raise RuntimeError(
                f"cohort {cohort_id}: model returned {len(streams)} "
                f"streams for K={k}")

        rows = []
        for r in range(k):
            toks = [int(t) for t in streams[r]]
            rows.append(RowResult(
                row=r, prompt_index=prompt_indices[r],
                prompt_text=prompt_texts[r],
                pad_tokens=pad_meta[r]["pad_tokens"],
                prompt_len=lstar,
                token_ids=toks,
                token_ids_sha256=token_ids_sha(toks),
                decoded_text=self.decode(toks) if self.decode else ""))

        res = CohortResult(
            cohort_id=cohort_id, k=k, prompt_len=lstar, rows=rows,
            wall_seconds=round(wall, 3),
            decode_timings_ms=[round(t, 3) for t in decode_ms])
        if self.on_cohort_counters is not None:
            try:
                res.counters = self.on_cohort_counters(cohort_id)
            except Exception as exc:
                res.counters = {"error": repr(exc)}
        self.results.append(res)
        return res

    def write_row_artifacts(self, res: CohortResult, *,
                            suffix: str = "",
                            extra: Optional[dict[str, Any]] = None
                            ) -> list[Path]:
        """Per-row result JSON — same evidence granularity as Phase-4
        per-prompt results, plus cohort/pad metadata for the gate."""
        paths = []
        for row in res.rows:
            payload = {
                "run_id": self.run_id,
                "cohort_id": res.cohort_id,
                "row": row.row,
                "cohort_k": res.k,
                "cohort_prompt_len": res.prompt_len,
                "prompt_index": row.prompt_index,
                "prompt": row.prompt_text,
                "pad_tokens": row.pad_tokens,
                "n_tokens": len(row.token_ids),
                "generated_token_ids": row.token_ids,
                "token_ids_sha256": row.token_ids_sha256,
                "decoded_text": row.decoded_text,
                "cohort_wall_seconds": res.wall_seconds,
            }
            if extra:
                payload.update(extra)
            p = self.out_dir / f"result-c{res.cohort_id}-r{row.row}{suffix}.json"
            p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            paths.append(p)
        return paths

    def write_cohort_summary(self, res: CohortResult,
                             suffix: str = "") -> Path:
        payload = {
            "run_id": self.run_id,
            "cohort_id": res.cohort_id,
            "k": res.k,
            "prompt_len": res.prompt_len,
            "wall_seconds": res.wall_seconds,
            "decode_timings_ms": res.decode_timings_ms,
            "dedup": res.dedup,
            "counters": res.counters,
            "rows": [{
                "row": r.row, "prompt_index": r.prompt_index,
                "pad_tokens": r.pad_tokens,
                "token_ids_sha256": r.token_ids_sha256,
                "n_tokens": len(r.token_ids),
            } for r in res.rows],
            "route_journal": res.route_journal,
        }
        p = self.out_dir / f"cohort-c{res.cohort_id}{suffix}.json"
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return p

    def write_manifest(self, suffix: str = "") -> Path:
        sums = {}
        for p in sorted(self.out_dir.glob("*")):
            if p.is_file() and p.name != f"SHA256SUMS{suffix}.json":
                sums[p.name] = _sha256_path(p)
        m = self.out_dir / f"SHA256SUMS{suffix}.json"
        m.write_text(json.dumps(sums, indent=1, sort_keys=True),
                     encoding="utf-8")
        return m


# ---------------------------------------------------------------------------
# Local self-test: synthetic model end-to-end through the driver.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from tests.test_deepseek_v4_cohort import _build, _run_bf16  # noqa

    ckpt_rows = {}

    def _mk_ckpt(cid, row):
        buf = []
        ckpt_rows[(cid, row)] = buf

        def _hk(step, tok):
            buf.append((step, tok))
        return _hk

    journal_records = []

    def _layer_hook(layer_id):
        journal_records.append({"layer": layer_id,
                                "expert_ids_rank_order": [[1, 2], [2, 3]]})

    tmp = Path(os.environ.get("P5_SELFTEST_OUT",
                              Path(__file__).parent / "_p5_selftest"))
    model = _build(4)
    vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .")
    drv = ServeDriver(
        model,
        tokenize=lambda t: [ord(c) % 64 for c in t][:16] or [0],
        decode=lambda ids: "".join(vocab[i % len(vocab)] for i in ids),
        out_dir=tmp, run_id="p5-selftest",
        make_row_checkpoint=_mk_ckpt,
        on_cohort_counters=lambda cid: {"demo_counter": cid})
    drv.post_layer_hook = _layer_hook

    prompts = ["alpha prompt", "a much longer beta prompt!!", "gamma"]
    res = _run_bf16(drv.run_cohort, prompts, 0, [0, 1, 2], 5)
    assert res.k == 3 and len(res.rows) == 3
    assert all(len(r.token_ids) == 5 for r in res.rows)
    # Checkpoints fan out per row and match the returned streams.
    for r in range(3):
        buf = ckpt_rows[(0, r)]
        assert [t for _, t in buf] == res.rows[r].token_ids
        assert [s for s, _ in buf] == list(range(5))
    # Per-row equality vs sequential padded runs is the gate.
    ids_list, _ = build_cohort_ids(drv.tokenize, prompts, 0)
    for r in range(3):
        solo = _build(1)
        solo.reset_state()
        ref = _run_bf16(solo.generate,
                        torch.tensor([ids_list[r]]), 5, eos_id=-1)
        assert ref == res.rows[r].token_ids, f"row {r} diverged"
    drv.write_row_artifacts(res)
    drv.write_cohort_summary(res)
    m = drv.write_manifest()
    sums = json.loads(m.read_text())
    assert f"result-c0-r0.json" in sums
    assert len(sums) == 4  # 3 rows + 1 summary
    dedup = dedup_stats_from_journal(journal_records)
    # 10 fake records (2 layers x 5 forwards), each [[1,2],[2,3]]:
    # 4 slots, 3 uniques per record.
    assert dedup["records"] == 10
    assert dedup["request_slots"] == 40 and dedup["unique_experts_staged"] == 30
    print(f"SELFTEST PASS: cohort K={res.k} L*={res.prompt_len} "
          f"wall={res.wall_seconds}s artifacts={len(sums)}")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("phase5_serve_driver: library module; run with --selftest")
