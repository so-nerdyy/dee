"""Data assembly for the predictor lab: legal prediction examples from the
two real traces.

An example is one source context with its ground-truth target set:

    task ``next_layer``: source = (token t, layer l) set S_{t,l} + history,
                         target = S_{t,l+1}  (within the same token);
    task ``next_step`` : source = (token t, layer l) set S_{t,l} + history,
                         target = S_{t+1,l}  (same layer, next forward step).

History features use ONLY information observable at hint time (past layer
calls within the token and past forward steps).  No example ever sees its own
target.  Records are (layer, expert) pairs -- one packed expert record each.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

Rec = Tuple[int, int]          # (layer, expert) = one packed expert record


@dataclass
class Example:
    """One legal prediction event."""
    call: Tuple[int, int]            # (token_index, source layer)
    task: str                        # next_layer | next_step
    s_src: Tuple[int, ...]           # observed source set (rank order)
    s_prev: Tuple[int, ...]          # same-token layer l-1 set (empty if absent)
    s_prev2: Tuple[int, ...]         # same-token layer l-2 set
    s_step: Tuple[int, ...]          # same-layer previous-step set
    s_step2: Tuple[int, ...]         # same-layer two-steps-ago set
    ranks: Tuple[int, ...]           # routing ranks of s_src entries
    weights: Tuple[float, ...]       # routing weights where legally observable
    target: Tuple[Rec, ...]          # ground-truth target record set
    cluster: int                     # bootstrap cluster id (token / step)
    dst_layer: int                   # target layer (hint destination)


@dataclass
class Dataset:
    name: str                        # ornith_t2 | sealed_t1
    record_bytes: int
    stream: List[Rec] = field(default_factory=list)      # whole window stream
    splits: Dict[str, Dict[str, List[Example]]] = field(default_factory=dict)
    first_touch: Set[Rec] = field(default_factory=set)
    n_layers: int = 0
    experts_per_layer: int = 256


# ==========================================================================
# Stream construction from theory.sources RouterStream objects
# ==========================================================================
def _sets_by_token(stream) -> Dict[int, Dict[int, Tuple[int, ...]]]:
    out: Dict[int, Dict[int, Tuple[int, ...]]] = defaultdict(dict)
    for (tok, layer) in stream.keys:
        out[tok][layer] = tuple(stream.sets[(tok, layer)])
    return out


def build_examples(stream, task: str, weights_by_call=None) -> List[Example]:
    """Legal examples for one task over one RouterStream."""
    by_tok = _sets_by_token(stream)
    toks = sorted(by_tok)
    tok_id = {t: i for i, t in enumerate(toks)}
    ex: List[Example] = []
    for t in toks:
        layers = by_tok[t]
        prev_layers = by_tok.get(t - 1, {}) if task == "next_step" else {}
        prev2_layers = by_tok.get(t - 2, {}) if task == "next_step" else {}
        for l in sorted(layers):
            if task == "next_layer":
                dst_l = l + 1
                tgt = layers.get(dst_l)
            else:
                dst_l = l
                tgt = by_tok.get(t + 1, {}).get(l)
            if tgt is None:
                continue
            s_src = layers[l]
            ranks, weights = _ranks_weights(stream, (t, l), s_src, weights_by_call)
            ex.append(Example(
                call=(t, l), task=task,
                s_src=s_src,
                s_prev=layers.get(l - 1, ()),
                s_prev2=layers.get(l - 2, ()),
                s_step=prev_layers.get(l, ()),
                s_step2=prev2_layers.get(l, ()),
                ranks=ranks, weights=weights,
                target=tuple(sorted((dst_l, e) for e in set(tgt))),
                cluster=tok_id[t], dst_layer=dst_l,
            ))
    return ex


def _ranks_weights(stream, call, s_src, weights_by_call):
    ranks: Dict[int, int] = {}
    for row_i, row in enumerate(stream.raw.get(call, [])):
        for rank_i, e in enumerate(row):
            ranks.setdefault(e, row_i * 8 + rank_i)
    weights: Dict[int, float] = {}
    if weights_by_call is not None and call in weights_by_call:
        for e, (r, w) in weights_by_call[call].items():
            ranks[e] = r
            weights[e] = w
    return (tuple(ranks.get(e, len(s_src)) for e in s_src),
            tuple(float(weights.get(e, 0.0)) for e in s_src))


def load_datasets() -> Dict[str, Dataset]:
    """Return {ornith_t2, sealed_t1} Datasets with train/test splits per task."""
    from ..sources import load_sealed_stream
    from . import PRED_SPLIT_T1, PRED_SPLIT_T2, PRED_T2_RECORD_BYTES

    runs, wmaps = _load_ornith_with_weights()
    groups = PRED_SPLIT_T2.value

    def collect(group_names: Sequence[str], task: str):
        out: List[Example] = []
        base = 0
        for rn in group_names:
            ex = build_examples(runs[rn], task, wmaps.get(rn))
            for e in ex:
                e.cluster += base
            base += 1 + max((e.cluster for e in ex), default=-1)
            out.extend(ex)
        return out

    ornith_splits = {}
    for split in ("train", "test"):
        names = [n for g in groups[f"{split}_stream_groups"] for n in g]
        ornith_splits[split] = {task: collect(names, task)
                                for task in ("next_layer", "next_step")}
    all_names = [n for g in groups["train_stream_groups"] + groups["test_stream_groups"]
                 for n in g]
    ornith_stream = _dedup_stream(all_names, runs)

    sealed = load_sealed_stream()
    st1 = PRED_SPLIT_T1.value
    train_steps = set(st1["train_forward_steps"])
    test_steps = set(st1["test_forward_steps"])
    max_train = max(train_steps)
    sealed_splits = {"train": {}, "test": {}}
    for task in ("next_layer", "next_step"):
        allx = build_examples(sealed, task, None)
        sealed_splits["train"][task] = [
            e for e in allx if e.call[0] in train_steps
            # a next_step train example sourced at the last train step has its
            # LABEL in the test window; drop it so no test-period target is
            # ever used in fitting
            and not (task == "next_step" and e.call[0] == max_train)]
        sealed_splits["test"][task] = [
            e for e in allx if e.call[0] in test_steps
            and not (task == "next_step" and e.call[0] == max(test_steps))]
    sealed_stream = sealed.stream()

    ds = {
        "ornith_t2": Dataset(
            name="ornith_t2", record_bytes=int(PRED_T2_RECORD_BYTES.value),
            stream=ornith_stream, splits=ornith_splits, n_layers=40),
        "sealed_t1": Dataset(
            name="sealed_t1", record_bytes=13369344,
            stream=sealed_stream, splits=sealed_splits, n_layers=43),
    }
    for d in ds.values():
        d.first_touch = _first_touch(d.stream)
    return ds


def _load_ornith_with_weights():
    """Parse the Ornith trace once, rebuilding run streams AND per-call
    (expert -> (rank, weight)) maps, with the exact token-index mapping of
    sources.load_ornith_streams (first appearance of (step, sequence_token)
    within each run's rows)."""
    import gzip
    import json

    from ..paths import TRACE_JSONL_GZ
    from ..sources import ORNITH_EXPERTS, ORNITH_LAYERS, ORNITH_TOPK, RouterStream

    rows: Dict[str, List[dict]] = {}
    with gzip.open(TRACE_JSONL_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("event_type") != "route_selection":
                continue
            rows.setdefault(d["run_id"], []).append(d)

    runs: Dict[str, RouterStream] = {}
    wmaps: Dict[str, Dict[Tuple[int, int], Dict[int, Tuple[int, float]]]] = {}
    for run_id, rs in rows.items():
        tok_index: Dict[Tuple[int, int], int] = {}
        by_rank: Dict[Tuple[int, int], Dict[int, int]] = {}
        wmap: Dict[Tuple[int, int], Dict[int, Tuple[int, float]]] = {}
        order: List[Tuple[int, int]] = []
        phases: Dict[Tuple[int, int], str] = {}
        for d in rs:
            tok = (int(d["step"]), int(d["sequence_token"]))
            if tok not in tok_index:
                tok_index[tok] = len(tok_index)
            t = tok_index[tok]
            layer = int(d["layer"])
            k = (t, layer)
            if k not in by_rank:
                by_rank[k] = {}
                order.append(k)
                phases[k] = d.get("phase", "")
            by_rank[k][int(d["routing_rank"])] = int(d["expert"])
            wmap.setdefault(k, {})[int(d["expert"])] = (
                int(d["routing_rank"]), float(d["routing_weight"]))
        calls = {k: [[ranks[r] for r in sorted(ranks)]] for k, ranks in by_rank.items()}
        sets = {}
        for k, rowsk in calls.items():
            seq: List[int] = []
            for row in rowsk:
                for e in row:
                    if e not in seq:
                        seq.append(e)
            sets[k] = seq
        runs[run_id] = RouterStream(
            run_id=run_id, n_layers=ORNITH_LAYERS, experts_per_layer=ORNITH_EXPERTS,
            topk=ORNITH_TOPK, n_calls=len(order),
            n_slots=sum(1 for k in order for row in calls[k] for _e in row),
            keys=order, sets=sets, raw=calls, phases=phases,
        )
        wmaps[run_id] = wmap
    return runs, wmaps


def _dedup_stream(names: Sequence[str], runs) -> List[Rec]:
    """Cache-request stream over the unique routing streams (each byte-identical
    stream counted once, first occurrence in split order)."""
    seen = set()
    out: List[Rec] = []
    for rn in names:
        s = runs[rn]
        sig = tuple((k, tuple(s.sets[k])) for k in s.keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.extend((k[1], e) for k in s.keys for e in s.sets[k])
    return out


def _first_touch(stream: Sequence[Rec]) -> Set[Rec]:
    seen: Set[Rec] = set()
    ft: Set[Rec] = set()
    for r in stream:
        if r not in seen:
            seen.add(r)
            ft.add(r)
    return ft
