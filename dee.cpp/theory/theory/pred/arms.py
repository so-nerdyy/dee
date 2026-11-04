"""Predictor arms.  Every arm exposes two scoring functions over the legal
candidate universe (the target layer's 256 expert ids):

    scores(ex)      -> (256,) context score: ranking key for the FULL source set
    pair_scores(ex) -> (|S_src|, 256) per-source-expert score: ranking key for
                       budget semantics "m candidates per source EXPERT"
                       (the semantics THEORY.md section 7 / FALSIFICATION P9-P10
                       were derived in -- see lab.py for the budget-semantics
                       reconciliation).

pair_scores(ex)[k, j] is DEFINED for every arm as the arm's score for
candidate j when the source set is reduced to the single expert S_src[k]
(history unchanged).  This is a uniform, leak-free definition.

Arms:
  popularity     -- per-target-layer empirical prior (context-free baseline)
  cond_pooled    -- empirical-conditional baseline P(j | S_l) = 1 - prod_i
                    (1 - P(j | i)), exactly theory/prefetch.py semantics
                    (expert-id keyed, pooled over layers, min_support=5)
  cond_perlayer  -- the same estimator keyed per layer-pair with backoff
                    per-layer -> pooled -> popularity
  logreg_pairwise-- hashed-feature logistic regression (scikit-learn SGD)
  mlp_set        -- small CPU torch MLP on multi-hot history features
  xlayer_bag     -- within-token cross-layer predictor on learned set
                    embeddings of earlier layers' sets in the same token

All fitting uses TRAIN examples only.  Hyperparameters are fixed in
theory/pred/__init__.py (never tuned on evaluation data).
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

Rec = Tuple[int, int]


def _hash(s: str, dim: int) -> int:
    return zlib.crc32(s.encode()) % dim


# ==========================================================================
# Shared statistical estimators
# ==========================================================================
def fit_popularity(train_ex, n_layers: int, n_experts: int = 256,
                   alpha: float = 1.0) -> Dict[int, np.ndarray]:
    """Smoothed P(j | dst_layer) over TRAIN target sets."""
    cnt = np.zeros((n_layers, n_experts))
    for ex in train_ex:
        for (_l, j) in ex.target:
            cnt[ex.dst_layer, j] += 1.0
    tot = cnt.sum(axis=1, keepdims=True)
    p = (cnt + alpha) / (tot + alpha * n_experts)
    return {"p": p}


def fit_conditional(train_ex, min_support: int, per_layer: bool):
    """Empirical P(j in target | i in S_src) over TRAIN examples.

    pooled:     keyed (i, j)                -- theory/prefetch.py convention
    per_layer:  keyed ((src_layer, i), j)   -- layer-pair specific
    """
    n_pair: Dict[tuple, int] = defaultdict(int)
    n_src: Dict[tuple, int] = defaultdict(int)
    for ex in train_ex:
        src = set(ex.s_src)
        tgt = {j for (_l, j) in ex.target}
        for i in src:
            ki = (ex.call[1], i) if per_layer else i
            n_src[ki] += 1
            for j in tgt:
                n_pair[(ki, j)] += 1
    cond = {}
    support = {}
    for (ki, j), c in n_pair.items():
        if n_src[ki] >= min_support:
            cond[(ki, j)] = c / n_src[ki]
            support[(ki, j)] = c
    return {"cond": cond, "support": support, "n_src": n_src}


def _aggregate(pair_p: List[Tuple[int, float]]) -> Dict[int, float]:
    """p_j(S) = 1 - prod_{i in S} (1 - P(j | i))  (conditional independence)."""
    agg: Dict[int, float] = {}
    for _i, j, p in pair_p:
        agg[j] = 1.0 - (1.0 - agg.get(j, 0.0)) * (1.0 - p)
    return agg


# ==========================================================================
# Arm base
# ==========================================================================
class Arm:
    name = "base"

    def fit(self, train_ex, meta) -> None:
        raise NotImplementedError

    def scores(self, ex) -> np.ndarray:
        """(256,) ranking scores for candidates of ex.dst_layer."""
        raise NotImplementedError

    def pair_scores(self, ex) -> np.ndarray:
        """(|S_src|, 256) per-source-expert ranking scores."""
        return np.stack([self._scores_single(ex, i) for i in ex.s_src]
                        ) if ex.s_src else np.zeros((0, 256))

    def _scores_single(self, ex, i: int) -> np.ndarray:
        """Score vector with the source set reduced to the single expert i."""
        raise NotImplementedError


# ==========================================================================
# 1. Popularity prior
# ==========================================================================
class PopularityArm(Arm):
    name = "popularity"

    def fit(self, train_ex, meta):
        self.pop = fit_popularity(train_ex, meta["n_layers"])
        self.n_layers = meta["n_layers"]

    def scores(self, ex):
        return self.pop["p"][ex.dst_layer]

    def _scores_single(self, ex, i):
        return self.pop["p"][ex.dst_layer]


# ==========================================================================
# 2-3. Empirical conditional arms
# ==========================================================================
class CondArm(Arm):
    per_layer = False
    backoff = False

    def fit(self, train_ex, meta):
        self.meta = meta
        self.pooled = fit_conditional(train_ex, meta["min_support"], per_layer=False)
        self.pl = (fit_conditional(train_ex, meta["min_support"], per_layer=True)
                   if self.per_layer else None)
        self.pop = fit_popularity(train_ex, meta["n_layers"]) if self.backoff else None

    def _pair(self, ex, i) -> Dict[int, float]:
        out: Dict[int, float] = {}
        if self.pl is not None:
            for j in range(256):
                v = self.pl["cond"].get(((ex.call[1], i), j))
                if v is not None:
                    out[j] = v
        if not out or self.backoff:
            for j in range(256):
                if j in out:
                    continue
                v = self.pooled["cond"].get((i, j))
                if v is not None:
                    out[j] = v
        if self.backoff:
            p = self.pop["p"][ex.dst_layer]
            for j in range(256):
                if j not in out:
                    out[j] = float(p[j])
        return out

    def pair_scores(self, ex):
        rows = []
        for i in ex.s_src:
            d = self._pair(ex, i)
            v = np.zeros(256)
            for j, p in d.items():
                v[j] = p
            rows.append(v)
        return np.stack(rows) if rows else np.zeros((0, 256))

    def scores(self, ex):
        agg = np.zeros(256)
        for i in ex.s_src:
            d = self._pair(ex, i)
            for j, p in d.items():
                agg[j] = 1.0 - (1.0 - agg[j]) * (1.0 - p)
        return agg

    def _scores_single(self, ex, i):
        v = np.zeros(256)
        for j, p in self._pair(ex, i).items():
            v[j] = p
        return v


class CondPooledArm(CondArm):
    name = "cond_pooled"


class CondPerLayerArm(CondArm):
    name = "cond_perlayer"
    per_layer = True
    backoff = True


# ==========================================================================
# 4. Hashed-feature logistic regression (scikit-learn)
# ==========================================================================
# Feature scheme (arithmetic hash, vectorized; fixed a priori):
#   (TAG_J, j)                candidate identity
#   (TAG_P, i, j)             i in source set (full set or single-source view)
#   (TAG_Q, i, j)             i in same-token layer l-1 set
#   (TAG_R, i, j)             i in same-layer previous-step set
#   (TAG_DL, dst_layer, j)    target-layer interaction
#   (TAG_SL, src_layer, j)    source-layer interaction
class LogRegArm(Arm):
    name = "logreg_pairwise"
    P1, P2, P3 = 2654435761, 40503, 97281
    TAG_J, TAG_P, TAG_Q, TAG_R, TAG_DL, TAG_SL = 1, 2, 3, 4, 5, 6

    def __init__(self, cfg):
        self.cfg = cfg

    def _contexts(self, ex, single=None):
        """Feature contexts for one example: full source set or single-source."""
        import numpy as np
        src = (single,) if single is not None else tuple(ex.s_src)
        dim = self.cfg["hash_dim"]
        js = np.arange(256, dtype=np.int64)
        cols = [((self.TAG_J * self.P3) + js * self.P2) % dim]
        for i in src:
            cols.append((self.TAG_P * self.P3 + i * self.P1 + js * self.P2) % dim)
        for i in ex.s_prev:
            cols.append((self.TAG_Q * self.P3 + i * self.P1 + js * self.P2) % dim)
        for i in ex.s_step:
            cols.append((self.TAG_R * self.P3 + i * self.P1 + js * self.P2) % dim)
        cols.append((self.TAG_DL * self.P3 + ex.dst_layer * self.P1 + js * self.P2) % dim)
        cols.append((self.TAG_SL * self.P3 + ex.call[1] * self.P1 + js * self.P2) % dim)
        return np.concatenate(cols)

    def _rows(self, examples, singles=False):
        """Sparse design matrix.  rows(ex) = 256-row block per context; with
        ``singles`` each example yields |S_src| blocks (one per single-source
        view), else one block (the full source set)."""
        from scipy.sparse import csr_matrix
        import numpy as np

        col_parts, row_parts, ys = [], [], []
        row = 0
        for ex in examples:
            views = list(ex.s_src) if singles else [None]
            for single in views:
                cols = self._contexts(ex, single)
                rows = np.repeat(np.arange(row, row + 256, dtype=np.int64),
                                 cols.size // 256)
                col_parts.append(cols)
                row_parts.append(rows)
                y = np.zeros(256, dtype=np.int64)
                for (_l, j) in ex.target:
                    y[j] = 1
                ys.append(y)
                row += 256
        if row == 0:
            return csr_matrix((0, self.cfg["hash_dim"])), np.zeros(0, dtype=int)
        data = np.ones(int(sum(c.size for c in col_parts)))
        X = csr_matrix((data,
                        (np.concatenate(row_parts), np.concatenate(col_parts))),
                       shape=(row, self.cfg["hash_dim"]))
        return X, np.concatenate(ys)

    def fit(self, train_ex, meta):
        from sklearn.linear_model import SGDClassifier
        from threadpoolctl import threadpool_limits

        self.meta = meta
        # train on BOTH context types (full set + single-source views) so both
        # scoring interfaces see in-distribution contexts
        X_full, y_full = self._rows(train_ex, singles=False)
        X_sing, y_sing = self._rows(train_ex, singles=True)
        from scipy.sparse import vstack
        X = vstack([X_full, X_sing])
        y = np.concatenate([y_full, y_sing])
        self.clf = SGDClassifier(loss="log_loss", alpha=self.cfg["alpha"],
                                 max_iter=self.cfg["max_iter"], random_state=meta["seed"],
                                 class_weight="balanced")
        with threadpool_limits(1):     # deterministic BLAS reductions
            self.clf.fit(X, y)

    def _predict(self, examples, singles):
        from threadpoolctl import threadpool_limits

        X, _ = self._rows(examples, singles=singles)
        with threadpool_limits(1):
            p = self.clf.predict_proba(X)[:, 1]
        n_ctx = len(examples) if not singles else sum(max(1, len(e.s_src)) for e in examples)
        return p.reshape(n_ctx, 256)

    def scores(self, ex):
        return self._predict([ex], singles=False)[0]

    def scores_many(self, examples):
        return self._predict(examples, singles=False)

    def pair_scores(self, ex):
        if not ex.s_src:
            return np.zeros((0, 256))
        return self._predict([ex], singles=True)


# ==========================================================================
# 5-6. Torch arms (CPU, deterministic)
# ==========================================================================
class _TorchArm(Arm):
    """Shared machinery for the two torch arms."""

    def _torch(self):
        """Seed BEFORE any network construction: torch weight init draws from
        the process-global RNG, which is randomly seeded at process start."""
        import torch
        torch.manual_seed(self.meta["seed"])
        torch.set_num_threads(1)
        return torch

    def _train(self, net, X, Y, steps, lr, pos_weight):
        torch = self._torch()
        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        n = Xt.shape[0]
        batch = min(256, n)
        g = torch.Generator().manual_seed(self.meta["seed"])
        for step in range(steps):
            perm = torch.randperm(n, generator=g)[:batch]
            opt.zero_grad()
            loss = lossf(net(Xt[perm]), Yt[perm])
            loss.backward()
            opt.step()
        self.net = net.eval()

    def _sig(self, X):
        torch = self._torch()
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32))
            return torch.sigmoid(z).numpy()


class MLPSetArm(_TorchArm):
    name = "mlp_set"

    def __init__(self, cfg):
        self.cfg = cfg

    def _vec(self, src, ex) -> np.ndarray:
        v = np.zeros(5 * 256 + 43 + 3)
        for k, s in enumerate((src, ex.s_prev, ex.s_prev2, ex.s_step, ex.s_step2)):
            for i in s:
                v[k * 256 + i] = 1.0
        v[5 * 256 + (ex.call[1] % 43)] = 1.0
        w = ex.weights if ex.weights else (0.0,)
        r = ex.ranks if ex.ranks else (0,)
        v[-3] = float(np.mean(w))
        v[-2] = float(np.mean(r)) / 8.0
        v[-1] = 1.0
        return v

    def fit(self, train_ex, meta):
        self.meta = meta
        torch = self._torch()          # seed before Linear() weight init
        X = np.stack([self._vec(ex.s_src, ex) for ex in train_ex])
        Y = np.zeros((len(train_ex), 256))
        for a, ex in enumerate(train_ex):
            for (_l, j) in ex.target:
                Y[a, j] = 1.0
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.cfg["hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.cfg["hidden"], 256))
        pos = max(1.0, Y.sum())
        pos_weight = min(200.0, (Y.size - pos) / pos)
        self._train(net, X, Y, self.cfg["steps"], self.cfg["lr"], pos_weight)

    def scores(self, ex):
        return self._sig(self._vec(ex.s_src, ex)[None, :])[0]

    def scores_many(self, examples):
        X = np.stack([self._vec(ex.s_src, ex) for ex in examples])
        return self._sig(X)

    def pair_scores(self, ex):
        if not ex.s_src:
            return np.zeros((0, 256))
        X = np.stack([self._vec((i,), ex) for i in ex.s_src])
        return self._sig(X)


class XLayerBagArm(_TorchArm):
    name = "xlayer_bag"

    def __init__(self, cfg):
        self.cfg = cfg

    def _bags(self, src, ex):
        return (tuple(src), tuple(ex.s_prev) if ex.task == "next_layer" else tuple(ex.s_step),
                tuple(ex.s_prev2) if ex.task == "next_layer" else tuple(ex.s_step2))

    def fit(self, train_ex, meta):
        self.meta = meta
        torch = self._torch()          # seed before Embedding()/Linear() init
        E = self.cfg["emb_dim"]
        cfg = self.cfg

        class Net(torch.nn.Module):
            def __init__(self, layers_dim):
                super().__init__()
                self.emb = torch.nn.Embedding(256, E)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(3 * E + layers_dim, cfg["hidden"]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(cfg["hidden"], 256))
                self.layers_dim = layers_dim

            def forward(self, bags, layer_oh):
                # bags: (B, 3, maxlen) with -1 padding
                x = []
                for s in range(3):
                    idx = bags[:, s, :]
                    mask = (idx >= 0).float().unsqueeze(-1)
                    idx_c = idx.clamp(min=0)
                    e = self.emb(idx_c) * mask
                    x.append(e.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0))
                h = torch.cat(x + [layer_oh], dim=1)
                return self.mlp(h)

        self._net_cls = Net
        self._torch_mod = torch
        B, L, Y = self._batch(train_ex)
        net = Net(43)
        pos = max(1.0, Y.sum())
        pos_weight = min(200.0, (Y.size - pos) / pos)
        self._train_net(net, B, L, Y, pos_weight)

    def _batch(self, examples, src_of=None):
        torch = self._torch_mod
        maxl = 16
        B = np.full((len(examples), 3, maxl), -1, dtype=np.int64)
        L = np.zeros((len(examples), 43))
        Y = np.zeros((len(examples), 256))
        for a, ex in enumerate(examples):
            src = src_of(ex, a) if src_of else ex.s_src
            bags = self._bags(src, ex)
            for s, bag in enumerate(bags):
                for k, i in enumerate(bag[:maxl]):
                    B[a, s, k] = i
            L[a, ex.call[1] % 43] = 1.0
            for (_l, j) in ex.target:
                Y[a, j] = 1.0
        return B, L, Y

    def _train_net(self, net, B, L, Y, pos_weight):
        torch = self._torch_mod
        Bt = torch.tensor(B, dtype=torch.long)
        Lt = torch.tensor(L, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg["lr"])
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        n = Bt.shape[0]
        batch = min(256, n)
        g = torch.Generator().manual_seed(self.meta["seed"])
        for step in range(self.cfg["steps"]):
            perm = torch.randperm(n, generator=g)[:batch]
            opt.zero_grad()
            loss = lossf(net(Bt[perm], Lt[perm]), Yt[perm])
            loss.backward()
            opt.step()
        self.net = net.eval()

    def _sig(self, B, L):
        torch = self._torch_mod
        with torch.no_grad():
            z = self.net(torch.tensor(B, dtype=torch.long),
                         torch.tensor(L, dtype=torch.float32))
            return torch.sigmoid(z).numpy()

    def scores_many(self, examples):
        B, L, _ = self._batch(examples)
        return self._sig(B, L)

    def scores(self, ex):
        return self.scores_many([ex])[0]

    def pair_scores(self, ex):
        if not ex.s_src:
            return np.zeros((0, 256))
        rows = []
        for i in ex.s_src:
            B, L, _ = self._batch([ex], src_of=lambda e, a: (i,))
            rows.append(self._sig(B, L)[0])
        return np.stack(rows)


def build_arms() -> List[Arm]:
    from . import PRED_LR, PRED_MLP, PRED_XLAYER
    return [
        PopularityArm(),
        CondPooledArm(),
        CondPerLayerArm(),
        LogRegArm(PRED_LR.value),
        MLPSetArm(PRED_MLP.value),
        XLayerBagArm(PRED_XLAYER.value),
    ]
