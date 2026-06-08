"""教師あり・非パラメトリック予測器 — 埋め込み上の kNN / ロジスティック回帰（LLM不使用）。

キャッシュの「非パラメトリック経路」。埋め込みは L2 正規化して数値を安定化。
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .dataset import LabeledItem
from .embedding_retriever import _embed


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed_matrix(items: Sequence[LabeledItem], model: str) -> list[list[float]]:
    return [_l2(_embed(it.text, model)) for it in items]


def knn_predict(train: Sequence[LabeledItem], test: Sequence[LabeledItem], model: str,
                k: int = 4) -> list[str]:
    Xtr = embed_matrix(train, model)
    ytr = [it.label for it in train]
    Xte = embed_matrix(test, model)
    preds = []
    for q in Xte:
        sims = sorted(((sum(a * b for a, b in zip(q, x)), i) for i, x in enumerate(Xtr)), reverse=True)[:k]
        votes = Counter(ytr[i] for _, i in sims)
        preds.append(votes.most_common(1)[0][0])
    return preds


def logreg_predict(train: Sequence[LabeledItem], test: Sequence[LabeledItem], model: str) -> list[str]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    Xtr = np.asarray(embed_matrix(train, model), dtype=float)
    Xte = np.asarray(embed_matrix(test, model), dtype=float)
    ytr = [it.label for it in train]
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(Xtr, ytr)
    return list(clf.predict(Xte))


def accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else 0.0
