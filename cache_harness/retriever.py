"""近傍検索 — 文字 n-gram TF-IDF コサイン（stdlib のみ）。

日本語の短文では分かち書きが要らない文字 n-gram が頑健。依存ゼロで即動く。
インタフェースは差し替え可能（埋め込みリトリーバを後から足せるよう Protocol を定義）。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def char_ngrams(text: str, sizes: tuple[int, ...] = (2, 3)) -> list[str]:
    """文字 n-gram を生成。空白等は除かずそのまま（短文の癖を残す）。"""
    text = text.strip()
    grams: list[str] = []
    for n in sizes:
        if len(text) < n:
            if text:
                grams.append(text)
            continue
        grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


@runtime_checkable
class Retriever(Protocol):
    """近傍検索の共通契約。cache 層はこの契約にのみ依存する。"""

    def add(self, doc_id: int, text: str) -> None: ...

    def search(self, text: str, k: int, *, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        """類似度降順で (doc_id, cosine) を最大 k 件。cosine は 0..1。"""
        ...


@dataclass
class NgramRetriever:
    """文字 n-gram TF-IDF コサイン近傍検索。インクリメンタル add 対応（IDF は遅延再計算）。"""

    sizes: tuple[int, ...] = (2, 3)
    _tf: dict[int, Counter[str]] = field(default_factory=dict)
    _df: Counter[str] = field(default_factory=Counter)
    _n_docs: int = 0

    def add(self, doc_id: int, text: str) -> None:
        grams = char_ngrams(text, self.sizes)
        tf = Counter(grams)
        self._tf[doc_id] = tf
        for gram in tf:  # df は「その gram を含む文書数」
            self._df[gram] += 1
        self._n_docs += 1

    def _idf(self, gram: str) -> float:
        # 平滑化 IDF。未知 gram でも 0 除算しない。
        return math.log((1 + self._n_docs) / (1 + self._df.get(gram, 0))) + 1.0

    def _vector(self, tf: Counter[str]) -> dict[str, float]:
        return {gram: count * self._idf(gram) for gram, count in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        # 小さい方を走査
        if len(a) > len(b):
            a, b = b, a
        dot = sum(weight * b.get(gram, 0.0) for gram, weight in a.items())
        if dot == 0.0:
            return 0.0
        na = math.sqrt(sum(w * w for w in a.values()))
        nb = math.sqrt(sum(w * w for w in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, text: str, k: int, *, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self._n_docs == 0 or k <= 0:
            return []
        query_vec = self._vector(Counter(char_ngrams(text, self.sizes)))
        if not query_vec:
            return []
        skip = exclude or set()
        scored: list[tuple[int, float]] = []
        for doc_id, tf in self._tf.items():
            if doc_id in skip:
                continue
            scored.append((doc_id, self._cosine(query_vec, self._vector(tf))))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
