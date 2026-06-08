"""埋め込みリトリーバ（任意のアブレーション・アーム）— 文字 n-gram の言い換え再現率の差を測る用途。

stdlib コアは本ファイルを import しない（依存・サーバ競合を隔離）。retriever=='embedding' の時だけ使う。
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field

_EMBED_URL = "http://localhost:11434/api/embeddings"


def _embed(text: str, model: str, timeout: float = 60.0) -> list[float]:
    body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(_EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp).get("embedding", [])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class EmbeddingRetriever:
    """ollama 埋め込みによる近傍検索。Retriever Protocol（add/search）に準拠。"""

    model: str = "nomic-embed-text"
    _vecs: dict[int, list[float]] = field(default_factory=dict)
    _cache: dict[str, list[float]] = field(default_factory=dict)

    def _vec_of(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = _embed(text, self.model)
        return self._cache[text]

    def add(self, doc_id: int, text: str) -> None:
        self._vecs[doc_id] = self._vec_of(text)

    def search(self, text: str, k: int, *, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if not self._vecs or k <= 0:
            return []
        q = self._vec_of(text)
        skip = exclude or set()
        scored = [(doc_id, _cosine(q, vec)) for doc_id, vec in self._vecs.items() if doc_id not in skip]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
