"""経験キャッシュ — 検証済みの「役立った経験」を記憶し、瞬時に再利用する中核層。

人間の「経験から学ぶ」をコンピュータの完全記憶 + 高速検索で再現する:
  - add(experience): 検証済みの (入力 → 正答) を蓄積。
  - retrieve(text, k): 類似経験 top-k を動的 few-shot 用に取り出す。
  - short_circuit(text, tau): ほぼ同一の経験があれば LLM を呼ばず即答（速度・決定論）。
永続化で「セッションを跨いで覚え続ける」を実現。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .retriever import NgramRetriever, Retriever


@dataclass(frozen=True)
class Experience:
    """1 件の経験（不変）。source は学びの出所を記録（検証可能性のため）。"""

    text: str
    label: str
    insight: str = ""           # 任意の蒸留メモ（例: "客観的事実なので neutral"）
    source: str = "gold"        # gold | teacher | self
    theme: str = ""             # 由来テーマ（リーク監査・分析用）

    def as_shot(self) -> str:
        """few-shot 1 行表現。"""
        return f'文:「{self.text}」→ {self.label}'


@dataclass
class RetrievalHit:
    experience: Experience
    score: float


@dataclass
class ExperienceCache:
    """経験の追加・検索・短絡・永続化。検索器は差し替え可能。"""

    retriever: Retriever = field(default_factory=NgramRetriever)
    _items: list[Experience] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[Experience, ...]:
        return tuple(self._items)

    def add(self, exp: Experience) -> None:
        doc_id = len(self._items)
        self._items.append(exp)
        self.retriever.add(doc_id, exp.text)

    def add_many(self, experiences: list[Experience]) -> None:
        for exp in experiences:
            self.add(exp)

    def retrieve(self, text: str, k: int, *, exclude_ids: set[int] | None = None) -> list[RetrievalHit]:
        hits = self.retriever.search(text, k, exclude=exclude_ids)
        return [RetrievalHit(self._items[doc_id], score) for doc_id, score in hits]

    def short_circuit(self, text: str, tau: float) -> RetrievalHit | None:
        """最類似が tau 以上なら、その経験を即答候補として返す（LLM を呼ばない）。"""
        top = self.retrieve(text, 1)
        if top and top[0].score >= tau:
            return top[0]
        return None

    # 永続化（セッションを跨いで覚え続ける）
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(e) for e in self._items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, retriever: Retriever | None = None) -> "ExperienceCache":
        cache = cls(retriever=retriever or NgramRetriever())
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cache.add_many([Experience(**row) for row in data])
        return cache
