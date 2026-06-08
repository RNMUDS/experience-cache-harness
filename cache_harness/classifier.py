"""制御プレーン — 1 件の判定を司る（タスク非依存）。小型モデルは最後の短い判断だけを担う。

アブレーション各アームは ArmConfig のフラグだけで切り替える（プロンプト書式は prompt.py に一元化）。
provenance（誰が答えたか）を必ず記録し、教師の正解を生徒の手柄にしない。
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from .cache import Experience, ExperienceCache, RetrievalHit
from .config import Settings
from .llm import OllamaClient
from .parsing import parse_label
from .prompt import build_messages
from .task import Task


@dataclass(frozen=True)
class Prediction:
    label: str | None
    valid: bool
    raw: str
    source: str            # short_circuit | student | teacher | student_then_teacher | nn_copy
    n_llm_calls: int
    latency: float
    top_score: float
    retrieved_labels: tuple[str, ...]
    ambiguous: bool


@dataclass(frozen=True)
class ArmConfig:
    use_retrieval: bool = False
    use_rules: bool = False
    use_short_circuit: bool = False
    use_escalation: bool = False
    corrupt_exemplars: bool = False
    use_nn_route: bool = False   # 高信頼の近傍は NN 多数決で 0 LLM 即答、低信頼のみ LLM へ
    mode: str = "llm"      # llm | nn_copy
    static_exemplars: tuple[Experience, ...] | None = None


def _rotate(labels: Sequence[str]) -> dict[str, str]:
    """ラベルを 1 つずらす破壊写像（コピー検出プローブ用）。"""
    return {labels[i]: labels[(i + 1) % len(labels)] for i in range(len(labels))}


def _majority(labels: Sequence[str], scores: Sequence[float], order: Sequence[str]) -> str:
    if not labels:
        return order[0]
    counts: Counter[str] = Counter(labels)
    weight: dict[str, float] = defaultdict(float)
    for label, score in zip(labels, scores):
        weight[label] += score
    best = max(counts.values())
    tied = [label for label, c in counts.items() if c == best]
    tied.sort(key=lambda label: (-weight[label], order.index(label) if label in order else 99))
    return tied[0]


def _select_exemplars(hits: list[RetrievalHit], k: int, balance: bool, labels: Sequence[str]) -> list[RetrievalHit]:
    if k <= 0 or not hits:
        return []
    if not balance:
        chosen = hits[:k]
    else:
        by_label: dict[str, list[RetrievalHit]] = defaultdict(list)
        for hit in hits:
            by_label[hit.experience.label].append(hit)
        chosen = []
        depth = 0
        while len(chosen) < k and any(depth < len(by_label[l]) for l in labels):
            for label in labels:
                bucket = by_label[label]
                if depth < len(bucket) and len(chosen) < k:
                    chosen.append(bucket[depth])
            depth += 1
    return sorted(chosen, key=lambda h: h.score)


@dataclass
class Classifier:
    cache: ExperienceCache | None
    settings: Settings
    client: OllamaClient
    task: Task
    rules: tuple[str, ...] = ()
    arm: ArmConfig = field(default_factory=ArmConfig)

    def _exemplar_pairs(self, text: str) -> tuple[list[tuple[str, str]], float, tuple[str, ...]]:
        if self.arm.static_exemplars is not None:
            pairs = [(e.text, e.label) for e in self.arm.static_exemplars]
            labels = tuple(e.label for e in self.arm.static_exemplars)
            return self._maybe_corrupt(pairs), 0.0, labels
        if not (self.arm.use_retrieval and self.cache):
            return [], 0.0, ()
        pool = self.cache.retrieve(text, max(self.settings.k * 4, self.settings.k))
        top = pool[0].score if pool else 0.0
        gated = [h for h in pool if h.score >= self.settings.min_exemplar_sim]
        chosen = _select_exemplars(gated, self.settings.k, self.settings.balance_kshot, self.task.labels)
        pairs = [(h.experience.text, h.experience.label) for h in chosen]
        return self._maybe_corrupt(pairs), top, tuple(h.experience.label for h in chosen)

    def _maybe_corrupt(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if not self.arm.corrupt_exemplars:
            return pairs
        rot = _rotate(self.task.labels)
        return [(text, rot.get(label, label)) for text, label in pairs]

    def _call(self, model: str, messages: list[dict[str, str]], num_predict: int):
        res = self.client.chat(model, messages, think=False, num_predict=num_predict)
        if not res.ok:
            return None, False, True, res.content, res.latency
        label, valid, ambiguous = parse_label(res.content, self.task.labels)
        return label, valid, ambiguous, res.content, res.latency

    def _messages(self, text: str, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return build_messages(text, self.task, rules=self.rules if self.arm.use_rules else (),
                              exemplars=pairs, max_exemplar_chars=self.settings.max_prompt_chars)

    def classify(self, text: str) -> Prediction:
        t0 = time.time()

        if self.arm.mode == "nn_copy":
            pool = self.cache.retrieve(text, self.settings.k) if self.cache else []
            label = _majority([h.experience.label for h in pool], [h.score for h in pool], self.task.labels)
            top = pool[0].score if pool else 0.0
            return Prediction(label, True, "", "nn_copy", 0, time.time() - t0, top,
                              tuple(h.experience.label for h in pool), False)

        if self.arm.use_short_circuit and self.cache:
            hit = self.cache.short_circuit(text, self.settings.tau_short_circuit)
            if hit is not None:
                return Prediction(hit.experience.label, True, "", "short_circuit", 0,
                                  time.time() - t0, hit.score, (hit.experience.label,), False)

        # 近傍ルート: 検索が高信頼なら小型LLMを使わず NN 多数決（小型LLMが弱いタスクで有効）
        if self.arm.use_nn_route and self.cache:
            pool = self.cache.retrieve(text, self.settings.k)
            if pool and pool[0].score >= self.settings.tau_nn:
                label = _majority([h.experience.label for h in pool], [h.score for h in pool], self.task.labels)
                return Prediction(label, True, "", "nn_route", 0, time.time() - t0, pool[0].score,
                                  tuple(h.experience.label for h in pool), False)

        pairs, top_score, retrieved = self._exemplar_pairs(text)

        if self.arm.use_escalation and self.arm.use_retrieval and top_score < self.settings.tau_escalate:
            label, valid, ambiguous, raw, lat = self._call(
                self.settings.teacher_model, self._messages(text, pairs), self.settings.teacher_num_predict)
            return Prediction(label, valid, raw, "teacher", 1, lat, top_score, retrieved, ambiguous)

        messages = self._messages(text, pairs)
        label, valid, ambiguous, raw, lat = self._call(
            self.settings.student_model, messages, self.settings.student_num_predict)

        if self.arm.use_escalation and (ambiguous or not valid):
            t_label, t_valid, t_amb, t_raw, t_lat = self._call(
                self.settings.teacher_model, messages, self.settings.teacher_num_predict)
            return Prediction(t_label, t_valid, t_raw, "student_then_teacher", 2,
                              lat + t_lat, top_score, retrieved, t_amb)

        return Prediction(label, valid, raw, "student", 1, lat, top_score, retrieved, ambiguous)
