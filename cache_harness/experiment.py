"""実験オーケストレーション（タスク非依存）— アブレーション/学習曲線/多モデルの共通部品。

新規性（汎化の指標）は **検索指標と独立**な difflib 編集距離比で判定する（循環論法の回避）。
"""
from __future__ import annotations

import difflib
import json
import random
from collections.abc import Sequence
from pathlib import Path

from .cache import Experience, ExperienceCache
from .classifier import Classifier
from .config import Settings
from .dataset import LabeledItem, Split
from .llm import CachingClient, OllamaClient
from .metrics import Outcome
from .parsing import parse_label
from .prompt import build_messages
from .retriever import NgramRetriever
from .rules import distill_rules
from .task import Task


def make_retriever(settings: Settings):
    if settings.retriever == "embedding":
        from .embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever(model=settings.embed_model)
    return NgramRetriever(sizes=settings.ngram_sizes)


def make_cache(train: Sequence[LabeledItem], settings: Settings, *, source: str = "seed") -> ExperienceCache:
    cache = ExperienceCache(retriever=make_retriever(settings))
    cache.add_many([Experience(it.text, it.label, source=source, theme=it.theme) for it in train])
    return cache


def load_public(task_name: str, data_dir: str = "data") -> Split:
    """data/<task>_{train,dev,test}.jsonl を読み、Split として返す（theme=タスク名）。"""
    def _read(split: str) -> tuple[LabeledItem, ...]:
        path = Path(data_dir) / f"{task_name}_{split}.jsonl"
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                items.append(LabeledItem(row["text"], row["label"], task_name, True))
        return tuple(items)
    return Split(_read("train"), _read("dev"), _read("test"), "public")


def _max_edit_ratio(text: str, train_texts: Sequence[str]) -> float:
    return max((difflib.SequenceMatcher(None, text, t).ratio() for t in train_texts), default=0.0)


def to_outcome(item: LabeledItem, pred, train_texts: Sequence[str], settings: Settings) -> Outcome:
    edit_sim = _max_edit_ratio(item.text, train_texts)
    return Outcome(
        text=item.text, gold=item.label, pred=pred.label, valid=pred.valid,
        latency=pred.latency, llm_calls=pred.n_llm_calls, route=pred.source,
        novel=edit_sim < settings.dup_edit_ratio, sim=pred.top_score, edit_sim=edit_sim,
    )


def run_arm(classifier: Classifier, test: Sequence[LabeledItem], train_texts: Sequence[str],
            settings: Settings) -> list[Outcome]:
    return [to_outcome(it, classifier.classify(it.text), train_texts, settings) for it in test]


def static_sample(train: Sequence[LabeledItem], k: int, seed: int, labels: Sequence[str]) -> tuple[Experience, ...]:
    """静的 few-shot 用の均衡サンプル（動的検索との公平比較）。"""
    rng = random.Random(seed)
    by_label: dict[str, list[LabeledItem]] = {label: [] for label in labels}
    for it in train:
        if it.label in by_label:
            by_label[it.label].append(it)
    chosen: list[LabeledItem] = []
    depth = 0
    while len(chosen) < k and any(depth < len(by_label[l]) for l in labels):
        for label in labels:
            bucket = by_label[label]
            if depth == 0:
                rng.shuffle(bucket)
            if depth < len(bucket) and len(chosen) < k:
                chosen.append(bucket[depth])
        depth += 1
    return tuple(Experience(it.text, it.label, source="seed", theme=it.theme) for it in chosen)


def distill_from_train(train: Sequence[LabeledItem], settings: Settings, client: CachingClient,
                       task: Task) -> tuple[tuple[str, ...], list[tuple[str, str, str]]]:
    """素の生徒モデルを train に当てて誤分類を集め、ルールを蒸留（test は不可視）。"""
    train_eval: list[tuple[str, str, str]] = []
    for it in train:
        msgs = build_messages(it.text, task)
        res = client.chat(settings.student_model, msgs, think=False, num_predict=settings.student_num_predict)
        label, _v, _a = parse_label(res.content, task.labels) if res.ok else (None, False, True)
        train_eval.append((it.text, it.label, label or ""))
    return distill_rules(train_eval, task, max_rules=settings.max_rules), train_eval


def make_client(path: str | None = None) -> CachingClient:
    client = CachingClient(inner=OllamaClient(), path=path)
    client.load()
    return client
