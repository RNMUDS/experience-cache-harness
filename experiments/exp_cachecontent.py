#!/usr/bin/env python3
"""キャッシュ内容のアブレーション — 何を入れるかで性能を比較。

内容パターン:
  gold-all   : train 全件 × 検証済み正解（既定）
  failure    : モデルが zero-shot で間違えた train 事例のみ × 正解（失敗経験＝矯正メモリ）
  self-resp  : train 全件 × モデル自身の予測ラベル（単なる応答, 正誤未検証）
  gold-match : gold をランダムに |failure| 件へ縮小（サイズ効果の対照）
評価: cache-kNN / cache-LR（LLM不使用）と LLM dyn-embed の test 精度。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import random
import sys

from cache_harness import metrics
from cache_harness.cache import Experience, ExperienceCache
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import LabeledItem, split_same_theme
from cache_harness.experiment import distill_from_train, load_public, make_client, make_retriever, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, knn_predict, logreg_predict
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
MODEL = "qwen3.5:0.8b"
N_TEST = 100
S = Settings(student_model=MODEL, retriever="embedding")
TASK_NAMES = ["jp_sentiment_authored", "agnews", "banking77"]


def build_task(name, split) -> Task:
    if name == "jp_sentiment_authored":
        return SENTIMENT_JP
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))
    return Task(name, labels, "Classify into one of: " + ", ".join(labels) + ".", lang="en")


def load(name):
    return split_same_theme(0) if name == "jp_sentiment_authored" else load_public(name)


def items_from(pairs):  # pairs: list[(text, label, theme)]
    return [LabeledItem(t, l, th) for (t, l, th) in pairs]


def knn_acc(cache_items, test, gold):
    if not cache_items:
        return None
    return accuracy(knn_predict(cache_items, test, EMBED, k=min(4, len(cache_items))), gold)


def lr_acc(cache_items, test, gold):
    if len({it.label for it in cache_items}) < 2:
        return None  # logreg は2クラス以上必要
    try:
        return accuracy(logreg_predict(cache_items, test, EMBED), gold)
    except Exception:
        return None


def llm_acc(cache_items, task, test, client):
    cache = ExperienceCache(retriever=make_retriever(S))
    cache.add_many([Experience(it.text, it.label, theme=it.theme) for it in cache_items])
    tr = [it.text for it in cache_items]
    clf = Classifier(cache, S, client, task, (), ArmConfig(use_retrieval=True))
    return metrics.aggregate("c", run_arm(clf, test, tr, S)).accuracy if cache_items else None


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL, EMBED]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)

    rows = []
    for name in TASK_NAMES:
        sp = load(name); task = build_task(name, sp)
        test = list(sp.test)[:N_TEST]; gold = [it.label for it in test]
        # train 上のモデル zero-shot 予測（text,gold,pred）
        _rules, train_eval = distill_from_train(sp.train, S, client, task)
        client.save()
        by_text_theme = {it.text: it.theme for it in sp.train}

        gold_all = [(t, g, by_text_theme.get(t, name)) for (t, g, p) in train_eval]
        failure = [(t, g, by_text_theme.get(t, name)) for (t, g, p) in train_eval if p and p != g]
        self_resp = [(t, p, by_text_theme.get(t, name)) for (t, g, p) in train_eval if p in task.labels]
        rng = random.Random(0); gm = list(gold_all); rng.shuffle(gm)
        gold_match = gm[:len(failure)] if failure else []

        contents = [("gold-all", gold_all), ("failure", failure), ("self-resp", self_resp),
                    ("gold-match", gold_match)]
        rec = {"task": name, "n_classes": len(task.labels), "n_train": len(gold_all),
               "model_train_acc": sum(1 for (t, g, p) in train_eval if p == g) / len(train_eval),
               "contents": {}}
        print(f"\n{'#'*92}\n# {name} ({len(task.labels)}cls)  train={len(gold_all)} "
              f"model-train-acc={rec['model_train_acc']*100:.0f}%  |failure|={len(failure)}")
        print(f"  {'content':<12}{'size':>6}{'kNN':>8}{'embedLR':>9}{'LLM-dyn':>9}")
        for cname, pairs in contents:
            it = items_from(pairs)
            ka = knn_acc(it, test, gold); la = lr_acc(it, test, gold); lm = llm_acc(it, task, test, client)
            rec["contents"][cname] = {"size": len(it), "knn": ka, "embed_lr": la, "llm_dyn": lm}
            fmt = lambda v: f"{v*100:.1f}" if v is not None else "n/a"
            print(f"  {cname:<12}{len(it):>6}{fmt(ka):>8}{fmt(la):>9}{fmt(lm):>9}")
            client.save()
        rows.append(rec)

    with open("results/cachecontent_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: cachecontent_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
