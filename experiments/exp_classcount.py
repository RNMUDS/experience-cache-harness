#!/usr/bin/env python3
"""決定打: クラス数 vs 精度 — 小型LLM(zeroshot)はクラス数とともに崩壊、キャッシュ(kNN/embed-LR)は保つ。

タスクをクラス数順に: SST-2(2)/JP感情(3)/AG News(4)/TREC(6)/banking77(77)。
方式: zeroshot(LLM, 0.5B & 0.8B) / kNN(埋込) / embed-LR(埋込)。各々 Wilson 95%CI。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import split_same_theme
from cache_harness.experiment import load_public, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, knn_predict, logreg_predict
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
ZS_MODELS = ["qwen2.5:0.5b", "qwen3.5:0.8b"]
N_TEST = 200


def build_task(name: str, split) -> Task:
    if name == "jp_sentiment_authored":
        return SENTIMENT_JP
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))  # banking77 等を動的構築
    instr = "Classify the customer banking query into one of these intents: " + ", ".join(labels) + "."
    return Task(name, labels, instr, lang="en")


def load(name):
    return split_same_theme(0) if name == "jp_sentiment_authored" else load_public(name)


def ci(pred, gold):
    n = len(gold); c = sum(p == g for p, g in zip(pred, gold))
    return c / n, (metrics._wilson_low(c, n), metrics._wilson_high(c, n))


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, ZS_MODELS + [EMBED])
    tasks = ["sst2", "jp_sentiment_authored", "agnews", "trec", "banking77"]
    rows = []
    print(f"\n{'#'*92}\n# クラス数 vs 精度（zeroshot LLM vs 非パラメトリック・キャッシュ）")
    print(f"  {'task':<16}{'classes':>8}{'zs-0.5B':>9}{'zs-0.8B':>9}{'kNN':>8}{'embedLR':>9}")
    for name in tasks:
        split = load(name)
        task = build_task(name, split)
        train = list(split.train)
        test = list(split.test)[:N_TEST]
        gold = [it.label for it in test]
        tr_txt = [it.text for it in train]

        zs = {}
        for m in ZS_MODELS:
            S = Settings(student_model=m)
            outs = run_arm(Classifier(None, S, client, task, (), ArmConfig()), test, tr_txt, S)
            zs[m] = metrics.aggregate("z", outs).accuracy
        client.save()
        knn_acc, knn_ci = ci(knn_predict(train, test, EMBED, k=4), gold)
        lr_acc, lr_ci = ci(logreg_predict(train, test, EMBED), gold)

        rows.append({"task": name, "n_classes": len(task.labels), "n_test": len(test),
                     "zeroshot": zs, "knn": knn_acc, "knn_ci": knn_ci, "embed_lr": lr_acc, "embed_lr_ci": lr_ci})
        print(f"  {name:<16}{len(task.labels):>8}{zs[ZS_MODELS[0]]*100:>8.1f}%{zs[ZS_MODELS[1]]*100:>8.1f}%"
              f"{knn_acc*100:>7.1f}%{lr_acc*100:>8.1f}%")

    with open("results/classcount_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: classcount_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
