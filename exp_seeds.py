#!/usr/bin/env python3
"""多シード CI — 非パラメトリック・キャッシュ(kNN/embed-LR)の train サンプリング分散を測る。

埋め込みは各データで一度だけ計算し、seed ごとに train 部分集合をインデックス再標本化（高速）。
zeroshot-0.8B は train 非依存=決定論（Wilson CI を併記）。
"""
from __future__ import annotations

import json
import random
import statistics
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import split_same_theme
from cache_harness.embedding_retriever import _embed
from cache_harness.experiment import load_public, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
MID = "qwen3.5:0.8b"
SEEDS = [0, 1, 2, 3, 4]
TASKS_S = ["ext_imdb", "ext_yelp", "ext_emotion", "ext_dbpedia", "banking77"]
FRAC = 0.8


def emb_mat(items):
    import math
    out = []
    for it in items:
        v = _embed(it.text, EMBED)
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return np.asarray(out, dtype=float)


def knn(Xtr, ytr, Xte, k=4):
    sims = Xte @ Xtr.T
    preds = []
    for row in sims:
        idx = np.argsort(row)[::-1][:k]
        labs = [ytr[i] for i in idx]
        preds.append(max(set(labs), key=labs.count))
    return preds


def build_task(name, split) -> Task:
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))
    return Task(name, labels, "Classify: " + ", ".join(labels) + ".", lang="en")


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, [MID, EMBED])
    rows = []
    print(f"\n{'#'*88}\n# 多シード CI（{len(SEEDS)} seeds, train {int(FRAC*100)}% 部分標本）")
    print(f"  {'dataset':<12}{'cls':>4}{'zs-0.8B[CI]':>18}{'kNN(mean±sd)':>16}{'embedLR(mean±sd)':>18}")
    for name in TASKS_S:
        split = load_public(name)
        task = build_task(name, split)
        test = list(split.test)[:150]
        gold = [it.label for it in test]
        tr = list(split.train)

        # zeroshot（決定論）+ Wilson CI
        S = Settings(student_model=MID)
        zo = metrics.aggregate("z", run_arm(Classifier(None, S, client, task, (), ArmConfig()), test, [t.text for t in tr], S))
        client.save()

        Xtr_all = emb_mat(tr); ytr_all = [t.label for t in tr]
        Xte = emb_mat(test)
        knn_accs, lr_accs = [], []
        for seed in SEEDS:
            rng = random.Random(seed)
            idx = rng.sample(range(len(tr)), max(len(task.labels), int(len(tr) * FRAC)))
            Xs, ys = Xtr_all[idx], [ytr_all[i] for i in idx]
            knn_accs.append(sum(p == g for p, g in zip(knn(Xs, ys, Xte), gold)) / len(gold))
            clf = LogisticRegression(max_iter=3000, C=10.0).fit(Xs, ys)
            lr_accs.append(sum(p == g for p, g in zip(clf.predict(Xte), gold)) / len(gold))

        rec = {"task": name, "n_classes": len(task.labels),
               "zeroshot": zo.accuracy, "zeroshot_ci": list(zo.acc_ci),
               "knn_mean": statistics.mean(knn_accs), "knn_sd": statistics.pstdev(knn_accs),
               "embed_lr_mean": statistics.mean(lr_accs), "embed_lr_sd": statistics.pstdev(lr_accs)}
        rows.append(rec)
        lo, hi = zo.acc_ci
        zs_s = f"{zo.accuracy*100:.0f}[{lo*100:.0f},{hi*100:.0f}]"
        knn_s = f"{rec['knn_mean']*100:.1f}±{rec['knn_sd']*100:.1f}"
        lr_s = f"{rec['embed_lr_mean']*100:.1f}±{rec['embed_lr_sd']*100:.1f}"
        print(f"  {name:<12}{len(task.labels):>4}{zs_s:>18}{knn_s:>16}{lr_s:>18}")

    with open("seeds_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: seeds_results.json")


if __name__ == "__main__":
    main()
