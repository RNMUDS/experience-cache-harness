#!/usr/bin/env python3
"""外部・実データでの再検証 — 主要知見(クラス数頑健性 / cache>小型LLM)が実世界データでも成立するか。

データ: tweet(3)/imdb(2)/yelp(5)/emotion(6)/dbpedia(14)（HFミラー, Kaggle系の実テキスト）。
方式: zeroshot(0.5B,0.8B) / dyn_embed(0.8B, LLM+埋込例) / kNN / embed-LR。各 Wilson 95%CI。
"""
from __future__ import annotations

import json
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.experiment import distill_from_train, load_public, make_cache, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, knn_predict, logreg_predict
from cache_harness.task import Task

EMBED = "nomic-embed-text"
SMALL, MID = "qwen2.5:0.5b", "qwen3.5:0.8b"
DEFAULT_TASKS = ["ext_imdb", "ext_tweet", "ext_airline", "ext_yelp", "ext_emotion", "ext_dbpedia"]
TASKS = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TASKS


def build_task(name, split) -> Task:
    labels = tuple(sorted({it.label for it in split.train}))
    instr = "Classify the text into exactly one of these categories: " + ", ".join(labels) + "."
    return Task(name, labels, instr, lang="en")


def ci(pred, gold):
    n = len(gold); c = sum(p == g for p, g in zip(pred, gold))
    return c / n, [metrics._wilson_low(c, n), metrics._wilson_high(c, n)]


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, [SMALL, MID, EMBED])
    rows = []
    print(f"\n{'#'*96}\n# 外部実データ再検証（zeroshot vs 埋込キャッシュ）")
    print(f"  {'task':<14}{'cls':>4}{'zs-0.5B':>9}{'zs-0.8B':>9}{'dynEmb-0.8B':>12}{'kNN':>8}{'embedLR':>9}")
    for name in TASKS:
        split = load_public(name)
        task = build_task(name, split)
        test = list(split.test)[:150]
        gold = [it.label for it in test]
        tr = [it.text for it in split.train]

        S05 = Settings(student_model=SMALL)
        S08 = Settings(student_model=MID, retriever="embedding")
        zs05 = metrics.aggregate("z", run_arm(Classifier(None, S05, client, task, (), ArmConfig()), test, tr, S05)).accuracy
        zs08 = metrics.aggregate("z", run_arm(Classifier(None, S08, client, task, (), ArmConfig()), test, tr, S08)).accuracy
        cache = make_cache(split.train, S08)
        rules, _ = distill_from_train(split.train, S08, client, task)
        dyn = metrics.aggregate("d", run_arm(
            Classifier(cache, S08, client, task, rules, ArmConfig(use_retrieval=True, use_rules=True)), test, tr, S08)).accuracy
        client.save()
        knn_acc, knn_ci = ci(knn_predict(split.train, test, EMBED, k=4), gold)
        lr_acc, lr_ci = ci(logreg_predict(split.train, test, EMBED), gold)

        rows.append({"task": name, "n_classes": len(task.labels), "n_test": len(test),
                     "zs_0.5b": zs05, "zs_0.8b": zs08, "dyn_embed_0.8b": dyn,
                     "knn": knn_acc, "knn_ci": knn_ci, "embed_lr": lr_acc, "embed_lr_ci": lr_ci})
        print(f"  {name:<14}{len(task.labels):>4}{zs05*100:>8.1f}%{zs08*100:>8.1f}%{dyn*100:>11.1f}%"
              f"{knn_acc*100:>7.1f}%{lr_acc*100:>8.1f}%")

    # 既存結果へマージ（タスク名で上書き）。クラス数順に並べ替えて保存。
    from pathlib import Path
    prev = []
    if Path("external_results.json").exists():
        prev = json.loads(Path("external_results.json").read_text(encoding="utf-8"))
    merged = {r["task"]: r for r in prev}
    for r in rows:
        merged[r["task"]] = r
    out = sorted(merged.values(), key=lambda r: (r["n_classes"], r["task"]))
    with open("external_results.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: external_results.json ({len(out)}件) (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
