#!/usr/bin/env python3
"""E2: マルチタスク評価 — 経験キャッシュが複数タスク/言語/クラス数で素モデルを上回るか。

タスク: 自作JP感情(3) / SST-2(2,英) / AG News(4,英) / TREC(6,英)。モデル: qwen3.5:0.8b。
アーム: zeroshot / static / dynamic / rules-only / cache(dyn+rules) / escalation / nn-copy。
"""
from __future__ import annotations

import json
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import LabeledItem, Split, split_same_theme
from cache_harness.experiment import (distill_from_train, load_public, make_cache,
                                      make_client, run_arm, static_sample)
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

MODEL = "qwen3.5:0.8b"
N_TRAIN, N_TEST = 150, 120


def _cap(split: Split, n_train: int, n_test: int) -> Split:
    return Split(split.train[:n_train], split.dev, split.test[:n_test], split.kind)


def load_task(task: Task) -> Split:
    if task.name == "jp_sentiment_authored":
        return split_same_theme(0)
    return _cap(load_public(task.name), N_TRAIN, N_TEST)


def arms(distilled, static_ex):
    return [
        ("zeroshot", ArmConfig(), ()),
        ("static-kshot", ArmConfig(static_exemplars=static_ex), ()),
        ("dynamic-kshot", ArmConfig(use_retrieval=True), ()),
        ("rules-only", ArmConfig(use_rules=True), distilled),
        ("cache(dyn+rules)", ArmConfig(use_retrieval=True, use_rules=True), distilled),
        ("escalation", ArmConfig(use_retrieval=True, use_rules=True, use_short_circuit=True, use_escalation=True), distilled),
        ("nn-copy", ArmConfig(mode="nn_copy", use_retrieval=True), ()),
    ]


def run_task(task: Task, client) -> dict:
    settings = Settings(student_model=MODEL)
    split = load_task(task)
    train_texts = [it.text for it in split.train]
    cache = make_cache(split.train, settings)
    distilled, _ = distill_from_train(split.train, settings, client, task)
    static_ex = static_sample(split.train, settings.k, 0, task.labels)

    print(f"\n{'#'*88}\n# タスク: {task.name}  ({len(task.labels)}クラス, {task.lang})  "
          f"train={len(split.train)} test={len(split.test)}")
    print(f"# 蒸留ルール({len(distilled)}件): " + (" / ".join(distilled) if distilled else "なし")[:200])
    print("\n" + metrics.header())

    reports, outcomes = {}, {}
    for name, arm, rules in arms(distilled, static_ex):
        clf = Classifier(cache=cache, settings=settings, client=client, task=task, rules=rules, arm=arm)
        outs = run_arm(clf, split.test, train_texts, settings)
        reports[name] = metrics.aggregate(name, outs)
        outcomes[name] = outs
        print(reports[name].summary_line())

    base = outcomes["zeroshot"]
    print("\n  [McNemar 片側 vs zeroshot]")
    for name in ("dynamic-kshot", "rules-only", "cache(dyn+rules)", "escalation"):
        m = metrics.mcnemar_better(base, outcomes[name])
        print(f"    {name:<18} b={m['b_base_only']:2d} c={m['c_variant_only']:2d} p={m['p_one_sided']:.4f}")

    return {"task": task.name, "n_classes": len(task.labels), "lang": task.lang,
            "n_test": len(split.test), "distilled_rules": list(distilled),
            "reports": {n: {"acc": r.accuracy, "acc_ci": r.acc_ci, "macro_f1": r.macro_f1,
                            "total_llm_calls": r.total_llm_calls, "route_counts": r.route_counts}
                        for n, r in reports.items()}}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL, Settings().teacher_model]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)
    tasks = [SENTIMENT_JP, TASKS["sst2"], TASKS["agnews"], TASKS["trec"]]
    results = []
    for task in tasks:
        results.append(run_task(task, client))
        client.save()
    with open("multitask_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: multitask_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
