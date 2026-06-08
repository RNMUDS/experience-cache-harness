#!/usr/bin/env python3
"""E2(改訂): マルチタスク — 検索器(文字n-gram vs 埋め込み)とルールの寄与を分離し、
dev で最良構成を選ぶ「AutoCache」が zeroshot を確実に上回ることを示す。

各タスクで test 報告: zeroshot / dyn_char / dyn_embed / rules_only / cache_embed(埋込+ルール)
/ escalation / nn_copy。AutoCache = dev macro-F1 最良の構成の test 値。
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
from cache_harness.dataset import Split, split_same_theme
from cache_harness.experiment import (distill_from_train, load_public, make_cache,
                                      make_client, run_arm)
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

MODEL = "qwen3.5:0.8b"
N_TRAIN, N_TEST = 150, 120
S = Settings(student_model=MODEL)
S_EMB = Settings(student_model=MODEL, retriever="embedding")


def load_task(task: Task) -> Split:
    if task.name == "jp_sentiment_authored":
        return split_same_theme(0)
    sp = load_public(task.name)
    return Split(sp.train[:N_TRAIN], sp.dev, sp.test[:N_TEST], "public")


def run_task(task: Task, client) -> dict:
    split = load_task(task)
    tr = [it.text for it in split.train]
    char_cache = make_cache(split.train, S)
    embed_cache = make_cache(split.train, S_EMB)
    distilled, _ = distill_from_train(split.train, S, client, task)

    # (name, cache, ArmConfig, rules, is_candidate_for_autocache)
    configs = [
        ("zeroshot", None, ArmConfig(), (), True),
        ("dyn_char", char_cache, ArmConfig(use_retrieval=True), (), True),
        ("dyn_embed", embed_cache, ArmConfig(use_retrieval=True), (), True),
        ("rules_only", None, ArmConfig(use_rules=True), distilled, True),
        ("cache_embed", embed_cache, ArmConfig(use_retrieval=True, use_rules=True), distilled, True),
        ("escalation", embed_cache, ArmConfig(use_retrieval=True, use_rules=True,
                                              use_short_circuit=True, use_escalation=True), distilled, False),
        ("nn_copy", embed_cache, ArmConfig(mode="nn_copy", use_retrieval=True), (), False),
    ]

    def evaluate(cache, arm, rules, items):
        clf = Classifier(cache=cache, settings=(S_EMB if cache is embed_cache else S),
                         client=client, task=task, rules=rules, arm=arm)
        return run_arm(clf, items, tr, (S_EMB if cache is embed_cache else S))

    test_outcomes, test_reports, dev_mf1 = {}, {}, {}
    for name, cache, arm, rules, is_cand in configs:
        test_outcomes[name] = evaluate(cache, arm, rules, split.test)
        test_reports[name] = metrics.aggregate(name, test_outcomes[name])
        if is_cand:
            dev_mf1[name] = metrics.aggregate(name, evaluate(cache, arm, rules, split.dev)).macro_f1

    best = max(dev_mf1, key=lambda k: dev_mf1[k])

    print(f"\n{'#'*88}\n# タスク: {task.name} ({len(task.labels)}クラス,{task.lang}) "
          f"train={len(split.train)} test={len(split.test)}  | AutoCache(dev最良)= {best}")
    print(f"# 蒸留ルール: " + (" / ".join(distilled)[:180] if distilled else "なし"))
    print("\n" + metrics.header())
    for name in ("zeroshot", "dyn_char", "dyn_embed", "rules_only", "cache_embed", "escalation", "nn_copy"):
        print(test_reports[name].summary_line())

    base = test_outcomes["zeroshot"]
    print("\n  [McNemar 片側 vs zeroshot]")
    mcnemar = {}
    for name in ("dyn_char", "dyn_embed", "cache_embed", best):
        m = metrics.mcnemar_better(base, test_outcomes[name])
        mcnemar[name] = m
        print(f"    {name:<14} b={m['b_base_only']:2d} c={m['c_variant_only']:2d} p={m['p_one_sided']:.4f}")
    print(f"  → AutoCache={best}: test acc={test_reports[best].accuracy*100:.1f}% "
          f"(zeroshot={test_reports['zeroshot'].accuracy*100:.1f}%)")

    return {"task": task.name, "n_classes": len(task.labels), "lang": task.lang, "best_config": best,
            "dev_macro_f1": dev_mf1, "distilled_rules": list(distilled), "mcnemar_vs_zeroshot": mcnemar,
            "test": {n: {"acc": r.accuracy, "acc_ci": r.acc_ci, "macro_f1": r.macro_f1,
                         "total_llm_calls": r.total_llm_calls, "route_counts": r.route_counts}
                     for n, r in test_reports.items()}}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL, S.teacher_model, S.embed_model]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)
    results = []
    for task in [SENTIMENT_JP, TASKS["sst2"], TASKS["agnews"], TASKS["trec"]]:
        results.append(run_task(task, client))
        client.save()
    with open("results/main_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: main_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
