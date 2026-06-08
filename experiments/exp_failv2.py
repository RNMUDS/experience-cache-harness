#!/usr/bin/env python3
"""fail_v2 — 失敗を個別記憶するのではなく、構造化知識(蒸留ルール)として蓄積する版を比較。

LLM経路(qwen3.5:0.8b)で比較するアーム:
  zeroshot          : 素のzero-shot
  gold-exemplars    : 検証済み正解を検索exemplarで提示
  failure-exemplars : 失敗事例(個別)を正解付きで検索提示  ← 前回の「失敗経験」
  fail_v2-rules     : 失敗から蒸留した構造化ルールのみ(個別事例なし) ← 今回
  fail_v2 + gold-ex : 構造化ルール + 正解exemplar(知識と例の併用)
参考: cache-LR(gold) 非パラメトリック上限。
知識のコンパクトさ(ルール数・文字数 vs 失敗事例数)も記録。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

from cache_harness import metrics
from cache_harness.cache import Experience, ExperienceCache
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import LabeledItem, split_same_theme
from cache_harness.experiment import distill_from_train, load_public, make_client, make_retriever, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, logreg_predict
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


def cache_of(pairs):
    c = ExperienceCache(retriever=make_retriever(S))
    c.add_many([Experience(t, l, theme=th) for (t, l, th) in pairs])
    return c


def run(clf, test, tr):
    return metrics.aggregate("x", run_arm(clf, test, tr, S)).accuracy


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL, EMBED]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)

    rows = []
    for name in TASK_NAMES:
        sp = load(name); task = build_task(name, sp)
        test = list(sp.test)[:N_TEST]; gold = [it.label for it in test]
        rules, train_eval = distill_from_train(sp.train, S, client, task)  # rules = 失敗の構造化知識
        client.save()
        theme = {it.text: it.theme for it in sp.train}
        gold_pairs = [(t, g, theme.get(t, name)) for (t, g, p) in train_eval]
        fail_pairs = [(t, g, theme.get(t, name)) for (t, g, p) in train_eval if p and p != g]
        gold_cache, fail_cache = cache_of(gold_pairs), cache_of(fail_pairs)
        tr_gold = [p[0] for p in gold_pairs]; tr_fail = [p[0] for p in fail_pairs]

        arms = {
            "zeroshot":          run(Classifier(None, S, client, task, (), ArmConfig()), test, tr_gold),
            "gold-exemplars":    run(Classifier(gold_cache, S, client, task, (), ArmConfig(use_retrieval=True)), test, tr_gold),
            "failure-exemplars": run(Classifier(fail_cache, S, client, task, (), ArmConfig(use_retrieval=True)), test, tr_fail) if fail_pairs else None,
            "fail_v2-rules":     run(Classifier(None, S, client, task, tuple(rules), ArmConfig(use_rules=True)), test, tr_gold) if rules else None,
            "fail_v2+gold-ex":   run(Classifier(gold_cache, S, client, task, tuple(rules), ArmConfig(use_retrieval=True, use_rules=True)), test, tr_gold) if rules else None,
        }
        cache_lr_gold = accuracy(logreg_predict([LabeledItem(*p) for p in gold_pairs], test, EMBED), gold)
        knowledge_chars = sum(len(r) for r in rules)
        rec = {"task": name, "n_classes": len(task.labels), "n_fail_cases": len(fail_pairs),
               "n_rules": len(rules), "knowledge_chars": knowledge_chars, "rules": list(rules),
               "arms": arms, "cache_lr_gold": cache_lr_gold}
        rows.append(rec)
        print(f"\n{'#'*92}\n# {name} ({len(task.labels)}cls)  failure-cases={len(fail_pairs)}  "
              f"rules={len(rules)} ({knowledge_chars} chars)")
        for k, v in arms.items():
            print(f"  {k:<20}{'n/a' if v is None else f'{v*100:.1f}%':>8}")
        print(f"  {'[ref] cache-LR gold':<20}{cache_lr_gold*100:>7.1f}%")
        if rules:
            print("  distilled knowledge:")
            for r in rules[:6]:
                print(f"    - {r[:96]}")
        client.save()

    with open("results/failv2_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: failv2_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
