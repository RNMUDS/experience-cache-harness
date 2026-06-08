#!/usr/bin/env python3
"""E1: スケーリング則 — 経験キャッシュの効果はモデルが小さいほど大きいか。

モデル軸（Qwen2.5 0.5/1.5/3B の同系統ラダー + 他系統で頑健性）× タスク（自作JP感情, SST-2）。
アーム: zeroshot / rules-only / cache(dyn+rules)。指標: gain = cache - zeroshot。
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
from cache_harness.experiment import distill_from_train, make_cache, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

from exp_multitask import load_task  # 同じ分割ロジックを再利用

# (model, おおよそのパラメータ数B, 系統)
MODELS = [
    ("qwen2.5:0.5b", 0.5, "qwen2.5"),
    ("qwen3.5:0.8b", 0.8, "qwen3.5"),
    ("llama3.2:1b", 1.2, "llama3.2"),
    ("qwen2.5:1.5b", 1.5, "qwen2.5"),
    ("gemma2:2b", 2.6, "gemma2"),
    ("qwen2.5:3b", 3.1, "qwen2.5"),
    ("llama3.2:3b", 3.2, "llama3.2"),
    ("qwen3.5:9b", 9.7, "qwen3.5"),
    ("gemma4:latest", 8.0, "gemma"),
    ("qwen3.6:latest", 36.0, "qwen3.6"),
]
TASK_MODELS = {
    "jp_sentiment_authored": [m[0] for m in MODELS],                 # 全系統
    "sst2": ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],          # Qwen2.5 ラダー
}


def run_model_task(model: str, task: Task, client) -> dict:
    settings = Settings(student_model=model, retriever="embedding")  # 最良構成（埋め込み検索）
    split = load_task(task)
    train_texts = [it.text for it in split.train]
    cache = make_cache(split.train, settings)  # 埋め込みキャッシュ
    distilled, _ = distill_from_train(split.train, settings, client, task)

    # クリーンなキャッシュ構成: 埋め込み exemplar（ノイズなルールは除外）と kNN（LLM不使用）
    out = {}
    for name, arm in [("zeroshot", ArmConfig()),
                      ("dyn_embed", ArmConfig(use_retrieval=True)),
                      ("knn", ArmConfig(mode="nn_copy", use_retrieval=True))]:
        use_cache = None if name == "zeroshot" else cache
        clf = Classifier(cache=use_cache, settings=settings, client=client, task=task, rules=(), arm=arm)
        out[name] = metrics.aggregate(name, run_arm(clf, split.test, train_texts, settings))
    best = max(out["dyn_embed"].accuracy, out["knn"].accuracy)  # 利用可能なキャッシュ最良
    return {"model": model, "task": task.name,
            "zeroshot": out["zeroshot"].accuracy, "dyn_embed": out["dyn_embed"].accuracy,
            "knn": out["knn"].accuracy, "cache": best, "gain": best - out["zeroshot"].accuracy,
            "zeroshot_mf1": out["zeroshot"].macro_f1, "cache_mf1": out["dyn_embed"].macro_f1}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    params = {m[0]: (m[1], m[2]) for m in MODELS}
    tasks = {"jp_sentiment_authored": SENTIMENT_JP, "sst2": TASKS["sst2"]}

    rows = []
    for task_name, models in TASK_MODELS.items():
        task = tasks[task_name]
        print(f"\n{'#'*80}\n# スケーリング: {task_name}")
        print(f"  {'model':<16}{'B':>5}{'zeroshot':>10}{'dynEmb':>8}{'kNN':>8}{'gain':>8}")
        for model in models:
            if not warmup(client.inner, [model]).get(model):
                print(f"  {model}: 接続エラー（スキップ）"); continue
            r = run_model_task(model, task, client)
            r["params_b"], r["family"] = params[model]
            rows.append(r)
            print(f"  {model:<16}{params[model][0]:>5}{r['zeroshot']*100:9.1f}%"
                  f"{r['dyn_embed']*100:7.1f}%{r['knn']*100:7.1f}%{r['gain']*100:+7.1f}")
            client.save()

    with open("results/scaling_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: scaling_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
