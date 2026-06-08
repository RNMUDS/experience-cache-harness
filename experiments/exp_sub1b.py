#!/usr/bin/env python3
"""sub-1B 横断検証 — 「10億未満なら“どのモデルでも”キャッシュで使えるか」を多様な極小モデルで確認。

各 sub-1B モデルで zero-shot と cache(埋め込み exemplar) を測り、モデル非依存の floor(embed-LR/kNN, LLM不使用)と比較。
仮説: zero-shot は機種で激しくばらつくが、(i) cache-exemplar は弱い機種ほど底上げし、
(ii) 非パラメトリック floor は機種に依らず一定 → 「どの sub-1B でもキャッシュで一定の答えに到達」。
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
from cache_harness.experiment import distill_from_train, load_public, make_cache, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, knn_predict, logreg_predict
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
N_TEST = 100
# (ollama名, 表示名, おおよそのB)
CANDIDATES = [
    ("smollm2:135m", "SmolLM2-135M", 0.135),
    ("gemma3:270m", "Gemma3-270M", 0.27),
    ("smollm:360m", "SmolLM-360M", 0.36),
    ("smollm2:360m", "SmolLM2-360M", 0.36),
    ("qwen2:0.5b", "Qwen2-0.5B", 0.49),
    ("qwen2.5:0.5b", "Qwen2.5-0.5B", 0.5),
    ("qwen3:0.6b", "Qwen3-0.6B", 0.6),
    ("qwen3.5:0.8b", "Qwen3.5-0.8B", 0.8),
]
TASK_NAMES = ["jp_sentiment_authored", "agnews", "banking77"]


def build_task(name, split) -> Task:
    if name == "jp_sentiment_authored":
        return SENTIMENT_JP
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))
    return Task(name, labels, "Classify the customer query into one of: " + ", ".join(labels) + ".", lang="en")


def load(name):
    return split_same_theme(0) if name == "jp_sentiment_authored" else load_public(name)


def avail(client, model) -> bool:
    return warmup(client.inner, [model]).get(model, False)


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, [EMBED])
    models = [(m, n, b) for (m, n, b) in CANDIDATES if avail(client, m)]
    print("利用可能 sub-1B:", [n for _, n, _ in models])

    # タスクごとに floor（モデル非依存）を一度だけ
    floors, splits, tasks = {}, {}, {}
    for tn in TASK_NAMES:
        sp = load(tn); tk = build_task(tn, sp)
        splits[tn], tasks[tn] = sp, tk
        te = list(sp.test)[:N_TEST]; gold = [it.label for it in te]
        floors[tn] = {
            "n_classes": len(tk.labels),
            "embed_lr": accuracy(logreg_predict(sp.train, te, EMBED), gold),
            "knn": accuracy(knn_predict(sp.train, te, EMBED, k=4), gold),
        }

    rows = []
    print(f"\n{'#'*92}\n# sub-1B 横断: zero-shot / cache(emb exemplar) （floor=embed-LR はモデル非依存）")
    hdr = f"  {'model':<16}{'B':>5}"
    for tn in TASK_NAMES:
        hdr += f"{tn[:8]+'(zs/ca)':>16}"
    print(hdr)
    for m, name, b in models:
        S = Settings(student_model=m, retriever="embedding")
        rec = {"model": name, "params_b": b, "per_task": {}}
        cells = ""
        for tn in TASK_NAMES:
            sp, tk = splits[tn], tasks[tn]
            te = list(sp.test)[:N_TEST]; tr = [it.text for it in sp.train]
            cache = make_cache(sp.train, S)
            zs = metrics.aggregate("z", run_arm(Classifier(None, S, client, tk, (), ArmConfig()), te, tr, S)).accuracy
            ca = metrics.aggregate("c", run_arm(Classifier(cache, S, client, tk, (), ArmConfig(use_retrieval=True)), te, tr, S)).accuracy
            rec["per_task"][tn] = {"zeroshot": zs, "cache_exemplar": ca}
            cells += f"{f'{zs*100:.0f}/{ca*100:.0f}':>16}"
        rows.append(rec)
        print(f"  {name:<16}{b:>5}{cells}")
        client.save()

    print(f"\n  floor (LLM不使用, 全モデル共通):")
    for tn in TASK_NAMES:
        f = floors[tn]
        print(f"    {tn:<22}({f['n_classes']}cls)  embed-LR={f['embed_lr']*100:.0f}%  kNN={f['knn']*100:.0f}%")

    with open("results/sub1b_results.json", "w", encoding="utf-8") as fp:
        json.dump({"models": rows, "floors": floors, "tasks": TASK_NAMES}, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: sub1b_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
