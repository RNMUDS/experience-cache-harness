#!/usr/bin/env python3
"""成功体験の構造化 — 正しく分類できた例から teacher が「クラスXとは何か」を1文に抽象化。

失敗の構造化(混同ペア→矯正ルール)の成功側アナログ。比較: zeroshot / success-rules / fail-rules / both。
(自己蒸留は不採用。執筆は teacher=qwen3.5:9b。)
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys
from collections import Counter

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import split_same_theme
from cache_harness.experiment import distill_from_train, load_public, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

MODEL = "qwen3.5:0.8b"
TEACHER = "qwen3.5:9b"
N_TEST = 100
TOP_PAIRS = 6
TOP_CLASSES = 6
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


def fail_rules(train_eval, client) -> list[str]:
    fails = [(t, g, p) for (t, g, p) in train_eval if p and p != g]
    by: dict[tuple[str, str], list[str]] = {}
    for t, g, p in fails:
        by.setdefault((g, p), []).append(t)
    top = [k for k, _ in Counter({k: len(v) for k, v in by.items()}).most_common(TOP_PAIRS)]
    out = []
    for (g, p) in top:
        joined = "\n".join(f"- {e[:200]}" for e in by[(g, p)][:4])
        pr = (f"A text classifier confuses two classes. The following texts truly belong to class \"{g}\" "
              f"but were misclassified as \"{p}\".\n{joined}\n\nWrite ONE concise rule (max 25 words, same "
              f"language as the texts) that helps decide \"{g}\" rather than \"{p}\". Output only the rule, no preamble.")
        r = client.chat(TEACHER, [{"role": "user", "content": pr}], think=False, num_predict=80)
        ln = (r.content or "").strip().splitlines()[0].strip() if r.ok and r.content else ""
        if len(ln) > 8:
            out.append(f"To choose \"{g}\" over \"{p}\": {ln}")
    return out


def success_rules(train_eval, client) -> list[str]:
    succ: dict[str, list[str]] = {}
    for t, g, p in train_eval:
        if p == g:
            succ.setdefault(g, []).append(t)
    top = [c for c, _ in Counter({c: len(v) for c, v in succ.items()}).most_common(TOP_CLASSES)]
    out = []
    for c in top:
        joined = "\n".join(f"- {e[:200]}" for e in succ[c][:4])
        pr = (f"The following texts all belong to the class \"{c}\".\n{joined}\n\nWrite ONE concise rule "
              f"(max 25 words, same language as the texts) describing what makes a text belong to \"{c}\". "
              f"Output only the rule, no preamble.")
        r = client.chat(TEACHER, [{"role": "user", "content": pr}], think=False, num_predict=80)
        ln = (r.content or "").strip().splitlines()[0].strip() if r.ok and r.content else ""
        if len(ln) > 8:
            out.append(f"A text is \"{c}\" if: {ln}")
    return out


def run(clf, test, tr):
    return metrics.aggregate("x", run_arm(clf, test, tr, S)).accuracy


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not (warmup(client.inner, [MODEL]).get(MODEL) and warmup(client.inner, [TEACHER]).get(TEACHER)):
        print("接続エラー", file=sys.stderr); sys.exit(1)

    rows = []
    for name in TASK_NAMES:
        sp = load(name); task = build_task(name, sp)
        test = list(sp.test)[:N_TEST]
        _lex, train_eval = distill_from_train(sp.train, S, client, task)
        client.save()
        fr = fail_rules(train_eval, client)        # 失敗の構造化(cache再利用)
        sr = success_rules(train_eval, client)     # 成功の構造化(新規)
        both = list(fr) + list(sr)
        client.save()
        tr_txt = [it.text for it in sp.train]
        arms = {
            "zeroshot":      run(Classifier(None, S, client, task, (), ArmConfig()), test, tr_txt),
            "fail-rules":    run(Classifier(None, S, client, task, tuple(fr), ArmConfig(use_rules=True)), test, tr_txt) if fr else None,
            "success-rules": run(Classifier(None, S, client, task, tuple(sr), ArmConfig(use_rules=True)), test, tr_txt) if sr else None,
            "both-rules":    run(Classifier(None, S, client, task, tuple(both), ArmConfig(use_rules=True)), test, tr_txt) if both else None,
        }
        rec = {"task": name, "n_classes": len(task.labels), "n_fail_rules": len(fr),
               "n_success_rules": len(sr), "success_rules": sr, "arms": arms}
        rows.append(rec)
        print(f"\n{'#'*92}\n# {name} ({len(task.labels)}cls)  fail_rules={len(fr)} success_rules={len(sr)}")
        for k, v in arms.items():
            print(f"  {k:<16}{'n/a' if v is None else f'{v*100:.1f}%':>8}")
        print("  success knowledge (teacher):")
        for r in sr:
            print(f"    - {r[:108]}")
        client.save()

    with open("results/success_rules_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: success_rules_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
