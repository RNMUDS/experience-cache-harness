#!/usr/bin/env python3
"""fail_v2-self — 意味ルールを teacher でなく 0.8B 自身に書かせる自己蒸留版。

問い: 失敗するモデル自身が、自分の失敗から矯正知識(NLルール)を言語化できるか？(教師なしで成立するか)
比較(全て適用は 0.8B): zeroshot / fail_v2-self(0.8Bがルール執筆) / fail_v2-teacher(9bが執筆, 参照=cache) / fail_v2-self+gold
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys
from collections import Counter

from cache_harness import metrics
from cache_harness.cache import Experience, ExperienceCache
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import LabeledItem, split_same_theme
from cache_harness.experiment import distill_from_train, load_public, make_client, make_retriever, run_arm
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
MODEL = "qwen3.5:0.8b"
TEACHER = "qwen3.5:9b"
N_TEST = 100
TOP_PAIRS = 6
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


def semantic_rules(train_eval, writer, client) -> list[str]:
    """混同ペアごとに writer モデルが1文ルールを書く。writer=MODEL なら自己蒸留。"""
    fails = [(t, g, p) for (t, g, p) in train_eval if p and p != g]
    by_pair: dict[tuple[str, str], list[str]] = {}
    for t, g, p in fails:
        by_pair.setdefault((g, p), []).append(t)
    top = [pair for pair, _ in Counter({k: len(v) for k, v in by_pair.items()}).most_common(TOP_PAIRS)]
    rules: list[str] = []
    for (g, p) in top:
        joined = "\n".join(f"- {e[:200]}" for e in by_pair[(g, p)][:4])
        prompt = (f"A text classifier confuses two classes. The following texts truly belong to class "
                  f"\"{g}\" but were misclassified as \"{p}\".\n{joined}\n\n"
                  f"Write ONE concise rule (max 25 words, same language as the texts) that helps decide "
                  f"\"{g}\" rather than \"{p}\". Output only the rule, no preamble.")
        r = client.chat(writer, [{"role": "user", "content": prompt}], think=False, num_predict=80)
        line = (r.content or "").strip().splitlines()[0].strip() if r.ok and r.content else ""
        if line and len(line) > 8:
            rules.append(f"To choose \"{g}\" over \"{p}\": {line}")
    return rules


def cache_of(pairs):
    c = ExperienceCache(retriever=make_retriever(S))
    c.add_many([Experience(t, l, theme=th) for (t, l, th) in pairs])
    return c


def run(clf, test, tr):
    return metrics.aggregate("x", run_arm(clf, test, tr, S)).accuracy


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not (warmup(client.inner, [MODEL, EMBED]).get(MODEL) and warmup(client.inner, [TEACHER]).get(TEACHER)):
        print("接続エラー", file=sys.stderr); sys.exit(1)

    rows = []
    for name in TASK_NAMES:
        sp = load(name); task = build_task(name, sp)
        test = list(sp.test)[:N_TEST]
        _lex, train_eval = distill_from_train(sp.train, S, client, task)
        client.save()
        self_rules = semantic_rules(train_eval, MODEL, client)       # 自己蒸留(新規)
        teach_rules = semantic_rules(train_eval, TEACHER, client)    # 参照(同一prompt→cache)
        client.save()
        theme = {it.text: it.theme for it in sp.train}
        gold_pairs = [(t, g, theme.get(t, name)) for (t, g, p) in train_eval]
        gold_cache = cache_of(gold_pairs); tr_gold = [p[0] for p in gold_pairs]

        arms = {
            "zeroshot":         run(Classifier(None, S, client, task, (), ArmConfig()), test, tr_gold),
            "fail_v2-self":     run(Classifier(None, S, client, task, tuple(self_rules), ArmConfig(use_rules=True)), test, tr_gold) if self_rules else None,
            "fail_v2-teacher":  run(Classifier(None, S, client, task, tuple(teach_rules), ArmConfig(use_rules=True)), test, tr_gold) if teach_rules else None,
            "fail_v2-self+gold": run(Classifier(gold_cache, S, client, task, tuple(self_rules), ArmConfig(use_retrieval=True, use_rules=True)), test, tr_gold) if self_rules else None,
        }
        rec = {"task": name, "n_classes": len(task.labels), "n_self_rules": len(self_rules),
               "n_teacher_rules": len(teach_rules), "self_rules": self_rules, "arms": arms}
        rows.append(rec)
        print(f"\n{'#'*92}\n# {name} ({len(task.labels)}cls) self_rules={len(self_rules)} teacher_rules={len(teach_rules)}")
        for k, v in arms.items():
            print(f"  {k:<20}{'n/a' if v is None else f'{v*100:.1f}%':>8}")
        print("  self-written knowledge (0.8B):")
        for r in self_rules:
            print(f"    - {r[:110]}")
        client.save()

    with open("results/failv2_self_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: failv2_self_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
