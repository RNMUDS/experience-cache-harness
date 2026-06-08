#!/usr/bin/env python3
"""fail_v2-semantic — 失敗を「意味的に体系化した知識」に蒸留して蓄積（語彙キュー版の上位互換）。

失敗を混同ペア(gold→pred)ごとに束ね、各ペアを有能なteacher(qwen3.5:9b)が
1文の自然言語ルールへ抽象化＝「構造化知識」。それを 0.8B のプロンプトに注入して比較:
  zeroshot / failure-exemplars(個別) / fail_v2-lexical(rules.py) / fail_v2-semantic(teacher NLルール) / +gold-ex
"""
from __future__ import annotations

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
from cache_harness.supervised import accuracy, logreg_predict
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


def semantic_rules(train_eval, task, client) -> list[str]:
    """混同ペアごとに teacher が1文ルールを書く＝意味的に構造化した知識。"""
    fails = [(t, g, p) for (t, g, p) in train_eval if p and p != g]
    by_pair: dict[tuple[str, str], list[str]] = {}
    for t, g, p in fails:
        by_pair.setdefault((g, p), []).append(t)
    top = [pair for pair, _ in Counter({k: len(v) for k, v in by_pair.items()}).most_common(TOP_PAIRS)]
    rules: list[str] = []
    for (g, p) in top:
        ex = by_pair[(g, p)][:4]
        joined = "\n".join(f"- {e[:200]}" for e in ex)
        prompt = (f"A text classifier confuses two classes. The following texts truly belong to class "
                  f"\"{g}\" but were misclassified as \"{p}\".\n{joined}\n\n"
                  f"Write ONE concise rule (max 25 words, same language as the texts) that helps decide "
                  f"\"{g}\" rather than \"{p}\". Output only the rule, no preamble.")
        r = client.chat(TEACHER, [{"role": "user", "content": prompt}], think=False, num_predict=80)
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
    if not (warmup(client.inner, [MODEL, EMBED, TEACHER]).get(MODEL) and warmup(client.inner, [TEACHER]).get(TEACHER)):
        print("接続エラー (teacher 含む)", file=sys.stderr); sys.exit(1)

    rows = []
    for name in TASK_NAMES:
        sp = load(name); task = build_task(name, sp)
        test = list(sp.test)[:N_TEST]; gold = [it.label for it in test]
        lex_rules, train_eval = distill_from_train(sp.train, S, client, task)
        client.save()
        sem_rules = semantic_rules(train_eval, task, client)
        client.save()
        theme = {it.text: it.theme for it in sp.train}
        gold_pairs = [(t, g, theme.get(t, name)) for (t, g, p) in train_eval]
        fail_pairs = [(t, g, theme.get(t, name)) for (t, g, p) in train_eval if p and p != g]
        gold_cache, fail_cache = cache_of(gold_pairs), cache_of(fail_pairs)
        tr_gold = [p[0] for p in gold_pairs]; tr_fail = [p[0] for p in fail_pairs]

        arms = {
            "zeroshot":          run(Classifier(None, S, client, task, (), ArmConfig()), test, tr_gold),
            "failure-exemplars": run(Classifier(fail_cache, S, client, task, (), ArmConfig(use_retrieval=True)), test, tr_fail) if fail_pairs else None,
            "fail_v2-lexical":   run(Classifier(None, S, client, task, tuple(lex_rules), ArmConfig(use_rules=True)), test, tr_gold) if lex_rules else None,
            "fail_v2-semantic":  run(Classifier(None, S, client, task, tuple(sem_rules), ArmConfig(use_rules=True)), test, tr_gold) if sem_rules else None,
            "fail_v2-sem+gold":  run(Classifier(gold_cache, S, client, task, tuple(sem_rules), ArmConfig(use_retrieval=True, use_rules=True)), test, tr_gold) if sem_rules else None,
        }
        cache_lr = accuracy(logreg_predict([LabeledItem(*p) for p in gold_pairs], test, EMBED), gold)
        rec = {"task": name, "n_classes": len(task.labels), "n_fail": len(fail_pairs),
               "n_lex_rules": len(lex_rules), "n_sem_rules": len(sem_rules),
               "sem_rules": sem_rules, "arms": arms, "cache_lr_gold": cache_lr}
        rows.append(rec)
        print(f"\n{'#'*92}\n# {name} ({len(task.labels)}cls) fail={len(fail_pairs)} "
              f"lex_rules={len(lex_rules)} sem_rules={len(sem_rules)}")
        for k, v in arms.items():
            print(f"  {k:<20}{'n/a' if v is None else f'{v*100:.1f}%':>8}")
        print(f"  {'[ref] cache-LR gold':<20}{cache_lr*100:>7.1f}%")
        print("  semantic knowledge:")
        for r in sem_rules:
            print(f"    - {r[:110]}")
        client.save()

    with open("failv2_semantic_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: failv2_semantic_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
