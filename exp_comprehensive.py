#!/usr/bin/env python3
"""網羅検証 — 「失敗の構造化 > 成功の構造化」が全データセット×全sub-1Bモデルで成立するか。

各 (model, dataset) で LLM経路の4アームを比較:
  zeroshot / fail-rules(失敗の識別的構造化) / success-rules(成功の記述的構造化) / both
ルール執筆は teacher=qwen3.5:9b。失敗ルールはモデル固有(各モデルの誤分類)、成功ルールはデータセット固有(gold per class, モデル非依存)。
参考に cache-LR(gold, LLM不使用) も。結果は逐次保存し、最後に集計(失敗 vs 成功の勝敗・平均Δ)。
"""
from __future__ import annotations

import json
import sys
from collections import Counter

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import LabeledItem, split_same_theme
from cache_harness.experiment import distill_from_train, load_public, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.supervised import accuracy, logreg_predict
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
TEACHER = "qwen3.5:9b"
N_TEST = 50
TRAIN_CAP = 200
TOP = 4
MODELS = ["smollm2:135m", "gemma3:270m", "smollm:360m", "smollm2:360m",
          "qwen2:0.5b", "qwen2.5:0.5b", "qwen3:0.6b", "qwen3.5:0.8b"]
DATASETS = ["jp_sentiment_authored", "sst2", "agnews", "trec", "banking77",
            "ext_imdb", "ext_airline", "ext_tweet", "ext_yelp", "ext_emotion", "ext_dbpedia"]


def build_task(name, split) -> Task:
    if name == "jp_sentiment_authored":
        return SENTIMENT_JP
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))
    return Task(name, labels, "Classify the text into exactly one of: " + ", ".join(labels) + ".", lang="en")


def load(name):
    return split_same_theme(0) if name == "jp_sentiment_authored" else load_public(name)


def _rule(client, prompt):
    r = client.chat(TEACHER, [{"role": "user", "content": prompt}], think=False, num_predict=80)
    return (r.content or "").strip().splitlines()[0].strip() if r.ok and r.content else ""


def fail_rules(train_eval, client):
    by = {}
    for t, g, p in train_eval:
        if p and p != g:
            by.setdefault((g, p), []).append(t)
    top = [k for k, _ in Counter({k: len(v) for k, v in by.items()}).most_common(TOP)]
    out = []
    for (g, p) in top:
        joined = "\n".join(f"- {e[:180]}" for e in by[(g, p)][:4])
        ln = _rule(client, f"Texts truly of class \"{g}\" were misclassified as \"{p}\":\n{joined}\n\n"
                           f"Write ONE concise rule (max 25 words, same language) to choose \"{g}\" over \"{p}\". Only the rule.")
        if len(ln) > 8:
            out.append(f"To choose \"{g}\" over \"{p}\": {ln}")
    return out


def success_rules(train, client):
    by = {}
    for it in train:
        by.setdefault(it.label, []).append(it.text)
    top = [c for c, _ in Counter({c: len(v) for c, v in by.items()}).most_common(TOP)]
    out = []
    for c in top:
        joined = "\n".join(f"- {e[:180]}" for e in by[c][:4])
        ln = _rule(client, f"The following texts all belong to class \"{c}\":\n{joined}\n\n"
                           f"Write ONE concise rule (max 25 words, same language) describing what makes a text \"{c}\". Only the rule.")
        if len(ln) > 8:
            out.append(f"A text is \"{c}\" if: {ln}")
    return out


def acc_arm(clf, test, tr):
    return metrics.aggregate("x", run_arm(clf, test, tr, S_GLOBAL["S"], )).accuracy


S_GLOBAL = {}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, [EMBED, TEACHER])
    avail = {m: warmup(client.inner, [m]).get(m, False) for m in MODELS}
    models = [m for m in MODELS if avail[m]]
    print("models:", models)

    rows = []
    for ds in DATASETS:
        sp = load(ds); task = build_task(ds, sp)
        test = list(sp.test)[:N_TEST]; gold = [it.label for it in test]
        train = list(sp.train)[:TRAIN_CAP]; tr_txt = [it.text for it in train]
        succ_rules = success_rules(train, client)              # データセット固有(モデル非依存)
        client.save()
        cache_lr = accuracy(logreg_predict(list(sp.train), test, EMBED), gold)
        print(f"\n{'#'*96}\n# {ds} ({len(task.labels)}cls)  cache-LR(gold)={cache_lr*100:.1f}%  success_rules={len(succ_rules)}")
        print(f"  {'model':<16}{'zeroshot':>9}{'fail':>8}{'success':>9}{'both':>8}")
        for m in models:
            S = Settings(student_model=m, retriever="embedding"); S_GLOBAL["S"] = S
            f_rules, train_eval = distill_from_train(train, S, client, task)  # 失敗(モデル固有)
            frules = fail_rules(train_eval, client)
            both = list(frules) + list(succ_rules)
            z = acc_arm(Classifier(None, S, client, task, (), ArmConfig()), test, tr_txt)
            fa = acc_arm(Classifier(None, S, client, task, tuple(frules), ArmConfig(use_rules=True)), test, tr_txt) if frules else None
            su = acc_arm(Classifier(None, S, client, task, tuple(succ_rules), ArmConfig(use_rules=True)), test, tr_txt) if succ_rules else None
            bo = acc_arm(Classifier(None, S, client, task, tuple(both), ArmConfig(use_rules=True)), test, tr_txt) if both else None
            rows.append({"dataset": ds, "n_classes": len(task.labels), "model": m, "cache_lr_gold": cache_lr,
                         "zeroshot": z, "fail": fa, "success": su, "both": bo,
                         "n_fail_rules": len(frules), "n_success_rules": len(succ_rules)})
            f = lambda v: f"{v*100:.1f}" if v is not None else "n/a"
            print(f"  {m:<16}{f(z):>9}{f(fa):>8}{f(su):>9}{f(bo):>8}")
            with open("comprehensive_results.json", "w", encoding="utf-8") as fp:
                json.dump(rows, fp, ensure_ascii=False, indent=2)
            client.save()

    # 集計
    def delta(a):
        return [r[a] - r["zeroshot"] for r in rows if r.get(a) is not None and r.get("zeroshot") is not None]
    fd, sd = delta("fail"), delta("success")
    fw = sum(1 for r in rows if r.get("fail") is not None and r.get("success") is not None and r["fail"] > r["success"])
    sw = sum(1 for r in rows if r.get("fail") is not None and r.get("success") is not None and r["success"] > r["fail"])
    tie = sum(1 for r in rows if r.get("fail") is not None and r.get("success") is not None and r["fail"] == r["success"])
    print(f"\n{'='*70}\nSUMMARY ({len(rows)} model×dataset cells)")
    print(f"  mean Δacc vs zeroshot:  fail-rules {sum(fd)/len(fd)*100:+.2f}pp | success-rules {sum(sd)/len(sd)*100:+.2f}pp")
    print(f"  head-to-head (fail vs success): fail wins {fw}, success wins {sw}, tie {tie}")
    print("保存: comprehensive_results.json")


if __name__ == "__main__":
    main()
