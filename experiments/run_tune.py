#!/usr/bin/env python3
"""組み合わせ最適化 — ルール + 類似度ゲート付き exemplar を **dev でのみ**チューニングし test で確認。

ablation の知見（インサイトが主役、exemplar は 0.8B にノイズになり得る）を受け、
「relevant な時だけ exemplar を足す」ゲートが rules-only を超えられるかを検証。
HP リーク防止: 選択は dev の macro-F1 のみ。test は最後に 1 回。
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
from cache_harness.experiment import distill_from_train, make_cache, make_client, run_arm
from cache_harness.llm import warmup

BASE = Settings()

# (名前, k, min_exemplar_sim, use_retrieval)
CONFIGS = [
    ("rules-only(k0)", 0, 0.0, False),
    ("rules+ret k4 g0.00", 4, 0.00, True),
    ("rules+ret k4 g0.15", 4, 0.15, True),
    ("rules+ret k4 g0.20", 4, 0.20, True),
    ("rules+ret k2 g0.00", 2, 0.00, True),
    ("rules+ret k2 g0.15", 2, 0.15, True),
]


def eval_config(name, k, gate, use_ret, cache, distilled, items, train_texts, client):
    settings = BASE.derive(k=k, min_exemplar_sim=gate)
    arm = ArmConfig(use_retrieval=use_ret, use_rules=True)
    clf = Classifier(cache=cache, settings=settings, client=client, rules=distilled, arm=arm)
    return metrics.aggregate(name, run_arm(clf, items, train_texts, settings))


def main() -> None:
    client = make_client(path="llm_cache.json")
    client.load()
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [BASE.student_model]).get(BASE.student_model):
        print("接続エラー", file=sys.stderr)
        sys.exit(1)

    split = split_same_theme(0)
    train_texts = [it.text for it in split.train]
    cache = make_cache(split.train, BASE)
    distilled, _ = distill_from_train(split.train, BASE, client)

    print(f"\n=== dev チューニング (dev n={len(split.dev)}) ===")
    print(f"  {'config':<22}{'acc':>7}{'macroF1':>9}{'neu_R':>7}{'neu_P':>7}")
    dev_reports = {}
    for name, k, gate, use_ret in CONFIGS:
        r = eval_config(name, k, gate, use_ret, cache, distilled, split.dev, train_texts, client)
        dev_reports[name] = r
        print(f"  {name:<22}{r.accuracy*100:6.1f}%{r.macro_f1*100:8.1f}"
              f"{r.per_class_recall.get('neutral',0)*100:6.0f}%{r.per_class_precision.get('neutral',0)*100:6.0f}%")
        client.save()

    best = max(CONFIGS, key=lambda c: (dev_reports[c[0]].macro_f1, dev_reports[c[0]].accuracy))
    print(f"\n  → dev 最良: {best[0]} (macroF1={dev_reports[best[0]].macro_f1*100:.1f})")

    print(f"\n=== test 確認 (test n={len(split.test)}) — 凍結した最良設定 ===")
    r = eval_config("BEST@test", best[1], best[2], best[3], cache, distilled, split.test, train_texts, client)
    ro = eval_config("rules-only@test", 0, 0.0, False, cache, distilled, split.test, train_texts, client)
    client.save()
    print(f"  {best[0]:<22} acc={r.accuracy*100:.1f}%  macroF1={r.macro_f1*100:.1f}  "
          f"neu_R={r.per_class_recall.get('neutral',0)*100:.0f}% neu_P={r.per_class_precision.get('neutral',0)*100:.0f}%")
    print(f"  {'rules-only(参照)':<22} acc={ro.accuracy*100:.1f}%  macroF1={ro.macro_f1*100:.1f}")

    with open("results/tune_results.json", "w", encoding="utf-8") as fp:
        json.dump({"dev": {n: dev_reports[n].accuracy for n in dev_reports}, "best": best[0],
                   "test_best_acc": r.accuracy, "test_rules_only_acc": ro.accuracy}, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: tune_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
