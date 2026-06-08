#!/usr/bin/env python3
"""学習曲線 — キャッシュが育つほど held-out 精度が上がるかを多シードで測る。

中心主張「経験を貯めるほど凍結0.8Bが賢くなる」の核。記憶との切り分けのため:
  - 各 K で coverage（test に編集比 high の近傍がある割合）と low-coverage 部分集合精度も報告。
  - exemplar のみ（ルール無し）で純粋なキャッシュ効果を分離。
"""
from __future__ import annotations

import json
import statistics
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import LABELS, Settings
from cache_harness.dataset import LabeledItem, split_same_theme, split_loto
from cache_harness.experiment import make_cache, make_client, run_arm
from cache_harness.llm import warmup
import random

SETTINGS = Settings()
K_GRID = [0, 8, 16, 32, 64, 102]
SEEDS = [0, 1, 2]


def _sample_train(train: tuple[LabeledItem, ...], k: int, seed: int) -> list[LabeledItem]:
    if k >= len(train):
        return list(train)
    rng = random.Random(seed)
    pool = list(train)
    rng.shuffle(pool)
    return pool[:k]


def _low_coverage_acc(outcomes: list[metrics.Outcome]) -> tuple[float, int]:
    """新規（編集比<dup）= 記憶では解けない部分集合の精度。"""
    novel = [o for o in outcomes if o.novel]
    if not novel:
        return 0.0, 0
    return sum(o.correct for o in novel) / len(novel), len(novel)


def curve_for_split(split, client) -> dict:
    train_texts_full = [it.text for it in split.train]
    print(f"\n{'#'*80}\n# 学習曲線: {split.kind}  test={len(split.test)}  seeds={SEEDS}")
    print(f"  {'K':>4}{'  全体acc(mean±sd)':>20}{'  新規acc':>12}{'  新規n':>8}{'  cover率':>9}")
    rows = []
    for k in K_GRID:
        accs, novel_accs, novel_ns, covers = [], [], [], []
        for seed in SEEDS:
            sub = _sample_train(split.train, k, seed)
            cache = make_cache(sub, SETTINGS) if sub else make_cache([], SETTINGS)
            arm = ArmConfig(use_retrieval=bool(sub))  # K=0 は素のベースライン
            clf = Classifier(cache=cache, settings=SETTINGS, client=client, arm=arm)
            outs = run_arm(clf, split.test, [it.text for it in sub], SETTINGS)
            rep = metrics.aggregate(f"K{k}", outs)
            accs.append(rep.accuracy)
            na, nn = _low_coverage_acc(outs)
            novel_accs.append(na)
            novel_ns.append(nn)
            covers.append(1 - nn / len(outs) if outs else 0.0)
        mean_acc = statistics.mean(accs)
        sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        rows.append({"k": k, "acc_mean": mean_acc, "acc_sd": sd,
                     "novel_acc": statistics.mean(novel_accs), "novel_n": statistics.mean(novel_ns),
                     "coverage": statistics.mean(covers)})
        print(f"  {k:>4}{mean_acc*100:>13.1f}±{sd*100:>4.1f}%{statistics.mean(novel_accs)*100:>11.1f}%"
              f"{statistics.mean(novel_ns):>8.0f}{statistics.mean(covers)*100:>8.0f}%")
    return {"split": split.kind, "k_grid": K_GRID, "seeds": SEEDS, "rows": rows}


def main() -> None:
    client = make_client(path="llm_cache.json")
    client.load()
    print("[warmup]...", flush=True)
    status = warmup(client.inner, [SETTINGS.student_model])
    if not status.get(SETTINGS.student_model):
        print("接続エラー", file=sys.stderr)
        sys.exit(1)
    results = [curve_for_split(split, client) for split in (split_same_theme(0), split_loto(0))]
    client.save()
    with open("learning_curve_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: learning_curve_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
