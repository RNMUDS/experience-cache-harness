#!/usr/bin/env python3
"""既知/未知ルーティングの工学的検証 — 「未知をどう判定するか」に答える。

各テスト項目で gate 信号（①最近傍類似度 top1_sim ②非パラメ分類器の信頼度 margin）が
「非LLM経路(cache-LR)が正答するか」をどれだけ予測できるか(AUROC)を測定。
そのうえで gate≥τ→非LLM(0 LLM呼び出し) / gate<τ→LLM のルータを構成し、
精度 vs LLM呼び出し率の Pareto を、LR単独・LLM単独・oracle(項目ごと最良)と比較する。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import split_same_theme
from cache_harness.experiment import load_public, make_client
from cache_harness.llm import warmup
from cache_harness.supervised import embed_matrix
from cache_harness.task import SENTIMENT_JP, TASKS, Task

EMBED = "nomic-embed-text"
MODEL = "qwen3.5:0.8b"
N_TEST = 150
KAGREE = 8
S = Settings(student_model=MODEL, retriever="embedding")
DATASETS = ["jp_sentiment_authored", "sst2", "agnews", "trec", "banking77", "ext_imdb", "ext_dbpedia"]


def build_task(name, split) -> Task:
    if name == "jp_sentiment_authored":
        return SENTIMENT_JP
    if name in TASKS:
        return TASKS[name]
    labels = tuple(sorted({it.label for it in split.train}))
    return Task(name, labels, "Classify the text into exactly one of: " + ", ".join(labels) + ".", lang="en")


def load(name):
    return split_same_theme(0) if name == "jp_sentiment_authored" else load_public(name)


def auroc(score, target):
    if len(set(target)) < 2:
        return None
    return roc_auc_score(target, score)


def router_curve(gate, lr_correct, llm_correct):
    """gate≥τ→LR, gate<τ→LLM. 各τで (acc, llm_fraction)。"""
    out = []
    for tau in np.quantile(gate, np.linspace(0, 1, 21)):
        use_lr = gate >= tau
        acc = np.mean(np.where(use_lr, lr_correct, llm_correct))
        out.append((float(tau), float(acc), float(np.mean(~use_lr))))
    return out


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL]).get(MODEL):  # 埋め込みモデルは /api/embeddings で別途使用(chat warmup不可)
        print("接続エラー", file=sys.stderr); sys.exit(1)

    rows = []
    for ds in DATASETS:
        sp = load(ds); task = build_task(ds, sp)
        train = list(sp.train); test = list(sp.test)[:N_TEST]
        gold = np.array([it.label for it in test])
        Xtr = np.asarray(embed_matrix(train, EMBED)); Xte = np.asarray(embed_matrix(test, EMBED))
        ytr = np.array([it.label for it in train])
        # 非パラメ経路: ロジスティック回帰 + 信頼度(margin)
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, ytr)
        proba = clf.predict_proba(Xte); classes = clf.classes_
        lr_pred = classes[proba.argmax(1)]
        sp_ = np.sort(proba, axis=1)
        margin = sp_[:, -1] - sp_[:, -2]                  # gate②: 分類器信頼度
        sims = Xte @ Xtr.T
        top1_sim = sims.max(1)                            # gate①: 最近傍類似度
        # kNN 一致率(参考 gate)
        idx = np.argsort(-sims, axis=1)[:, :KAGREE]
        knn_agree = np.array([np.mean(ytr[ix] == max(set(ytr[ix]), key=list(ytr[ix]).count)) for ix in idx])
        # LLM 経路 (zeroshot 0.8b)
        zc = Classifier(None, S, client, task, (), ArmConfig())
        llm_pred = np.array([zc.classify(it.text).label or "" for it in test])
        client.save()

        lr_correct = (lr_pred == gold).astype(int)
        llm_correct = (llm_pred == gold).astype(int)
        lr_only = float(lr_correct.mean()); llm_only = float(llm_correct.mean())
        oracle = float(np.maximum(lr_correct, llm_correct).mean())
        gates = {"top1_sim": top1_sim, "lr_margin": margin, "knn_agree": knn_agree}
        au = {g: auroc(v, lr_correct) for g, v in gates.items()}
        # ルータ(信頼度margin採用): 精度を最大化するτ と、コスト最小化(acc≥max単独)τ
        curve = router_curve(margin, lr_correct, llm_correct)
        best_acc = max(curve, key=lambda c: c[1])
        floor = max(lr_only, llm_only)
        cost_min = min((c for c in curve if c[1] >= floor - 1e-9), key=lambda c: c[2], default=best_acc)
        rec = {"dataset": ds, "n_classes": len(task.labels), "n_test": len(test),
               "lr_only": lr_only, "llm_only": llm_only, "oracle": oracle, "auroc": au,
               "router_best": {"tau": best_acc[0], "acc": best_acc[1], "llm_frac": best_acc[2]},
               "router_costmin": {"tau": cost_min[0], "acc": cost_min[1], "llm_frac": cost_min[2]}}
        rows.append(rec)
        print(f"\n# {ds} ({len(task.labels)}cls, n={len(test)})")
        print(f"  LR-only={lr_only*100:.1f}  LLM-only={llm_only*100:.1f}  oracle={oracle*100:.1f}")
        print(f"  AUROC(gate→LR-correct): top1_sim={au['top1_sim']!s:.5}  lr_margin={au['lr_margin']!s:.5}  knn_agree={au['knn_agree']!s:.5}")
        print(f"  router(best acc): acc={best_acc[1]*100:.1f} @ LLM-calls={best_acc[2]*100:.0f}%")
        print(f"  router(min cost ≥ best single): acc={cost_min[1]*100:.1f} @ LLM-calls={cost_min[2]*100:.0f}%")
        with open("results/router_results.json", "w", encoding="utf-8") as fp:
            json.dump(rows, fp, ensure_ascii=False, indent=2)
        client.save()

    # 集計
    aus = [r["auroc"]["lr_margin"] for r in rows if r["auroc"]["lr_margin"] is not None]
    sims_au = [r["auroc"]["top1_sim"] for r in rows if r["auroc"]["top1_sim"] is not None]
    print(f"\n{'='*70}\nSUMMARY ({len(rows)} datasets)")
    print(f"  mean AUROC  lr_margin={np.mean(aus):.3f}  top1_sim={np.mean(sims_au):.3f}")
    print(f"  mean acc:  LR-only={np.mean([r['lr_only'] for r in rows])*100:.1f}  "
          f"LLM-only={np.mean([r['llm_only'] for r in rows])*100:.1f}  "
          f"oracle={np.mean([r['oracle'] for r in rows])*100:.1f}  "
          f"router(costmin)={np.mean([r['router_costmin']['acc'] for r in rows])*100:.1f} "
          f"@ {np.mean([r['router_costmin']['llm_frac'] for r in rows])*100:.0f}% LLM")
    print("保存: router_results.json")


if __name__ == "__main__":
    main()
