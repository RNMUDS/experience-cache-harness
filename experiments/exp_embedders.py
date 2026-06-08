#!/usr/bin/env python3
"""埋め込みモデル比較 — 非パラメトリック・キャッシュ(kNN/embed-LR)の強さは embedder に依存するか。

「強さは埋め込み由来」caveat を定量化。embedder=nomic/bge-m3/mxbai/all-minilm × データ(実6種+banking77)。
LLM不使用（埋め込みのみ）。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

from cache_harness import metrics
from cache_harness.experiment import load_public
from cache_harness.supervised import accuracy, knn_predict, logreg_predict

EMBEDDERS = ["nomic-embed-text", "bge-m3", "mxbai-embed-large", "all-minilm"]
TASKS = ["ext_imdb", "ext_airline", "ext_tweet", "ext_yelp", "ext_emotion", "ext_dbpedia", "banking77"]


def main() -> None:
    rows = []
    print(f"\n{'#'*96}\n# 埋め込みモデル比較（kNN / embed-LR, LLM不使用）")
    head = f"  {'dataset':<14}{'cls':>4}" + "".join(f"{e.split('-')[0][:7]:>16}" for e in EMBEDDERS)
    print(head); print("  " + "-" * (len(head)))
    for name in TASKS:
        split = load_public(name)
        test = list(split.test)[:150]
        gold = [it.label for it in test]
        n_cls = len({it.label for it in split.train})
        rec = {"task": name, "n_classes": n_cls, "by_embedder": {}}
        cells = []
        for emb in EMBEDDERS:
            try:
                knn = accuracy(knn_predict(split.train, test, emb, k=4), gold)
                lr = accuracy(logreg_predict(split.train, test, emb), gold)
                rec["by_embedder"][emb] = {"knn": knn, "embed_lr": lr}
                cells.append(f"{knn*100:6.1f}/{lr*100:5.1f}")
            except Exception as e:  # noqa: BLE001
                rec["by_embedder"][emb] = {"error": str(e)[:60]}
                cells.append("   n/a")
        rows.append(rec)
        print(f"  {name:<14}{n_cls:>4}" + "".join(f"{c:>16}" for c in cells))

    with open("results/embedders_results.json", "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print("\n  (各セル = kNN% / embed-LR%)")
    print(f"保存: embedders_results.json")


if __name__ == "__main__":
    main()
