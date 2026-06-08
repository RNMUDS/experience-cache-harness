#!/usr/bin/env python3
"""リトリーバ比較 — 文字n-gram vs 埋め込み。exemplar が効かないのは検索が弱いからか？を切り分け。

dynamic-kshot（ルール無し・exemplar のみ）で両リトリーバを比較。埋め込みで c が大きく上がるなら
「char-ngram の言い換え再現率が exemplar のボトルネック」だと分かる。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import split_same_theme
from cache_harness.experiment import make_cache, make_client, run_arm
from cache_harness.llm import warmup


def main() -> None:
    client = make_client(path="llm_cache.json")
    client.load()
    print("[warmup]...", flush=True)
    if not warmup(client.inner, ["qwen3.5:0.8b"]).get("qwen3.5:0.8b"):
        print("接続エラー", file=sys.stderr)
        sys.exit(1)

    split = split_same_theme(0)
    train_texts = [it.text for it in split.train]
    print(f"\n=== dynamic-kshot (ルール無し) リトリーバ比較  test n={len(split.test)} ===")
    print(f"  {'retriever':<14}{'acc':>7}{'macroF1':>9}{'neu_R':>7}{'平均top類似':>12}")
    for kind in ("char_ngram", "embedding"):
        settings = Settings(retriever=kind)
        cache = make_cache(split.train, settings)
        clf = Classifier(cache=cache, settings=settings, client=client, arm=ArmConfig(use_retrieval=True))
        outs = run_arm(clf, split.test, train_texts, settings)
        rep = metrics.aggregate(kind, outs)
        avg_sim = sum(o.sim for o in outs) / len(outs)
        print(f"  {kind:<14}{rep.accuracy*100:6.1f}%{rep.macro_f1*100:8.1f}"
              f"{rep.per_class_recall.get('neutral',0)*100:6.0f}%{avg_sim:11.3f}")
        client.save()
    print(f"\n(cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
