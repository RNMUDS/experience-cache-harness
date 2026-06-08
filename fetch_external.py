#!/usr/bin/env python3
"""外部・実世界オープンデータ(Kaggle系/HFミラー)を取得。クラス数 2/3/5/6/14 の実データで再検証する。

著者バイアス・templateバイアスの無い、ノイズの多い実テキストで主要知見を外部検証するのが目的。
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT = Path("data"); OUT.mkdir(exist_ok=True)
TRAIN_PC, DEV_PC = 25, 8


def clip(t: str) -> str:
    return " ".join(t.strip().split())[:500]


def per_class_split(rows, labels, seed=0):
    rng = random.Random(seed)
    test_pc = max(4, 150 // len(labels))
    by = defaultdict(list)
    for t, l in rows:
        by[l].append(t)
    out = {"train": [], "dev": [], "test": []}
    for l in labels:
        xs = by[l]; rng.shuffle(xs)
        out["train"] += [{"text": t, "label": l} for t in xs[:TRAIN_PC]]
        out["dev"] += [{"text": t, "label": l} for t in xs[TRAIN_PC:TRAIN_PC + DEV_PC]]
        out["test"] += [{"text": t, "label": l} for t in xs[TRAIN_PC + DEV_PC:TRAIN_PC + DEV_PC + test_pc]]
    for s in out:
        rng.shuffle(out[s])
    return out


def save(task, splits, labels):
    for s, items in splits.items():
        (OUT / f"{task}_{s}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")
    print(f"  {task}: {len(labels)}cls train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")


def grab(task, hf, text_field, label_field, *, cfg=None, n=40000):
    ds = load_dataset(hf, cfg, split="train") if cfg else load_dataset(hf, split="train")
    ds = ds.shuffle(seed=0).select(range(min(len(ds), n)))  # クラス順ブロックを崩す（dbpedia/yelp対策）
    feat = ds.features.get(label_field)
    names = getattr(feat, "names", None)
    rows, labels = [], set()
    for r in ds:
        t = clip(str(r[text_field]))
        if not t:
            continue
        lab = names[r[label_field]] if names else str(r[label_field])
        rows.append((t, lab)); labels.add(lab)
    labels = sorted(labels)
    save(task, per_class_split(rows, labels), labels)


def main():
    jobs = [
        ("ext_tweet", "mteb/tweet_sentiment_extraction", "text", "label_text", None),
        ("ext_imdb", "stanfordnlp/imdb", "text", "label", None),
        ("ext_yelp", "Yelp/yelp_review_full", "text", "label", None),
        ("ext_emotion", "dair-ai/emotion", "text", "label", None),
        ("ext_dbpedia", "fancyzhx/dbpedia_14", "content", "label", None),
    ]
    for task, hf, tf, lf, cfg in jobs:
        try:
            grab(task, hf, tf, lf, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  {task}: FAILED {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
