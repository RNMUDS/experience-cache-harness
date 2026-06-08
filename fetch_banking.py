#!/usr/bin/env python3
"""banking77（77クラス intent, parquet系）と JP公開感情（parquet経路）を取得。クラス数軸を77まで拡張。"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT = Path("data"); OUT.mkdir(exist_ok=True)


def per_class_split(rows, labels, n_tr, n_dev, n_te, seed=0):
    rng = random.Random(seed)
    by = defaultdict(list)
    for t, l in rows:
        by[l].append(t)
    out = {"train": [], "dev": [], "test": []}
    for l in labels:
        texts = by[l]; rng.shuffle(texts)
        out["train"] += [{"text": t, "label": l} for t in texts[:n_tr]]
        out["dev"] += [{"text": t, "label": l} for t in texts[n_tr:n_tr + n_dev]]
        out["test"] += [{"text": t, "label": l} for t in texts[n_tr + n_dev:n_tr + n_dev + n_te]]
    for s in out:
        rng.shuffle(out[s])
    return out


def save(task, splits, labels):
    for s, items in splits.items():
        (OUT / f"{task}_{s}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")
    print(f"  {task}: {len(labels)} labels train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")


def fetch_banking():
    ds = load_dataset("mteb/banking77", split="train")  # parquet-native ミラー
    rows = [(r["text"].strip(), r["label_text"]) for r in ds if r["text"].strip()]
    labels = sorted({l for _, l in rows})
    save("banking77", per_class_split(rows, labels, 5, 2, 3), labels)


def main():
    for name, fn in [("banking77", fetch_banking)]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
