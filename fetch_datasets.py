#!/usr/bin/env python3
"""公開データセットを取得し、バランス・サンプリングして data/<task>_{train,dev,test}.jsonl に保存。

著者バイアスを除くため公開ベンチを使う（英: SST-2/AG News/TREC, 日: multilingual-sentiments ja）。
自作JP感情(controlled)は cache_harness/dataset.py のものを別途利用。
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT = Path("data")
OUT.mkdir(exist_ok=True)
SEED = 0
N_TRAIN, N_DEV, N_TEST = 180, 60, 150


def _balanced_split(rows, labels, seed):
    """ラベル均衡で train/dev/test に分割。"""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for text, lab in rows:
        by_label[lab].append(text)
    for lab in by_label:
        rng.shuffle(by_label[lab])
    per = {"train": N_TRAIN, "dev": N_DEV, "test": N_TEST}
    n_lab = len(labels)
    out = {s: [] for s in per}
    for lab in labels:
        texts = by_label[lab]
        i = 0
        for split in ("train", "dev", "test"):
            take = max(1, per[split] // n_lab)
            for t in texts[i:i + take]:
                out[split].append({"text": t, "label": lab})
            i += take
    for split in out:
        rng.shuffle(out[split])
    return out


def save(task, rows, labels):
    splits = _balanced_split(rows, labels, SEED)
    for split, items in splits.items():
        path = OUT / f"{task}_{split}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in items), encoding="utf-8")
    print(f"  {task}: labels={labels} train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")


def fetch_sst2():
    ds = load_dataset("glue", "sst2", split="train")
    m = {0: "negative", 1: "positive"}
    rows = [(r["sentence"].strip(), m[r["label"]]) for r in ds if r["sentence"].strip()]
    save("sst2", rows, ["positive", "negative"])


def fetch_agnews():
    ds = load_dataset("ag_news", split="train")
    m = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
    rows = [(r["text"].strip().replace("\n", " "), m[r["label"]]) for r in ds]
    save("agnews", rows, ["world", "sports", "business", "scitech"])


def fetch_trec():
    ds = load_dataset("trec", split="train")
    key = "coarse_label" if "coarse_label" in ds.features else "label-coarse"
    m = {0: "abbr", 1: "entity", 2: "description", 3: "human", 4: "location", 5: "number"}
    rows = [(r["text"].strip(), m[r[key]]) for r in ds]
    save("trec", rows, ["abbr", "entity", "description", "human", "location", "number"])


def fetch_jp_sentiment():
    ds = load_dataset("tyqiangz/multilingual-sentiments", "japanese", split="train")
    names = ds.features["label"].names  # 例: ['positive','neutral','negative']
    rows = [(r["text"].strip().replace("\n", " "), names[r["label"]]) for r in ds if r["text"].strip()]
    save("jp_sentiment", rows, list(names))


def main():
    for name, fn in [("sst2", fetch_sst2), ("agnews", fetch_agnews),
                     ("trec", fetch_trec), ("jp_sentiment", fetch_jp_sentiment)]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
