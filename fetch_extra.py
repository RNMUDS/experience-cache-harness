#!/usr/bin/env python3
"""TREC（生ファイル）と JP感情（parquet系）をスクリプト不要の経路で取得。"""
from __future__ import annotations

import urllib.request

from fetch_datasets import save

TREC_MAP = {"ABBR": "abbr", "ENTY": "entity", "DESC": "description",
            "HUM": "human", "LOC": "location", "NUM": "number"}
TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def fetch_trec():
    rows = []
    for url in TREC_URLS.values():
        data = urllib.request.urlopen(url, timeout=30).read().decode("latin-1")
        for line in data.splitlines():
            if not line.strip():
                continue
            head, _, text = line.partition(" ")
            coarse = head.split(":")[0]
            if coarse in TREC_MAP and text.strip():
                rows.append((text.strip(), TREC_MAP[coarse]))
    save("trec", rows, list(TREC_MAP.values()))


def fetch_jp():
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "japanese", split="train")
    names = ds.features["label"].names  # ['negative','neutral','positive']
    rows = [(r["text"].strip().replace("\n", " "), names[r["label"]]) for r in ds if r["text"].strip()]
    save("jp_sentiment", rows, list(names))


def main():
    for name, fn in [("trec", fetch_trec), ("jp_sentiment", fetch_jp)]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
