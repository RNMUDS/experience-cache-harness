#!/usr/bin/env python3
"""Kaggle 直取得: Twitter US Airline Sentiment (3クラス実ツイート) → data/ext_airline_{train,dev,test}.jsonl。

要 Kaggle 認証: ~/.kaggle/kaggle.json （{"username": "...", "key": "..."}）。`pip install kaggle` 済みであること。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fetch_external import clip, per_class_split, save

OUT = Path("data/kaggle_airline")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "Tweets.csv").exists():
        r = subprocess.run(["kaggle", "datasets", "download", "-d",
                            "crowdflower/twitter-airline-sentiment", "-p", str(OUT), "--unzip"])
        if r.returncode != 0:
            print("kaggle download failed — set up ~/.kaggle/kaggle.json", file=sys.stderr)
            sys.exit(1)
    df = pd.read_csv(OUT / "Tweets.csv")
    rows = [(clip(str(t)), s) for t, s in zip(df["text"], df["airline_sentiment"]) if str(t).strip()]
    labels = ["negative", "neutral", "positive"]
    save("ext_airline", per_class_split(rows, labels), labels)


if __name__ == "__main__":
    main()
