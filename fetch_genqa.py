#!/usr/bin/env python3
"""web_questions（オープン短答QA）を取得 → data/genqa_{train,test}.jsonl。生成タスク用。"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path("data"); OUT.mkdir(exist_ok=True)


def main():
    ds = load_dataset("web_questions", split="train")
    rows = [{"question": r["question"].strip(), "answers": list(r["answers"])}
            for r in ds if r["question"].strip() and r["answers"]]
    rng = random.Random(0); rng.shuffle(rows)
    train, test = rows[:300], rows[300:450]
    (OUT / "genqa_train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train), encoding="utf-8")
    (OUT / "genqa_test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in test), encoding="utf-8")
    print(f"  genqa: train={len(train)} test={len(test)}  ex={train[0]['question']} -> {train[0]['answers'][:2]}")


if __name__ == "__main__":
    main()
