#!/usr/bin/env python3
"""生成タスク・パイロット — 経験キャッシュは「分類」を超えて「生成(短答QA)」にも効くか。

オープン短答QA(web_questions)で3方式を比較（EM / token-F1）:
  - cache-NN-answer : 最類似の既出質問の答えをそのまま返す（LLM不使用）。
  - zeroshot LLM    : 0.8B が直接生成。
  - LLM + RAG few-shot: 類似(Q,A)をk個提示して生成。
分類で効いた「検索キャッシュ」が、知識依存の開放生成では限界に当たるかを正直に測る。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import math
import re
import sys
from pathlib import Path

from cache_harness.embedding_retriever import _embed
from cache_harness.llm import OllamaClient, warmup

EMBED = "nomic-embed-text"
MODEL = "qwen3.5:0.8b"
K = 4


def load(split):
    return [json.loads(l) for l in (Path("data") / f"genqa_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> int:
    p = norm(pred)
    return int(any(norm(g) and (norm(g) == p or norm(g) in p) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    pt = norm(pred).split()
    best = 0.0
    for g in golds:
        gt = norm(g).split()
        if not pt or not gt:
            continue
        common = sum((min(pt.count(w), gt.count(w)) for w in set(pt)))
        if common == 0:
            continue
        prec, rec = common / len(pt), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def gen(client, prompt: str) -> str:
    res = client.chat(MODEL, [{"role": "user", "content": prompt}], think=False, num_predict=32)
    return (res.content or "").strip().split("\n")[0][:80] if res.ok else ""


def main() -> None:
    client = OllamaClient()
    print("[warmup]...", flush=True)
    if not warmup(client, [MODEL]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)
    train, test = load("train"), load("test")
    Xtr = [l2(_embed(r["question"], EMBED)) for r in train]
    Xte = [l2(_embed(r["question"], EMBED)) for r in test]

    res = {"nn": {"em": 0, "f1": 0.0}, "zeroshot": {"em": 0, "f1": 0.0}, "rag": {"em": 0, "f1": 0.0}}
    sim_top = []
    for q, qv in zip(test, Xte):
        sims = sorted(((sum(a * b for a, b in zip(qv, tv)), i) for i, tv in enumerate(Xtr)), reverse=True)
        top = sims[0][0]; sim_top.append(top)
        nn_ans = train[sims[0][1]]["answers"][0]
        exemplars = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answers'][0]}\n\n" for _, i in sims[:K])
        zs = gen(client, f"Answer with only the answer, as few words as possible.\nQuestion: {q['question']}\nAnswer:")
        rag = gen(client, f"Answer with only the answer, as few words as possible.\n\n{exemplars}Question: {q['question']}\nAnswer:")
        for name, pred in (("nn", nn_ans), ("zeroshot", zs), ("rag", rag)):
            res[name]["em"] += em(pred, q["answers"])
            res[name]["f1"] += f1(pred, q["answers"])

    n = len(test)
    print(f"\n{'#'*70}\n# 生成タスク(web_questions, オープン短答QA) n={n}  検索類似 平均top={sum(sim_top)/n:.2f}")
    print(f"  {'method':<22}{'EM':>8}{'token-F1':>10}")
    out = {}
    for name in ("nn", "zeroshot", "rag"):
        emv, f1v = res[name]["em"] / n, res[name]["f1"] / n
        out[name] = {"em": emv, "f1": f1v}
        label = {"nn": "cache-NN-answer(noLLM)", "zeroshot": "zeroshot LLM", "rag": "LLM+RAG few-shot"}[name]
        print(f"  {label:<22}{emv*100:>7.1f}%{f1v*100:>9.1f}")

    with open("results/genqa_results.json", "w", encoding="utf-8") as fp:
        json.dump({"n": n, "mean_top_sim": sum(sim_top) / n, "methods": out}, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: genqa_results.json")


if __name__ == "__main__":
    main()
