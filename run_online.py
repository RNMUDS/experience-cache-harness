#!/usr/bin/env python3
"""オンライン生涯学習ショーケース — ゼロ知識から始め、経験を貯め、瞬時に再利用する。

ユーザーの中心像「経験から役立った知識をキャッシュし、つぎに瞬時に活かす」を 1 本のストリームで可視化:
  1. 空のキャッシュ・ルール無しで開始（素の 0.8B = 約73%）。
  2. 毎ステップ: 検索 exemplar + 現行ルール + 短絡 で回答 → 正解を開示してキャッシュに記憶。
  3. 一定間隔で、自分の誤りからルールを自己蒸留して常設ルールを更新（自己改善ループ）。
  4. 既出クエリが再来したら短絡で即答（0 LLM・瞬時）。
教師は使わない（凍結 0.8B 単独の学習を主張するため）。
時系列で 精度 / 短絡率 / 1問あたりLLM呼数 / 遅延 の推移を見る。
"""
from __future__ import annotations

import json
import random
import statistics
import sys

from cache_harness.cache import Experience, ExperienceCache
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.dataset import all_items
from cache_harness.experiment import make_retriever, make_client
from cache_harness.llm import warmup
from cache_harness.rules import distill_rules

BASE = Settings()
SEEDS = [0, 1]
STREAM_LEN = 400
REDISTILL_EVERY = 40
WINDOW = 50


def stream_once(items, seed: int, client) -> list[dict]:
    rng = random.Random(seed)
    stream = [rng.choice(items) for _ in range(STREAM_LEN)]
    cache = ExperienceCache(retriever=make_retriever(BASE))
    rules: tuple[str, ...] = ()
    student_log: list[tuple[str, str, str]] = []  # 0.8B 自身の (text,gold,pred) のみ
    seen: set[str] = set()
    steps: list[dict] = []

    for t, it in enumerate(stream):
        arm = ArmConfig(use_retrieval=len(cache) > 0, use_rules=bool(rules), use_short_circuit=True)
        clf = Classifier(cache=cache, settings=BASE, client=client, rules=rules, arm=arm)
        pred = clf.classify(it.text)
        correct = int(pred.valid and pred.label == it.label)
        steps.append({"t": t, "correct": correct, "route": pred.source, "calls": pred.n_llm_calls,
                      "latency": pred.latency, "repeat": it.text in seen})
        if pred.source in ("student", "student_then_teacher"):
            student_log.append((it.text, it.label, pred.label or ""))
        seen.add(it.text)
        cache.add(Experience(it.text, it.label, source="gold", theme=it.theme))
        if (t + 1) % REDISTILL_EVERY == 0 and student_log:
            rules = distill_rules(student_log, max_rules=BASE.max_rules)
    return steps


def _windowed(all_steps: list[list[dict]]) -> list[dict]:
    """全シードを位置で平均し、WINDOW 区間ごとに集計。"""
    out = []
    for w0 in range(0, STREAM_LEN, WINDOW):
        w1 = min(w0 + WINDOW, STREAM_LEN)
        rows = [s for steps in all_steps for s in steps[w0:w1]]
        n = len(rows)
        out.append({
            "window": f"{w0}-{w1}",
            "acc": sum(r["correct"] for r in rows) / n,
            "short_circuit_rate": sum(r["route"] == "short_circuit" for r in rows) / n,
            "calls_per_q": sum(r["calls"] for r in rows) / n,
            "latency": statistics.mean(r["latency"] for r in rows),
        })
    return out


def main() -> None:
    client = make_client(path="llm_cache.json")
    client.load()
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [BASE.student_model]).get(BASE.student_model):
        print("接続エラー", file=sys.stderr)
        sys.exit(1)

    items = all_items()
    all_steps = [stream_once(items, s, client) for s in SEEDS]
    client.save()

    windows = _windowed(all_steps)
    print(f"\n{'#'*72}\n# オンライン学習ショーケース（ゼロ知識→自己改善, seeds={SEEDS}, len={STREAM_LEN}）")
    print(f"  {'区間':<10}{'精度':>7}{'短絡率':>8}{'LLM/問':>8}{'遅延':>8}")
    for w in windows:
        print(f"  {w['window']:<10}{w['acc']*100:6.1f}%{w['short_circuit_rate']*100:7.0f}%"
              f"{w['calls_per_q']:8.2f}{w['latency']:7.2f}s")

    # 初回 vs 再来（汎化 vs 瞬時再利用）
    first = [s for steps in all_steps for s in steps if not s["repeat"]]
    repeat = [s for steps in all_steps for s in steps if s["repeat"]]
    print(f"\n  初回出現(汎化): n={len(first)} acc={sum(r['correct'] for r in first)/len(first)*100:.1f}% "
          f"LLM/問={sum(r['calls'] for r in first)/len(first):.2f}")
    print(f"  再来(瞬時再利用): n={len(repeat)} acc={sum(r['correct'] for r in repeat)/len(repeat)*100:.1f}% "
          f"短絡率={sum(r['route']=='short_circuit' for r in repeat)/len(repeat)*100:.0f}% "
          f"LLM/問={sum(r['calls'] for r in repeat)/len(repeat):.2f}")

    with open("online_results.json", "w", encoding="utf-8") as fp:
        json.dump({"windows": windows, "seeds": SEEDS, "stream_len": STREAM_LEN}, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: online_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
