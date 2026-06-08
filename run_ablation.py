#!/usr/bin/env python3
"""経験キャッシュ型ハーネス — アブレーション実験。

各分割（同テーマ / LOTO）で 10 アームを比較し、検索・ルール・教師の寄与を分離する。
妥当性コントロール: 同一 test 上の再計測ベースライン / nn-copy / ラベル反転 / oracle / McNemar / 新規層別。
"""
from __future__ import annotations

import json
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import LABELS, Settings
from cache_harness.dataset import Split, audit_split, split_loto, split_same_theme
from cache_harness.experiment import (
    distill_from_train,
    make_cache,
    make_client,
    run_arm,
    static_sample,
)
from cache_harness.llm import warmup
from cache_harness.retriever import NgramRetriever, char_ngrams
from cache_harness.rules import oracle_rules

SETTINGS = Settings()
SEED = 0


def _cos_sim_fn(query: str, train_texts: list[str]) -> float:
    r = NgramRetriever(sizes=SETTINGS.ngram_sizes)
    for i, t in enumerate(train_texts):
        r.add(i, t)
    hits = r.search(query, 1)
    return hits[0][1] if hits else 0.0


def build_arms(distilled: tuple[str, ...], static_ex) -> list[tuple[str, ArmConfig, tuple[str, ...]]]:
    return [
        ("a:baseline(0.8B)", ArmConfig(), ()),
        ("b:static-kshot", ArmConfig(static_exemplars=static_ex), ()),
        ("c:dynamic-kshot", ArmConfig(use_retrieval=True), ()),
        ("d:rules-only(k0)", ArmConfig(use_rules=True), distilled),
        ("e:dynamic+rules", ArmConfig(use_retrieval=True, use_rules=True), distilled),
        ("f:+short-circuit", ArmConfig(use_retrieval=True, use_rules=True, use_short_circuit=True), distilled),
        ("g:+escalation", ArmConfig(use_retrieval=True, use_rules=True, use_short_circuit=True, use_escalation=True), distilled),
        ("h:nn-copy(noLLM)", ArmConfig(mode="nn_copy", use_retrieval=True), ()),
        ("i:corrupt-exemplar", ArmConfig(use_retrieval=True, corrupt_exemplars=True), ()),
        ("j:oracle-rules", ArmConfig(use_retrieval=True, use_rules=True), oracle_rules()),
    ]


def run_split(split: Split, client) -> dict:
    train_texts = [it.text for it in split.train]
    audit = audit_split(split, lambda q, ts: _cos_sim_fn(q, ts))
    print(f"\n{'#' * 86}\n# 分割: {split.kind}  (train={audit['n_train']} dev={audit['n_dev']} test={audit['n_test']})")
    print(f"# 漏洩監査: 完全一致={audit['exact_overlap']}  "
          f"編集比 max={audit['edit_ratio']['max']:.2f}/mean={audit['edit_ratio']['mean']:.2f}  "
          f"検索cos max={audit['retrieval_cosine']['max']:.2f}/mean={audit['retrieval_cosine']['mean']:.2f}")

    cache = make_cache(split.train, SETTINGS)
    distilled, _train_eval = distill_from_train(split.train, SETTINGS, client)
    print(f"# 蒸留ルール({len(distilled)}件, train誤分類のみ由来):")
    for r in distilled:
        print(f"#   - {r}")
    static_ex = static_sample(split.train, SETTINGS.k, SEED)

    reports: dict[str, metrics.Report] = {}
    outcomes_by_arm: dict[str, list] = {}
    print("\n" + metrics.header())
    for name, arm, rules in build_arms(distilled, static_ex):
        clf = Classifier(cache=cache, settings=SETTINGS, client=client, rules=rules, arm=arm)
        outcomes = run_arm(clf, split.test, train_texts, SETTINGS)
        rep = metrics.aggregate(name, outcomes)
        reports[name] = rep
        outcomes_by_arm[name] = outcomes
        print(rep.summary_line())

    _print_analysis(reports, outcomes_by_arm)
    return _to_json(split, audit, distilled, reports)


def _print_analysis(reports, outcomes_by_arm) -> None:
    base = outcomes_by_arm["a:baseline(0.8B)"]
    print("\n  [McNemar 片側検定 — base=a に対する改善の有意性]")
    for name in ("c:dynamic-kshot", "d:rules-only(k0)", "e:dynamic+rules", "g:+escalation"):
        m = metrics.mcnemar_better(base, outcomes_by_arm[name])
        print(f"    a → {name:<20} b={m['b_base_only']:2d} c={m['c_variant_only']:2d} p={m['p_one_sided']:.4f}")

    print("\n  [neutral クラス: 再現率/適合率 — 過補正(precision低下)の検査]")
    for name in ("a:baseline(0.8B)", "c:dynamic-kshot", "e:dynamic+rules", "j:oracle-rules"):
        r = reports[name]
        print(f"    {name:<20} recall={r.per_class_recall.get('neutral', 0)*100:5.1f}%  "
              f"precision={r.per_class_precision.get('neutral', 0)*100:5.1f}%")

    print("\n  [記憶 vs 汎化 — 新規(編集比<dup)のみ精度。検索コピー検証は h/i]")
    for name in ("a:baseline(0.8B)", "c:dynamic-kshot", "e:dynamic+rules", "h:nn-copy(noLLM)", "i:corrupt-exemplar"):
        r = reports[name]
        print(f"    {name:<20} 全体={r.accuracy*100:5.1f}%  新規({r.novel_n})={r.novel_accuracy*100:5.1f}%")

    print("\n  [混同行列 a:baseline]")
    _print_confusion(reports["a:baseline(0.8B)"])
    print("  [混同行列 e:dynamic+rules]")
    _print_confusion(reports["e:dynamic+rules"])


def _print_confusion(rep: metrics.Report) -> None:
    print("        pred:  " + "  ".join(f"{l[:3]:>5}" for l in LABELS))
    for gold in LABELS:
        row = rep.confusion.get(gold, {})
        print(f"    gold {gold[:3]:>3}: " + "  ".join(f"{row.get(p, 0):>5}" for p in LABELS))


def _to_json(split, audit, distilled, reports) -> dict:
    return {
        "split": split.kind,
        "audit": audit,
        "distilled_rules": list(distilled),
        "reports": {
            name: {
                "accuracy": r.accuracy, "acc_ci": r.acc_ci, "macro_f1": r.macro_f1,
                "valid_rate": r.valid_rate, "per_class_recall": r.per_class_recall,
                "per_class_precision": r.per_class_precision, "novel_accuracy": r.novel_accuracy,
                "novel_n": r.novel_n, "median_latency": r.median_latency, "p90_latency": r.p90_latency,
                "total_llm_calls": r.total_llm_calls, "route_counts": r.route_counts,
                "confusion": r.confusion,
            }
            for name, r in reports.items()
        },
    }


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    status = warmup(client.inner, [SETTINGS.student_model, SETTINGS.teacher_model])
    if not all(status.values()):
        print(f"接続エラー: {status}", file=sys.stderr)
        sys.exit(1)

    results = []
    for split in (split_same_theme(SEED), split_loto(SEED)):
        results.append(run_split(split, client))
        client.save()

    with open("ablation_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: ablation_results.json  (LLM応答キャッシュ: hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
