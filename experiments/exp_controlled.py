#!/usr/bin/env python3
"""統一パイプラインでの controlled 研究 — JP同テーマ & LOTO で汎化(記憶でない)を担保。

アーム: zeroshot / dyn_embed / cache(embed+rules) / oracle / corrupt(反転) / kNN / embed-LR。
コントロール: McNemar, neutral 再現/適合, 新規層別(編集距離=検索と独立), 漏洩監査, oracle vs distilled。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import LABELS, Settings
from cache_harness.dataset import audit_split, split_loto, split_same_theme
from cache_harness.experiment import distill_from_train, make_cache, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.retriever import NgramRetriever
from cache_harness.rules import oracle_rules_jp
from cache_harness.supervised import accuracy, logreg_predict
from cache_harness.task import SENTIMENT_JP

MODEL = "qwen3.5:0.8b"
S = Settings(student_model=MODEL, retriever="embedding")
TASK = SENTIMENT_JP


def cos_sim_fn(q, ts):
    r = NgramRetriever(); [r.add(i, t) for i, t in enumerate(ts)]
    h = r.search(q, 1); return h[0][1] if h else 0.0


def run_split(split, client) -> dict:
    tr = [it.text for it in split.train]
    audit = audit_split(split, cos_sim_fn)
    cache = make_cache(split.train, S)
    distilled, _ = distill_from_train(split.train, S, client, TASK)
    oracle = oracle_rules_jp()

    arms = [
        ("zeroshot", None, ArmConfig(), ()),
        ("dyn_embed", cache, ArmConfig(use_retrieval=True), ()),
        ("cache(emb+rules)", cache, ArmConfig(use_retrieval=True, use_rules=True), distilled),
        ("oracle-rules", cache, ArmConfig(use_retrieval=True, use_rules=True), oracle),
        ("corrupt-exemplar", cache, ArmConfig(use_retrieval=True, corrupt_exemplars=True), ()),
        ("kNN(noLLM)", cache, ArmConfig(mode="nn_copy", use_retrieval=True), ()),
    ]
    outs, reps = {}, {}
    print(f"\n{'#'*86}\n# {split.kind}: train={audit['n_train']} test={audit['n_test']}  "
          f"漏洩: 完全一致={audit['exact_overlap']} 編集max={audit['edit_ratio']['max']:.2f} 検索cos max={audit['retrieval_cosine']['max']:.2f}")
    print(f"# 蒸留ルール: {' / '.join(distilled)[:160] if distilled else 'なし'}")
    print("\n" + metrics.header())
    for name, c, arm, rules in arms:
        outs[name] = run_arm(Classifier(c, S, client, TASK, rules, arm), split.test, tr, S)
        reps[name] = metrics.aggregate(name, outs[name])
        print(reps[name].summary_line())

    # embed-LR（非パラメトリック教師あり）
    lr_acc = accuracy(logreg_predict(split.train, split.test, S.embed_model), [it.label for it in split.test])
    print(f"  {'embed-LR(noLLM)':<30}{lr_acc*100:6.1f}%")

    print("\n  [McNemar vs zeroshot]")
    for name in ("dyn_embed", "cache(emb+rules)", "kNN(noLLM)"):
        m = metrics.mcnemar_better(outs["zeroshot"], outs[name])
        print(f"    {name:<18} b={m['b_base_only']:2d} c={m['c_variant_only']:2d} p={m['p_one_sided']:.4f}")
    print("  [neutral 再現/適合]")
    for name in ("zeroshot", "dyn_embed", "cache(emb+rules)", "kNN(noLLM)"):
        r = reps[name]
        print(f"    {name:<18} R={r.per_class_recall.get('neutral',0)*100:5.1f}% P={r.per_class_precision.get('neutral',0)*100:5.1f}%")
    print("  [記憶 vs 汎化: 新規(編集比<dup)のみ精度]")
    for name in ("zeroshot", "dyn_embed", "cache(emb+rules)", "kNN(noLLM)"):
        r = reps[name]
        print(f"    {name:<18} 全体={r.accuracy*100:5.1f}% 新規({r.novel_n})={r.novel_accuracy*100:5.1f}%")

    return {"split": split.kind, "audit": audit, "distilled_rules": list(distilled), "embed_lr": lr_acc,
            "reports": {n: {"acc": r.accuracy, "acc_ci": r.acc_ci, "macro_f1": r.macro_f1,
                            "neutral_recall": r.per_class_recall.get("neutral", 0),
                            "neutral_precision": r.per_class_precision.get("neutral", 0),
                            "novel_acc": r.novel_accuracy, "novel_n": r.novel_n} for n, r in reps.items()}}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    if not warmup(client.inner, [MODEL, S.embed_model]).get(MODEL):
        print("接続エラー", file=sys.stderr); sys.exit(1)
    results = [run_split(split_same_theme(0), client), run_split(split_loto(0), client)]
    client.save()
    with open("results/controlled_results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: controlled_results.json (cache hits={client.hits} misses={client.misses})")


if __name__ == "__main__":
    main()
