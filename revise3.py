#!/usr/bin/env python3
"""レビュー指摘の反映: 数値の整合・過剰主張の緩和・宙ぶらりん参照修正・重複引用統合・冗長削除・件数修正。"""
from __future__ import annotations
from pathlib import Path

t = Path("paper.md").read_text(encoding="utf-8")
def rep(old, new):
    global t
    assert t.count(old) >= 1, f"NOT FOUND: {old[:80]!r}"
    t = t.replace(old, new)
def soft(old, new):
    global t
    if old in t: t = t.replace(old, new)

# 1. Abstract: '<40%' を限定 + 6データ列挙修正
rep("(≈80% accuracy from 2 to 77 classes, where the zero-shot tiny LLM falls below 40%)",
    "(≈80% from 2 to 77 classes, versus a zero-shot tiny LLM that is strong on 2–3 classes but falls below 40% once the label space exceeds ~3)")
rep("an external re-test on six real-world datasets (IMDB, Yelp, Emotion, DBpedia, and Kaggle tweet-sentiment sets)",
    "an external re-test on six real-world datasets (IMDB, Kaggle airline and tweet sentiment, Yelp, Emotion, and DBpedia)")

# 2. Intro contribution: '<40%' 限定
rep("nearly invariant to label-space size (≈80% accuracy) while the zero-shot tiny LLM collapses below 40%;",
    "nearly invariant to label-space size (≈80%) while the zero-shot tiny LLM, strong on 2–3 classes, collapses below 40% beyond ~3 classes;")

# 3. Intro 'Preliminary results': 廃止構成の 73→97/40 を現行数値へ
rep("Preliminary results are encouraging: on the controlled Japanese set, a 0.8B model improves from a re-measured zero-shot baseline of roughly 73% to 94–97% with the cache, with gains *holding on the leave-one-theme-out split*, indicating genuine generalization rather than memorization; the neutral-class recall, a known tiny-model failure, rises from about 40% toward 100%; and, starting from zero knowledge online, short-circuiting eventually serves the large majority of queries, cutting LLM calls and latency by several-fold. Final numbers across all tasks and models are reported in Section \\ref{sec:results}.",
    "The cache helps most where the base model is weakest: on a held-out 0.8B multi-task study it raises TREC from 22.5% to 40.0% and AG News from 32.5% to 44.2% (paired McNemar $p<0.02$); on the leave-one-theme-out controlled set it lifts the already-strong (~90%) zero-shot baseline to 92% while restoring neutral-class recall from ~80% to ~100%, with gains holding on unseen themes (generalization, not memorization). Used as a non-parametric predictor it stays near 80% from 2 to 77 classes and rivals LoRA at zero training cost. Full numbers are in Section \\ref{sec:results}.")

# 4. Related Work / Method: 宙ぶらりん Table~X を実参照へ
rep("we show (Table~X) that the relative benefit", "we show (Fig. \\ref{fig:scaling}) that the relative benefit")
rep("Table~X reports the fraction of online queries served with zero LLM calls and the resulting reduction in calls and latency, alongside",
    "Fig. \\ref{fig:online} reports the fraction of online queries served with zero LLM calls and the resulting reduction in calls and latency, alongside")
rep("providing instant zero-LLM-call recall on recurring queries (Table~X).",
    "providing instant zero-LLM-call recall on recurring queries (Table \\ref{tab:finetune}).")
rep("Table~X reports the resulting joint trajectory of accuracy, short-circuit rate, LLM calls per query, and latency over the stream.",
    "Fig. \\ref{fig:online} reports the resulting joint trajectory of accuracy, short-circuit rate, LLM calls per query, and latency over the stream.")

# 5. §5.D scaling: cross-family 例外を明記
rep("and falls to essentially zero from 1.5B through 36B, where capable models already score ~100% zero-shot on these tasks.",
    "and is near zero for capable models from 1.5B through 36B (which already score ~100% zero-shot here); the only non-trivial gain above 1B is a cross-family exception, Llama-3.2-3B (+8.8 pp), whose zero-shot JP is just 61.8%.")

# 6. §5.G online: 検証不能な 94.7/100 を削除
rep("(Fig. \\ref{fig:online}); first-occurrence (generalization) accuracy was 94.7% and recurring queries were answered at 100% accuracy and zero cost.",
    "(Fig. \\ref{fig:online}); first-occurrence queries are handled by the LLM path, while recurring queries are increasingly served by the short-circuit at zero cost.")

# 7. §5.K seeds: 主張を embed+LR・4/5データに限定、Emotionは境界、検定種別を明記
rep("The cache is stable (embed+LR std 0.6–2.9 points) and its mean lies **far above the upper end of the zero-shot CI on every dataset**: DBpedia 92.1±0.6 vs zero-shot [47,64]; Banking77 78.3±1.1 vs [29,44]; Yelp 57.5±2.5 vs [14,27]; Emotion 50.5±2.9 vs [32,47]; IMDB 80.0±2.0 vs [42,58]. The intervals do not overlap, so the gap is statistically significant rather than seed-luck.",
    "The cache is stable (embed+LR std 0.6–2.9 points). The embed+LR mean exceeds the upper bound of the zero-shot Wilson CI on four of five datasets — DBpedia 92.1±0.6 vs [47,64]; Banking77 78.3±1.1 vs [29,44]; Yelp 57.5±2.5 vs [14,27]; IMDB 80.0±2.0 vs [42,58] — with Emotion the borderline case (50.5±2.9 vs zero-shot upper 47.3); the weaker kNN can fall inside the zero-shot CI (Emotion 40.9). As the seed band is a sampling standard deviation and the zero-shot interval a binomial CI, we treat this as strong but informal evidence; the formal within-item significance is the paired McNemar test of §5-B.")

# 8. §4-E の限定事項を §6 に集約（重複削除）
rep("**Acknowledged limitations.** The controlled set is single-author Japanese sentiment with small test $n$ (34 / 50), so confidence intervals are wide; the public benchmarks and cross-family models are included precisely to test external validity. The escalation/short-circuit thresholds remain dataset-dependent values tuned on dev. These are stated openly and motivate the cross-task and cross-model breadth of the design.",
    "Limitations are consolidated in Section \\ref{sec:discussion}.")

# 9. Banking77 引用
rep("to 77 (Banking77 intent detection)", "to 77 (Banking77 intent detection \\cite{casanueva2020banking77})")

# 10. 重複引用キーを正規化
for alias, canon in [("cbr_llm","cbr2024llm"),("gptcache2023","bang2023gptcache"),
                     ("lu2022orderbias","lu2022order"),("ralm_robust","yoran2024robust"),
                     ("smallmodel2024tools","toolcalling_scale")]:
    t = t.replace("\\cite{"+alias+"}", "\\cite{"+canon+"}")

# 11. §5 前置き: 件数を正確化 + n がスタディで異なる旨の注記
rep("across four tasks, seven models, and a fine-tuning comparison. Statistics (temperature-0 determinism, Wilson 95% CIs, paired McNemar) follow Section \\ref{sec:setup}-D.",
    "across four tasks, a scaling ladder up to 36B, an eight-model sub-1B sweep, and a fine-tuning comparison at two base sizes. Test-set sizes vary by study (Table \\ref{tab:tasks}) and zero-shot is re-measured within each study, so a model's accuracy can differ by a few points across tables; statistics (temperature-0 determinism, Wilson 95% CIs, paired McNemar) follow Section \\ref{sec:setup}-D.")

# 12. 件数の誤り（任意）
soft("the seven `exp_*.py` experiment runners", "the `exp_*.py` and `run_*.py` experiment runners")
soft("seven `exp_*.py`", "the `exp_*.py`")

Path("paper.md").write_text(t, encoding="utf-8")
print("OK. remaining Table~X:", t.count("Table~X"), "| alias keys left:",
      sum(t.count("\\cite{"+a+"}") for a in ["cbr_llm","gptcache2023","lu2022orderbias","ralm_robust","smallmodel2024tools"]),
      "| 73% headline gone:", "roughly 73% to 94" not in t)
