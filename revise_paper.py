#!/usr/bin/env python3
"""paper.md を学術体裁へ改訂: 図表の取捨選択、実キャプション+ラベル、相互参照修正、Table1(データセット表)追加。"""
from __future__ import annotations

from pathlib import Path

p = Path("paper.md")
t = p.read_text(encoding="utf-8")


def rep(old: str, new: str, n_required: int = 1):
    global t
    c = t.count(old)
    assert c >= n_required, f"NOT FOUND ({c}x): {old[:70]!r}"
    t = t.replace(old, new)


# ---------- 1. 図の削除（テーブルと重複/孤立/重複スケーリング） ----------
for embed in [
    "![Multi-task accuracy by arm](figs/multitask.png)",
    "![Scaling on JP](figs/scaling_jp_sentiment_authored.png)",
    "![Cache vs LoRA vs supervised (0.5B)](figs/finetune.png)",
    "![External real-data validation](figs/external_classcount.png)",
    "![Cache across embedders](figs/embedders.png)",
    "![Multi-seed confidence intervals](figs/seeds_ci.png)",
]:
    rep(embed, "")

# ---------- 2. 残す図に実キャプション+ラベル ----------
rep("![Scaling: cache gain shrinks as models grow](figs/scaling_gain.png)",
    "![Cache gain shrinks as models grow: accuracy gain (pp) of the best cache configuration over zero-shot "
    "versus base-model size, on JP sentiment and SST-2.\\label{fig:scaling}](figs/scaling_gain.png)")
rep("![Cache vs LoRA across base sizes](figs/finetune_scale.png)",
    "![Cache versus LoRA across base sizes (Qwen2.5-0.5B and 1.5B): the training-free embed+LR predictor matches "
    "or beats LoRA on most tasks at both sizes; the LLM path becomes competitive only at 1.5B.\\label{fig:ftscale}]"
    "(figs/finetune_scale.png)")
rep("![Online lifelong learning](figs/online.png)",
    "![Online lifelong learning from a cold cache: accuracy rises from 92% to 100% while the semantic short-circuit "
    "serves up to 88% of queries at zero LLM calls, cutting LLM calls per query roughly seven-fold and latency "
    "six-fold.\\label{fig:online}](figs/online.png)")
rep("![Non-monotonic cache-size law](figs/learning_curve.png)",
    "![Non-monotonic cache-size law: exemplar-only accuracy versus cache size K peaks at a small cache "
    "(K about 8 to 32) and declines as a weak retriever surfaces noisier neighbors.\\label{fig:lc}]"
    "(figs/learning_curve.png)")
rep("![Accuracy vs class count](figs/classcount.png)",
    "![Robustness to class count (2 to 77 classes): accuracy versus label-space size (log scale) for zero-shot "
    "tiny LLMs and the non-parametric cache predictors; the cache stays near 80% while the LLM collapses."
    "\\label{fig:classcount}](figs/classcount.png)")

# ---------- 3. 偽キャプション(イタリック) → 実キャプション(: ... \label) ----------
rep("*Table~\\ref{tab:multitask}. Accuracy (%) on held-out test, Qwen3.5-0.8B. † = significantly above zero-shot "
    "(McNemar p<0.05). The rightmost column foreshadows §C.*",
    ": Multi-task ablation on Qwen3.5-0.8B: held-out accuracy (%) by harness arm; the dagger marks arms "
    "significantly above zero-shot (paired McNemar, $p<0.05$). The last column (kNN, no LLM) foreshadows "
    "Section V-C. \\label{tab:multitask}")
rep("*Table~\\ref{tab:finetune}. Accuracy (%), Qwen2.5-0.5B. cache-kNN/hybrid/embed+LR require no weight updates; "
    "LoRA trains per-task adapters.*",
    ": Cache versus fine-tuning on Qwen2.5-0.5B with identical labeled data: accuracy (%) for zero-shot, cache-LLM, "
    "cache-kNN, dev-routed hybrid, embedding+logistic-regression, and LoRA. The non-parametric methods require no "
    "weight updates. \\label{tab:finetune}")
rep("*Table~\\ref{tab:classcount}. Accuracy (%) vs. label-space size. The cache (non-parametric) is robust to "
    "class count; the zero-shot tiny LLM is not.*",
    ": Robustness to label-space size (2 to 77 classes): accuracy (%) of zero-shot tiny LLMs versus the "
    "non-parametric cache predictors at a fixed embedding. \\label{tab:classcount}")
rep("*Table~\\ref{tab:external}. Accuracy (%) on six real-world datasets, zero training. Best per row in bold.*",
    ": External validation on six real-world datasets (zero training): accuracy (%) of zero-shot 0.5B/0.8B, the "
    "LLM-with-exemplars path, and the non-parametric cache; best per row in bold. \\label{tab:external}")
rep("*Table~\\ref{tab:embedders}. Cache (embed+LR) accuracy (%) across embedding models.*",
    ": Multi-embedder ablation: cache (embed+LR) accuracy (%) across four embedding models (45 MB to 1.2 GB) on "
    "seven datasets. Competent embedders cluster within about two points; the tiny embedder trails. "
    "\\label{tab:embedders}")
rep("*Table~\\ref{tab:seeds}. Accuracy (%): zero-shot with Wilson 95% CI; cache predictors as mean±sd over 5 "
    "train sub-samples.*",
    ": Multi-seed statistical robustness: zero-shot 0.8B accuracy (%) with Wilson 95% CI versus cache predictors "
    "as mean$\\pm$sd over five 80% train sub-samples; the intervals do not overlap. \\label{tab:seeds}")

# ---------- 4. 本文の参照を、削除した図 → 対応する表 に張り替え ----------
rep(", Fig.~\\ref{fig:multitask}", "")
rep(", Fig.~\\ref{fig:external}", "")
rep(", Fig.~\\ref{fig:embedders}", "")
rep(", Fig.~\\ref{fig:seeds}", "")
rep(" (Fig.~\\ref{fig:genqa})", "")
# §D 本文を、残した scaling_gain（gain図）に合わせて言い換え
rep("Fig.~\\ref{fig:scaling} plots zero-shot, LLM+embedding-exemplars, and cache-kNN against model size.",
    "Fig.~\\ref{fig:scaling} plots the accuracy gain of the best cache configuration over zero-shot against "
    "model size.")

# ---------- 5. ablation 表に実キャプション ----------
rep("\n\nTwo further studies sweep continuous factors:",
    "\n\n: Ablation arms (a to j): each configuration of the control plane and the specific factor it isolates, "
    "from the re-measured zero-shot baseline to teacher escalation and the copy/corruption controls. "
    "\\label{tab:ablation}\n\nTwo further studies sweep continuous factors:")

# ---------- 6. Table 1: データセット表を追加（§A 末尾, ### B. Models の前） ----------
DATASETS = (
    "| Dataset | Lang | \\#Cls | Task | Train/Dev/Test |\n"
    "|---|---|---|---|---|\n"
    "| JP sentiment (authored) | ja | 3 | sentiment | 102/34/34 same-theme; 90/30/50 LOTO |\n"
    "| SST-2 | en | 2 | sentiment | 180/60/150 |\n"
    "| AG News | en | 4 | topic | 180/60/148 |\n"
    "| TREC | en | 6 | question-type | 180/60/150 |\n"
    "| Banking77 | en | 77 | intent | 385/154/231 |\n"
    "| IMDB | en | 2 | sentiment | 50/16/150 |\n"
    "| Airline (Kaggle) | en | 3 | sentiment | 75/24/150 |\n"
    "| Tweet (Kaggle) | en | 3 | sentiment | 75/24/150 |\n"
    "| Yelp | en | 5 | rating | 125/40/150 |\n"
    "| Emotion | en | 6 | emotion | 150/48/150 |\n"
    "| DBpedia | en | 14 | topic | 350/112/140 |\n"
    "| web-questions | en | -- | generative QA | 300/--/150 |\n\n"
    ": Datasets and tasks: classification (and one generative QA) tasks behind a single task abstraction, "
    "spanning 2 to 77 classes, two languages, and authored / curated-benchmark / real-world (Kaggle) sources. "
    "\\label{tab:tasks}\n\n"
)
rep("### B. Models\n", DATASETS + "### B. Models\n")

# ---------- 7. dangling tab:online を fig:online に, Task~\ref を Table に ----------
rep("Tables~\\ref{tab:ablation}–\\ref{tab:online}", "Tables~\\ref{tab:ablation}–\\ref{tab:seeds} and the figures")
rep("Task~\\ref{tab:tasks}", "Table~\\ref{tab:tasks}")

# ---------- 8. 本文の "~\ref" を通常スペースに（pandocで ~ がチルダ字形になる問題） ----------
t = t.replace("~\\ref{", " \\ref{")

p.write_text(t, encoding="utf-8")
import re
print("figures embedded now:", len(re.findall(r"!\[", t)))
print("table captions (: ...\\label{tab):", t.count(": ") and len(re.findall(r"\n: .*\\label\{tab:", t)))
print("fig labels:", len(re.findall(r"\\label\{fig:", t)), "tab labels:", len(re.findall(r"\\label\{tab:", t)))
print("remaining fake captions:", len(re.findall(r"^\*Table", t, re.M)))
print("remaining ~\\ref:", t.count("~\\ref{"))
