#!/usr/bin/env python3
"""仕上げ: 要旨の長文分割・Related Work圧縮・位置づけ引用追加・用語定義の一元化。"""
from __future__ import annotations
from pathlib import Path

t = Path("paper.md").read_text(encoding="utf-8")
def rep(old, new):
    global t
    assert t.count(old) == 1, f"expected 1 occurrence, got {t.count(old)}: {old[:70]!r}"
    t = t.replace(old, new)

# 1. 要旨: 90語の長文を分割
rep("we establish that cache gains grow as models shrink, that a small curated cache (K≈8–32) beats a large noisy pool, and that the cache exposes two *complementary* prediction paths: a non-parametric classifier that is nearly invariant to class count",
    "we establish that cache gains grow as models shrink and that a small curated cache (K≈8–32) beats a large noisy pool. The cache exposes two *complementary* prediction paths: a non-parametric classifier, nearly invariant to class count")

# 2. Related Work §2.A: 三段の自己再掲を1文へ圧縮
rep("We differ in three ways. First, our target is the *sub-1B* regime rather than the large frozen agents ExpeL studies, and we show (Fig. \\ref{fig:scaling}) that the relative benefit of cached experience *grows as model size shrinks* — a scaling relationship not characterized by prior insight-distillation work. Second, we treat insight and exemplar memory as two *complementary, separately-ablated levers* and characterize an insight-vs-exemplar tradeoff that depends on retriever strength, whereas ExpeL combines them into a single retrieval-augmented prompt. Third, we mine rules strictly from the *training pool* (anti-leakage) and contrast auto-distilled rules against oracle hand-written rules, reporting that naive lexical rules can *hurt* tiny models — a failure mode not surfaced when the base model is large.",
    "We differ by targeting the *sub-1B* regime, where the benefit grows as the model shrinks (Fig. \\ref{fig:scaling}); by treating insight and exemplar memory as two separately-ablated levers whose balance shifts with retriever strength; and by mining rules strictly from the training pool, finding that naive lexical rules can *hurt* tiny models — a failure mode not seen at large scale.")

# 3. §2.C: 非パラメトリック分類の先行研究で位置づけ（kNN-LM / SetFit / prototypical）
rep("and we include a 1-NN-no-LLM copy baseline and a label-corruption probe to separate genuine reasoning over retrieved context from mere nearest-neighbor copying.",
    "and we include a 1-NN-no-LLM copy baseline and a label-corruption probe to separate genuine reasoning over retrieved context from mere nearest-neighbor copying. The non-parametric cache path itself connects to retrieval-augmented LMs that interpolate predictions from neighbors (kNN-LM \\cite{khandelwal2020knnlm}) and to few-shot text classifiers built on a frozen sentence encoder plus a lightweight head (SetFit \\cite{tunstall2022setfit}) or class prototypes (prototypical networks \\cite{snell2017prototypical}); we differ by using this predictor as one of two interchangeable cache paths and by quantifying when it should replace the tiny LLM entirely.")

# 4. 用語定義を §5 前置きに一元化
rep("statistics (temperature-0 determinism, Wilson 95% CIs, paired McNemar) follow Section \\ref{sec:setup}-D.",
    "statistics (temperature-0 determinism, Wilson 95% CIs, paired McNemar) follow Section \\ref{sec:setup}-D. We write **cache-kNN** for the nearest-neighbor majority predictor over the cache and **cache-LR** (= embed+LR) for a logistic-regression head on the embeddings; both bypass the LLM.")

Path("paper.md").write_text(t, encoding="utf-8")
print("revise4 applied. words:", len(t.split()))
