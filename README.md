# Cache the Known, Ask Only the Novel: A Confidence-Routed Harness for Sub-1B Language Models

A deterministic harness that makes **frozen sub-1-billion-parameter language models** useful on classification **without fine-tuning** — not by strengthening the model, but by **separating known from novel inputs** and routing each to where it is cheapest and most reliable. Known inputs are answered by a non-parametric predictor (embedding kNN / logistic regression) that never calls the LLM; only novel inputs are sent to the LLM. The known/novel decision is made by a **confidence gate** that we validate (mean AUROC 0.83) rather than assume.

The accompanying manuscript ("Cache the Known, Ask Only the Novel: A Confidence-Routed Harness for Sub-1B Language Models") is included as `paper.pdf` / `paper_ieee.pdf` / `paper_access.pdf`.

## Key Results

- **Validated routing gate.** The non-parametric classifier's confidence predicts whether its no-LLM answer is correct at mean **AUROC 0.83** across seven datasets; nearest-neighbor similarity alone is near chance (0.56).
- **Accuracy up, cost down.** Routing matches or beats the best single path (**85.6%** vs 83.9% no-LLM, 57.0% LLM) while invoking the LLM on only **~4%** of inputs, approaching an oracle ceiling (92.1%).
- **The no-LLM path dominates.** Across **11 datasets × 8 sub-1B models (88 cells)**, the non-parametric path averages 71.9% vs 26–29% for every LLM-path variant, is nearly invariant to class count (≈80% from 2 to 77 classes), and rivals LoRA at zero training cost.
- **Scope.** Claims are limited to closed-set classification; a generative-QA pilot marks the boundary where the no-LLM path stops transferring.

## Repository Layout

```
.
├── cache_harness/            # core harness library (Python standard-library only)
│   ├── llm.py                # ollama client (think:false, lenient JSON) + memoizing cache
│   ├── classifier.py         # control plane: short-circuit / route / escalate
│   ├── cache.py              # experience cache (verified text->label cases)
│   ├── retriever.py          # char n-gram TF-IDF retriever
│   ├── embedding_retriever.py # embedding retriever (+ cosine)
│   ├── supervised.py         # non-parametric predictors: kNN, logistic regression
│   ├── rules.py              # rule distillation from the model's own train failures
│   ├── metrics.py            # accuracy, Wilson CI, paired McNemar, macro-F1
│   ├── dataset.py            # authored Japanese sentiment set + same-theme / LOTO splits
│   └── task.py, prompt.py, parsing.py, config.py, experiment.py
├── experiments/             # exp_*.py (router, comprehensive, scaling, sub1b, ...) + run_*.py drivers
├── scripts/                 # fetch_*.py (data), figs.py, build_pandoc_md.py, build_ieee.py, build_access.py
├── results/                 # *_results.json — experiment outputs (metrics only)
├── paper/                   # manuscript: paper.md (source), references.bib, hdr.tex, ieeeaccess.cls
│   ├── figs/                #   generated figures (PNG)
│   ├── paper.pdf, paper_ieee.pdf, paper_access.pdf
│   └── cover_letter.md, cover_letter.pdf
├── docs/                    # research notes (findings, harness catalog)
├── LICENSE
└── README.md
```

Scripts are run **from the repository root**; each experiment/script changes its working
directory to the repo root automatically, so relative paths (`results/`, `paper/figs/`, `data/`)
resolve regardless of where you invoke them.

> **Data note.** This repository does **not** redistribute third-party benchmark datasets. The `data/` directory and the response cache (`llm_cache.json`) are git-ignored; run the `fetch_*.py` scripts to obtain the public datasets from their original sources. The authored Japanese set lives in `cache_harness/dataset.py`.

## Method Summary

The harness wraps a frozen sub-1B model with an append-only **experience cache** of verified `(text, label)` cases and exposes two complementary prediction paths over the same cache:

1. **No-LLM path** — an embedding kNN or logistic-regression classifier (`cache-kNN` / `cache-LR`) that never invokes the language model. Strong and nearly class-count-invariant in-distribution.
2. **LLM path** — the frozen model conditioned on retrieved exemplars and distilled rules. Needed for genuinely novel (out-of-distribution) inputs, where it generalizes and the no-LLM path drops.

A **confidence gate** routes each input: high-confidence (known) → no-LLM path (zero LLM calls); low-confidence (novel) → LLM. The gate signal is the classifier's top-two probability margin, validated as a known/novel detector. All model calls are at temperature 0 and memoized, so runs are deterministic and resumable.

## How to Run

Requires **Python 3.11** and a local **ollama** server. The core library (`cache_harness/`) is standard-library only; baselines/figures additionally use `scikit-learn`, `numpy`, `matplotlib`, and (for LoRA) `transformers`/`peft`/`torch`; PDF builds use `pandoc` + `xelatex`/`pdflatex`.

```bash
ollama serve                 # in another terminal

# 1) fetch public datasets (not redistributed here) -> data/
python3 scripts/fetch_datasets.py    # SST-2, AG News
python3 scripts/fetch_extra.py       # TREC
python3 scripts/fetch_banking.py     # Banking77
python3 scripts/fetch_external.py    # IMDB / Yelp / Emotion / DBpedia / tweet
python3 scripts/fetch_kaggle.py      # Kaggle airline (needs ~/.kaggle/kaggle.json)

# 2) run experiments (each writes results/*_results.json)
python3 experiments/exp_router.py        # confidence-gated routing (AUROC, Pareto, oracle)
python3 experiments/exp_comprehensive.py # 11 datasets x 8 sub-1B models (88 cells)
python3 experiments/exp_scaling.py       # cache gain vs model size (0.5B-36B)
python3 experiments/exp_sub1b.py         # 8 sub-1B models
python3 experiments/exp_finetune.py      # cache vs LoRA vs embed+LR
# ... exp_classcount, exp_external, exp_embedders, exp_seeds, exp_controlled,
#     exp_cachecontent, exp_failv2_semantic, exp_success_rules, run_online, run_learning_curve

# 3) build the paper (figures -> paper/figs, then LaTeX from paper/)
python3 scripts/figs.py
python3 scripts/build_ieee.py            # -> paper/paper_ieee.tex
python3 scripts/build_access.py          # -> paper/paper_access.tex (official IEEE Access class)
( cd paper && pdflatex paper_ieee.tex && bibtex paper_ieee && pdflatex paper_ieee.tex && pdflatex paper_ieee.tex )
```

## Local LLM Stack

Served locally via **ollama**. Students (sub-1B): `smollm2:135m`, `gemma3:270m`, `smollm:360m`, `smollm2:360m`, `qwen2:0.5b`, `qwen2.5:0.5b`, `qwen3:0.6b`, `qwen3.5:0.8b`. Scaling ladder extends to a 36B model. Embeddings: `nomic-embed-text` (also `bge-m3`, `mxbai-embed-large`, `all-minilm` for the embedder ablation). Rule distillation / teacher: `qwen3.5:9b`.

## Citation

A manuscript reporting this work is under review at *IEEE Access*. Citation details will be added here upon publication. Provisional:

```bibtex
@misc{nakamura2026cache,
  title  = {Cache the Known, Ask Only the Novel: A Confidence-Routed Harness for Sub-1B Language Models},
  author = {Nakamura, Ryota},
  year   = {2026},
  note   = {Manuscript under review, IEEE Access}
}
```

## Acknowledgments

Built on the open-source [ollama](https://ollama.com) runtime and the IEEE Access LaTeX template. Public benchmarks are used under their respective licenses (see each `fetch_*.py`).

## License

Released under the [MIT License](LICENSE). Third-party datasets are **not** redistributed; obtain them via the `fetch_*.py` scripts under their original licenses.
