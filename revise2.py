#!/usr/bin/env python3
"""簡潔化パス: 図表キャプションを短く、§D本文から不要な弁明を削除。"""
from __future__ import annotations

import re
from pathlib import Path

p = Path("paper.md")
t = p.read_text(encoding="utf-8")

FIGCAPS = {
    "fig:scaling": "Cache gain over zero-shot versus model size (one point per model; color = family, marker = task). The gain is large below 1B and near zero from 1.5B to 36B.",
    "fig:ftscale": "Cache versus LoRA at two base sizes (Qwen2.5-0.5B and 1.5B). Training-free embed+LR matches or beats LoRA on most tasks; the LLM path is competitive only at 1.5B.",
    "fig:online": "Online lifelong learning from a cold cache: accuracy rises to 100% as the short-circuit serves more queries at zero LLM cost, cutting calls and latency several-fold.",
    "fig:lc": "Non-monotonic cache-size effect: exemplar-only accuracy peaks at a small cache (about 8 to 32 examples) and declines for larger pools.",
    "fig:classcount": "Accuracy versus class count (2 to 77, log scale): the zero-shot tiny LLM collapses while the non-parametric cache stays near 80%.",
}
TABCAPS = {
    "tab:tasks": "Datasets and tasks (2 to 77 classes; authored, benchmark, and real-world sources).",
    "tab:ablation": "Ablation arms and the factor each isolates.",
    "tab:multitask": "Multi-task accuracy (%) by arm (Qwen3.5-0.8B); daggers mark gains significant over zero-shot (paired McNemar, $p<0.05$).",
    "tab:finetune": "Cache versus fine-tuning on Qwen2.5-0.5B with the same data: accuracy (%); non-parametric methods need no training.",
    "tab:classcount": "Accuracy (%) versus class count (2 to 77).",
    "tab:external": "External validation on six real-world datasets (zero training): accuracy (%); best per row in bold.",
    "tab:embedders": "Cache (embed+LR) accuracy (%) across four embedding models.",
    "tab:seeds": "Statistical robustness: zero-shot accuracy with Wilson 95% CI versus cache predictors (mean$\\pm$sd over five sub-samples).",
}

out = []
for line in t.split("\n"):
    m = re.match(r"^!\[.*\\label\{(fig:[a-z0-9_]+)\}\]\((figs/[^)]+)\)\s*$", line)
    if m and m.group(1) in FIGCAPS:
        out.append(f"![{FIGCAPS[m.group(1)]}\\label{{{m.group(1)}}}]({m.group(2)})")
        continue
    m = re.match(r"^: .*\\label\{(tab:[a-z0-9_]+)\}\s*$", line)
    if m and m.group(1) in TABCAPS:
        out.append(f": {TABCAPS[m.group(1)]} \\label{{{m.group(1)}}}")
        continue
    out.append(line)
t = "\n".join(out)

# §D 本文: 線で結ばない弁明を削除し簡潔化
old_d = ("Fig. \\ref{fig:scaling} shows, for each model, the accuracy gain of the best cache configuration over "
         "zero-shot as a function of size. Because the models span four families (Qwen2.5, Qwen3.5/3.6, Llama-3.2, "
         "Gemma), we plot each model as an individual point (color = family) and deliberately do *not* connect the "
         "points with lines: the models differ in family and architecture, so a connecting line would falsely imply "
         "a continuous trend across distinct models and conflate family with size. The gain is large for the "
         "smallest models — +38.2 pp at 0.5B and +29.4 pp at 1.2B on JP, +20.8 pp at 0.5B on SST-2 — and collapses "
         "to essentially zero from 1.5B onward, remaining at +0.0 pp at 8B, 9.7B, and even **36B**, where capable "
         "models already reach ~100% zero-shot on these tasks. We could not run models beyond 36B on our single "
         "64 GB machine (a 20B model also failed to load), but the gain has already saturated to zero several "
         "billion parameters earlier, so larger models fall outside the small-model regime the harness targets. "
         "The cache's kNN path is model-independent by construction (a flat 64.7% on JP), so the shrinking gain "
         "reflects the *rising zero-shot baseline* of larger models, not a weakening cache. Finally, the saturation "
         "size is task-dependent: on the harder many-class tasks of Section \\ref{sec:results}-H even the 0.8B model "
         "is far from saturated, so there the cache keeps helping at larger sizes — the sub-1B focus here is a "
         "property of these easy 2–3-class scaling tasks, not a universal ceiling.")
new_d = ("Fig. \\ref{fig:scaling} reports the accuracy gain of the best cache over zero-shot for each model "
         "(color = family, marker = task). The gain is large for sub-1B models (+38.2 pp at 0.5B and +29.4 pp at "
         "1.2B on JP; +20.8 pp at 0.5B on SST-2) and falls to essentially zero from 1.5B through 36B, where capable "
         "models already score ~100% zero-shot on these tasks. The kNN path is model-independent (a flat 64.7% on "
         "JP), so the shrinking gain reflects the rising zero-shot baseline, not a weakening cache. The saturation "
         "size is task-dependent: on the many-class tasks of Section \\ref{sec:results}-H even the 0.8B model is far "
         "from saturated. Models beyond 36B were infeasible on our hardware, but the gain had long since saturated.")
assert old_d in t, "DELTA paragraph not found"
t = t.replace(old_d, new_d)

p.write_text(t, encoding="utf-8")
print("captions simplified:", sum(c in t for c in FIGCAPS.values()) + sum(c in t for c in TABCAPS.values()),
      "/", len(FIGCAPS) + len(TABCAPS))
print("§D simplified:", new_d[:40] in t)
