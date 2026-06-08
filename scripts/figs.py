#!/usr/bin/env python3
"""結果 JSON から論文用の図を生成（matplotlib Agg）。存在する JSON のみ処理。"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path("paper/figs"); FIGS.mkdir(exist_ok=True)


def _load(name):
    p = Path("results") / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_scaling():
    """異なる系統のモデルを線でつながない正直な散布図。色=系統, マーカ=タスク, 線は同一系統内のみ。"""
    data = _load("scaling_results.json")
    if not data:
        return
    from collections import defaultdict
    from matplotlib.lines import Line2D
    fams = sorted({r["family"] for r in data})
    cmap = {f: c for f, c in zip(fams, plt.cm.tab10.colors)}
    tasks = sorted({r["task"] for r in data})
    tmark = {t: m for t, m in zip(tasks, ["o", "s", "^", "D"])}
    tlabel = {"jp_sentiment_authored": "JP", "sst2": "SST-2"}

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    # 異なるモデルを線でつながない（散布図のみ）。各点が1モデル。
    for r in data:
        ax.scatter(r["params_b"], r["gain"] * 100, color=cmap[r["family"]], marker=tmark[r["task"]],
                   s=80, edgecolor="k", linewidth=0.5, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 5, 9, 20]); ax.set_xticklabels(["0.5", "1", "2", "3", "5", "9", "20"])
    ax.set_xlabel("model size (B parameters, log scale)"); ax.set_ylabel("cache gain over zero-shot (pp)")
    ax.set_title("Cache gain is large for small models and vanishes as models grow")
    fam_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[f], markeredgecolor="k",
                    label=f, markersize=8) for f in fams]
    task_h = [Line2D([0], [0], marker=tmark[t], color="k", linestyle="None",
                     label=tlabel.get(t, t), markersize=8) for t in tasks]
    leg1 = ax.legend(handles=fam_h, title="family", fontsize=7, loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=task_h, title="task", fontsize=7, loc="lower left")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(FIGS / "scaling_gain.png", dpi=140); plt.close(fig)


def fig_learning_curve():
    data = _load("learning_curve_results.json")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in data:
        rows = split["rows"]
        ax.errorbar([r["k"] for r in rows], [r["acc_mean"] * 100 for r in rows],
                    yerr=[r["acc_sd"] * 100 for r in rows], marker="o", capsize=3, label=split["split"])
    ax.set_xlabel("cache size K (exemplars only)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Non-monotonic cache-size law"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "learning_curve.png", dpi=140); plt.close(fig)


def fig_online():
    data = _load("online_results.json")
    if not data or "windows" not in data:
        return
    w = data["windows"]
    x = list(range(len(w)))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x, [r["acc"] * 100 for r in w], "o-", color="#1f77b4", label="accuracy")
    ax1.plot(x, [r["short_circuit_rate"] * 100 for r in w], "s--", color="#2ca02c", label="short-circuit %")
    ax1.set_xlabel("stream window"); ax1.set_ylabel("%"); ax1.set_xticks(x)
    ax1.set_xticklabels([r["window"] for r in w], rotation=45, fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["calls_per_q"] for r in w], "^:", color="#d62728", label="LLM calls/query")
    ax2.set_ylabel("LLM calls / query")
    ax1.set_title("Online lifelong learning: accuracy up, cost down")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="center right")
    ax1.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "online.png", dpi=140); plt.close(fig)


def fig_multitask():
    data = _load("main_results.json")
    if not data:
        return
    arms = ["zeroshot", "dyn_char", "dyn_embed", "rules_only", "cache_embed"]
    tasks = [d["task"] for d in data]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.16
    for i, arm in enumerate(arms):
        vals = [d["test"].get(arm, {}).get("acc", 0) * 100 for d in data]
        ax.bar([x + i * w for x in range(len(tasks))], vals, w, label=arm)
    ax.set_xticks([x + 2 * w for x in range(len(tasks))]); ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("accuracy (%)"); ax.set_title("Multi-task: retriever quality & rules vs zero-shot")
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.02))
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "multitask.png", dpi=140); plt.close(fig)


def fig_finetune():
    data = _load("finetune_results.json")
    if not data:
        return
    methods = ["zeroshot", "cache_llm", "knn", "hybrid", "embed_lr", "lora"]
    tasks = [d["task"] for d in data]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.14
    for i, m in enumerate(methods):
        vals = [(d.get(m) or 0) * 100 for d in data]
        ax.bar([x + i * w for x in range(len(tasks))], vals, w, label=m)
    ax.set_xticks([x + 1.5 * w for x in range(len(tasks))]); ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("accuracy (%)"); ax.set_title("Cache (no training) vs LoRA vs supervised")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "finetune.png", dpi=140); plt.close(fig)


def fig_classcount():
    data = _load("classcount_results.json")
    if not data:
        return
    data = sorted(data, key=lambda r: r["n_classes"])
    xs = [r["n_classes"] for r in data]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    zs_models = list(data[0]["zeroshot"].keys())
    for m in zs_models:
        ax.plot(xs, [r["zeroshot"][m] * 100 for r in data], "o--", label=f"zero-shot {m}")
    ax.plot(xs, [r["knn"] * 100 for r in data], "s-", color="#2ca02c", label="cache kNN (no LLM)")
    ax.plot(xs, [r["embed_lr"] * 100 for r in data], "^-", color="#9467bd", label="cache embed+LR (no LLM)")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("number of classes (log)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Tiny LLM collapses with class count; cache holds")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "classcount.png", dpi=140); plt.close(fig)


def fig_finetune_scale():
    d05 = _load("finetune_results.json"); d15 = _load("finetune15_results.json")
    if not (d05 and d15):
        return
    tasks = [d["task"] for d in d05]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, data, size in ((axes[0], d05, "0.5B"), (axes[1], d15, "1.5B")):
        methods = ["zeroshot", "cache_llm", "knn", "embed_lr", "lora"]
        w = 0.16
        for i, m in enumerate(methods):
            by = {x["task"]: x for x in data}
            vals = [(by[t].get(m) or 0) * 100 for t in tasks]
            ax.bar([x + i * w for x in range(len(tasks))], vals, w, label=m)
        ax.set_xticks([x + 2 * w for x in range(len(tasks))]); ax.set_xticklabels(tasks, fontsize=7, rotation=20)
        ax.set_title(f"base = {size}"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("accuracy (%)"); axes[0].legend(fontsize=8)
    fig.suptitle("Cache (no training) vs LoRA across base sizes")
    fig.tight_layout(); fig.savefig(FIGS / "finetune_scale.png", dpi=140); plt.close(fig)


def fig_external():
    data = _load("external_results.json")
    if not data:
        return
    data = sorted(data, key=lambda r: r["n_classes"])
    xs = [r["n_classes"] for r in data]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = [("zs_0.5b", "zero-shot 0.5B", "o--", "#aaa"), ("zs_0.8b", "zero-shot 0.8B", "o--", "#888"),
              ("dyn_embed_0.8b", "LLM+embed exemplars 0.8B", "d-", "#1f77b4"),
              ("knn", "cache kNN (no LLM)", "s-", "#2ca02c"),
              ("embed_lr", "cache embed+LR (no LLM)", "^-", "#9467bd")]
    for key, lab, st, col in series:
        ax.plot(xs, [r[key] * 100 for r in data], st, color=col, label=lab)
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("number of classes (log) — real-world datasets"); ax.set_ylabel("accuracy (%)")
    ax.set_title("External real-data validation: cache holds, tiny LLM does not")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "external_classcount.png", dpi=140); plt.close(fig)


def fig_embedders():
    data = _load("embedders_results.json")
    if not data:
        return
    data = sorted(data, key=lambda r: r["n_classes"])
    tasks = [r["task"].replace("ext_", "") for r in data]
    embs = list(data[0]["by_embedder"].keys())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(embs)
    for i, e in enumerate(embs):
        vals = [(r["by_embedder"].get(e, {}).get("embed_lr") or 0) * 100 for r in data]
        ax.bar([x + i * w for x in range(len(tasks))], vals, w, label=e.split("-")[0])
    ax.set_xticks([x + 0.4 for x in range(len(tasks))]); ax.set_xticklabels(tasks, fontsize=8, rotation=15)
    ax.set_ylabel("embed+LR accuracy (%)")
    ax.set_title("Cache (embed+LR) across embedding models — decent embedders cluster, tiny one lags")
    ax.legend(fontsize=8, title="embedder"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "embedders.png", dpi=140); plt.close(fig)


def fig_seeds():
    data = _load("seeds_results.json")
    if not data:
        return
    data = sorted(data, key=lambda r: r["n_classes"])
    tasks = [r["task"].replace("ext_", "") for r in data]
    x = range(len(tasks))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.errorbar(x, [r["embed_lr_mean"] * 100 for r in data], yerr=[r["embed_lr_sd"] * 100 for r in data],
                marker="^", capsize=4, label="cache embed+LR (mean±sd, 5 seeds)", color="#9467bd")
    ax.errorbar(x, [r["knn_mean"] * 100 for r in data], yerr=[r["knn_sd"] * 100 for r in data],
                marker="s", capsize=4, label="cache kNN (mean±sd)", color="#2ca02c")
    ax.errorbar(x, [r["zeroshot"] * 100 for r in data],
                yerr=[[(r["zeroshot"] - r["zeroshot_ci"][0]) * 100 for r in data],
                      [(r["zeroshot_ci"][1] - r["zeroshot"]) * 100 for r in data]],
                marker="o", capsize=4, label="zero-shot 0.8B (95% CI)", color="#888", ls="--")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{t}\n({r['n_classes']}c)" for t, r in zip(tasks, data)], fontsize=8)
    ax.set_ylabel("accuracy (%)"); ax.set_title("Statistical robustness: cache CI vs zero-shot CI (non-overlapping)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "seeds_ci.png", dpi=140); plt.close(fig)


def fig_genqa():
    g = _load("genqa_results.json")
    if not g:
        return
    methods = list(g["methods"].keys())
    labels = {"nn": "cache-NN", "zeroshot": "zero-shot", "rag": "LLM+RAG"}
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.38
    ax.bar([i - w / 2 for i in range(len(methods))], [g["methods"][m]["em"] * 100 for m in methods], w, label="EM")
    ax.bar([i + w / 2 for i in range(len(methods))], [g["methods"][m]["f1"] * 100 for m in methods], w, label="token-F1")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([labels.get(m, m) for m in methods])
    ax.set_ylabel("score (%)"); ax.set_title("Generative QA (web_questions): a scope limit")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "genqa.png", dpi=140); plt.close(fig)


def fig_sub1b():
    d = _load("sub1b_results.json")
    if not d:
        return
    tasks = d["tasks"]; models = sorted(d["models"], key=lambda r: r["params_b"])
    names = [m["model"].replace("SmolLM2", "Smol").replace("Qwen", "Q") for m in models]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.0), sharey=True)
    for ax, tn in zip(axes, tasks):
        zs = [m["per_task"][tn]["zeroshot"] * 100 for m in models]
        ca = [m["per_task"][tn]["cache_exemplar"] * 100 for m in models]
        x = range(len(models)); w = 0.38
        ax.bar([i - w / 2 for i in x], zs, w, label="zero-shot", color="#bbb")
        ax.bar([i + w / 2 for i in x], ca, w, label="cache (exemplars)", color="#1f77b4")
        floor = d["floors"][tn]["embed_lr"] * 100
        ax.axhline(floor, color="#d62728", ls="--", lw=1.4, label=f"embed+LR floor ({floor:.0f}%)")
        ax.set_title(f"{tn.replace('jp_sentiment_authored','JP')} ({d['floors'][tn]['n_classes']} cls)", fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Across sub-1B models: the non-parametric cache floor (red) is model-independent")
    fig.tight_layout(); fig.savefig(FIGS / "sub1b.png", dpi=140); plt.close(fig)


def fig_router():
    d = _load("router_results.json")
    if not d:
        return
    abbr = {"jp_sentiment_authored": "JP", "ext_imdb": "IMDB", "ext_dbpedia": "DBpedia"}
    names = [abbr.get(r["dataset"], r["dataset"].replace("ext_", "")) for r in d]
    x = range(len(d)); w = 0.38
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    mar = [r["auroc"]["lr_margin"] for r in d]
    sim = [r["auroc"]["top1_sim"] for r in d]
    axA.bar([i - w / 2 for i in x], mar, w, label="classifier confidence (margin)", color="#1f77b4")
    axA.bar([i + w / 2 for i in x], sim, w, label="nearest-neighbor similarity", color="#ff7f0e")
    axA.axhline(0.5, color="k", ls=":", lw=1, label="chance (0.5)")
    axA.set_ylim(0, 1); axA.set_ylabel("AUROC (predict no-LLM path correct)")
    axA.set_title("(a) Confidence separates known from novel", fontsize=9)
    axA.set_xticks(list(x)); axA.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axA.legend(fontsize=7, loc="lower right"); axA.grid(axis="y", alpha=.3)
    llm = [r["llm_only"] * 100 for r in d]; lr = [r["lr_only"] * 100 for r in d]
    rt = [r["router_costmin"]["acc"] * 100 for r in d]; orc = [r["oracle"] * 100 for r in d]
    frac = [r["router_costmin"]["llm_frac"] * 100 for r in d]
    bw = 0.2
    axB.bar([i - 1.5 * bw for i in x], llm, bw, label="LLM only", color="#bbb")
    axB.bar([i - 0.5 * bw for i in x], lr, bw, label="no-LLM only (cache-LR)", color="#2ca02c")
    axB.bar([i + 0.5 * bw for i in x], rt, bw, label="confidence router", color="#1f77b4")
    axB.bar([i + 1.5 * bw for i in x], orc, bw, label="oracle routing", color="#d62728", alpha=.55)
    for i, f in zip(x, frac):
        axB.text(i + 0.5 * bw, rt[i] + 1, f"{f:.0f}%", ha="center", va="bottom", fontsize=6)
    axB.set_ylabel("accuracy (%)"); axB.set_ylim(0, 108)
    axB.set_title("(b) Routing ≈ best single path at a few % LLM calls", fontsize=9)
    axB.set_xticks(list(x)); axB.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axB.legend(fontsize=7, loc="lower right"); axB.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "router.png", dpi=140); plt.close(fig)


def fig_cachecontent():
    d = _load("cachecontent_results.json")
    if not d:
        return
    order = ["gold-all", "gold-match", "failure", "self-resp"]
    fig, axes = plt.subplots(1, len(d), figsize=(4.3 * len(d), 4.0), sharey=True)
    for ax, rec in zip(axes, d):
        c = rec["contents"]
        x = range(len(order)); w = 0.38
        kn = [(c[o]["knn"] or 0) * 100 for o in order]
        lr = [(c[o]["embed_lr"] or 0) * 100 for o in order]
        ax.bar([i - w / 2 for i in x], kn, w, label="cache-kNN", color="#2ca02c")
        ax.bar([i + w / 2 for i in x], lr, w, label="cache-LR", color="#9467bd")
        ax.set_xticks(list(x)); ax.set_xticklabels(["gold", "gold\n(matched)", "failure", "self-\nresp"], fontsize=8)
        ax.set_title(f"{rec['task'].replace('jp_sentiment_authored','JP')} ({rec['n_classes']} cls)", fontsize=9)
        ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("What to cache: verified gold beats failures and raw (unverified) responses")
    fig.tight_layout(); fig.savefig(FIGS / "cachecontent.png", dpi=140); plt.close(fig)


def main():
    fig_router()
    fig_cachecontent()
    fig_sub1b()
    fig_scaling(); fig_learning_curve(); fig_online(); fig_multitask(); fig_finetune()
    fig_classcount(); fig_finetune_scale(); fig_external(); fig_embedders()
    fig_seeds(); fig_genqa()
    print("figs:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
