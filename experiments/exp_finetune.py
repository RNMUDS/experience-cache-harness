#!/usr/bin/env python3
"""E3: 再学習との比較 — 同一ベース(0.5B)で「キャッシュ(再学習なし)」vs「LoRA微調整」vs「埋め込み+ロジ回帰」。

同じ train ラベルを使う 3 方式を同一 test で比較:
  - cache    : cache_harness, ollama qwen2.5:0.5b（重み更新なし）
  - lora     : HF Qwen2.5-0.5B-Instruct を LoRA 微調整（MPS）
  - embed+LR : ollama nomic 埋め込み + sklearn ロジスティック回帰（教師あり下限）
LoRA 評価は verbalizer スコアリング（各候補ラベルの対数尤度の argmax）で頑健に。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import sys

import numpy as np

from cache_harness import metrics
from cache_harness.classifier import ArmConfig, Classifier
from cache_harness.config import Settings
from cache_harness.experiment import distill_from_train, make_cache, make_client, run_arm
from cache_harness.llm import warmup
from cache_harness.task import SENTIMENT_JP, TASKS, Task

from exp_multitask import load_task

OLLAMA_BASE = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:0.5b"
HF_MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
OUT_JSON = sys.argv[3] if len(sys.argv) > 3 else "finetune_results.json"
TASKS_E3 = {"jp_sentiment_authored": SENTIMENT_JP, "sst2": TASKS["sst2"],
            "agnews": TASKS["agnews"], "trec": TASKS["trec"]}


# ---- 方式1: キャッシュ（再学習なし）— LLM経路 / kNN / ハイブリッド + zeroshot 参照 ----
def run_cache(task: Task, split, client) -> dict:
    S = Settings(student_model=OLLAMA_BASE, retriever="embedding")  # 良い検索器
    cache = make_cache(split.train, S)
    rules, _ = distill_from_train(split.train, S, client, task)
    tr = [it.text for it in split.train]

    def ev(arm, items, settings=S, rules_=()):
        return run_arm(Classifier(cache, settings, client, task, rules_, arm), items, tr, settings)

    z = metrics.aggregate("z", ev(ArmConfig(), split.test)).accuracy
    cache_llm = metrics.aggregate("c", ev(ArmConfig(use_retrieval=True, use_rules=True), split.test, rules_=rules)).accuracy
    nn = metrics.aggregate("nn", ev(ArmConfig(mode="nn_copy", use_retrieval=True), split.test)).accuracy

    # ハイブリッド: tau_nn を dev で調整（高信頼は kNN, 低信頼は LLM）
    best_tau, best_dev = 0.6, -1.0
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        St = S.derive(tau_nn=tau)
        arm = ArmConfig(use_nn_route=True, use_retrieval=True, use_rules=True)
        dev_acc = metrics.aggregate("d", run_arm(Classifier(cache, St, client, task, rules, arm), split.dev, tr, St)).accuracy
        if dev_acc > best_dev:
            best_dev, best_tau = dev_acc, tau
    St = S.derive(tau_nn=best_tau)
    hyb = metrics.aggregate("h", run_arm(
        Classifier(cache, St, client, task, rules, ArmConfig(use_nn_route=True, use_retrieval=True, use_rules=True)),
        split.test, tr, St))
    nn_route_frac = hyb.route_counts.get("nn_route", 0) / hyb.n if hyb.n else 0.0
    return {"zeroshot": z, "cache_llm": cache_llm, "knn": nn, "hybrid": hyb.accuracy,
            "hybrid_tau": best_tau, "hybrid_nn_frac": nn_route_frac, "hybrid_llm_calls": hyb.total_llm_calls}


# ---- 方式2: 埋め込み + ロジスティック回帰 ----
def run_embed_lr(task: Task, split) -> dict:
    from sklearn.linear_model import LogisticRegression
    from cache_harness.embedding_retriever import _embed

    def emb(items):
        return np.array([_embed(it.text, "nomic-embed-text") for it in items])
    Xtr, ytr = emb(split.train), [it.label for it in split.train]
    Xte, yte = emb(split.test), [it.label for it in split.test]
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(np.mean([p == y for p, y in zip(pred, yte)]))
    return {"embed_lr": acc}


# ---- 方式3: LoRA 微調整（HF, MPS）----
def run_lora(task: Task, split) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.05, task_type="CAUSAL_LM"))
    model.train()

    def prompt_of(text: str) -> str:
        frame = "文:「{}」" if task.lang == "ja" else 'Text: "{}"'
        msg = [{"role": "user", "content": f"{task.instruction}\n{frame.format(text)}\nAnswer:"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    examples = [(prompt_of(it.text), " " + it.label) for it in split.train]
    for _epoch in range(3):
        for i in range(0, len(examples), 4):
            batch = examples[i:i + 4]
            loss = torch.tensor(0.0, device=dev)
            for prompt, target in batch:
                pids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                tids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
                ids = torch.cat([pids, tids], dim=1)
                labels = ids.clone()
                labels[0, : pids.shape[1]] = -100  # prompt 部分は損失から除外
                loss = loss + model(ids, labels=labels).loss
            (loss / len(batch)).backward()
            opt.step(); opt.zero_grad()

    model.eval()
    correct = 0
    with torch.no_grad():
        for it in split.test:
            pids = tok(prompt_of(it.text), return_tensors="pt").input_ids.to(dev)
            best, best_lp = None, -1e9
            for label in task.labels:
                tids = tok(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
                ids = torch.cat([pids, tids], dim=1)
                logits = model(ids).logits[0, pids.shape[1] - 1 : -1]
                lp = torch.log_softmax(logits, dim=-1).gather(1, tids[0].unsqueeze(1)).sum().item()
                if lp > best_lp:
                    best_lp, best = lp, label
            correct += int(best == it.label)
    return {"lora": correct / len(split.test)}


def main() -> None:
    client = make_client(path="llm_cache.json")
    print("[warmup]...", flush=True)
    warmup(client.inner, [OLLAMA_BASE])
    rows = []
    print(f"\n{'#'*92}\n# E3 再学習比較（ベース={OLLAMA_BASE} 0.5B / LoRA={HF_MODEL}）")
    print(f"  {'task':<14}{'zeroshot':>9}{'cacheLLM':>9}{'kNN':>7}{'hybrid':>8}{'embed+LR':>10}{'LoRA':>7}{'hyb LLM呼':>9}")
    for name, task in TASKS_E3.items():
        split = load_task(task)
        r = {"task": name}
        r.update(run_cache(task, split, client)); client.save()
        try:
            r.update(run_embed_lr(task, split))
        except Exception as e:  # noqa: BLE001
            print(f"  [embed_lr {name} FAILED: {type(e).__name__}: {e}]"); r["embed_lr"] = None
        try:
            r.update(run_lora(task, split))
        except Exception as e:  # noqa: BLE001
            print(f"  [lora {name} FAILED: {type(e).__name__}: {e}]"); r["lora"] = None
        rows.append(r)
        f = lambda k: f"{r[k]*100:.1f}%" if r.get(k) is not None else " n/a"
        print(f"  {name:<14}{f('zeroshot'):>9}{f('cache_llm'):>9}{f('knn'):>7}{f('hybrid'):>8}"
              f"{f('embed_lr'):>10}{f('lora'):>7}{r.get('hybrid_llm_calls',0):>7}")

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
