#!/usr/bin/env python3
"""
PoC: 小型ローカルモデルで「構造化された正しい出力」をどう安定させるか。

当初は「format(JSON Schema文法)が効くか」を測る予定だったが、診断の結果:
  - qwen3.5:0.8b は thinking モデル。format(文法)は最終回答のみ拘束し、
    thinking フェーズは無拘束 → 0.8B は thinking が暴走し JSON に到達せず(ハング/空)。
  - ollama 0.21 では think=false にすると format が無視される。
そこで実戦的な構成を 3 つ比較する:

  A) 0.8B  think=false + JSONプロンプト (文法なし)   ← 本命
  B) 0.8B  think=false + format文法                  ← 無視される様子の確認
  C) coder 非thinking + format文法                   ← 構造保証が要る時の代替

指標: 有効率 / 素直率(後処理なしでパース可) / 正答率 / 平均遅延。依存なし。
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import json
import re
import sys
import time
import urllib.request

URL = "http://localhost:11434/api/chat"
SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "reason": {"type": "string"},
    },
    "required": ["sentiment", "reason"],
}
VALID = {"positive", "negative", "neutral"}

DATA = [
    ("対応がとても丁寧で大満足です。", "positive"),
    ("想像以上に使いやすくて感動しました。", "positive"),
    ("料理は絶品で、また来たいです。", "positive"),
    ("配送が早くて助かりました。", "positive"),
    ("デザインが美しく、買ってよかった。", "positive"),
    ("スタッフの笑顔が素敵で気持ちよかった。", "positive"),
    ("コスパ最高、文句なしです。", "positive"),
    ("期待を超える品質でした。", "positive"),
    ("待ち時間が長すぎて最悪だった。", "negative"),
    ("すぐに壊れてがっかりした。", "negative"),
    ("店員の態度が悪くて不快でした。", "negative"),
    ("値段の割に品質が低い。", "negative"),
    ("二度と利用したくない。", "negative"),
    ("説明と違っていて騙された気分。", "negative"),
    ("音がうるさくて使い物にならない。", "negative"),
    ("返品対応が遅くてイライラした。", "negative"),
    ("営業時間は午前9時から午後6時までです。", "neutral"),
    ("この製品は3色展開です。", "neutral"),
    ("会議は明日の14時に開催されます。", "neutral"),
    ("商品は3営業日以内に発送されます。", "neutral"),
    ("アプリのサイズは約50MBです。", "neutral"),
    ("店舗は駅から徒歩5分の場所にあります。", "neutral"),
    ("サポートは平日のみ対応しています。", "neutral"),
    ("パッケージには説明書が同梱されています。", "neutral"),
]

PROMPT = "次の日本語の文を positive / negative / neutral のいずれかに分類してください。\n文: 「{text}」"
JSON_SUFFIX = '\n{"sentiment": "positive|negative|neutral", "reason": "短い理由"} のJSON「のみ」を返す。'


def call(model, text, fmt=None, think=None, json_prompt=False, num_predict=200, timeout=60):
    content = PROMPT.format(text=text) + (JSON_SUFFIX if json_prompt else "")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "options": {"num_ctx": 4096, "temperature": 0, "num_predict": num_predict},
    }
    if fmt is not None:
        body["format"] = fmt
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d.get("message", {}).get("content", ""), time.time() - t0


def extract(s):
    try:
        return json.loads(s), "raw"
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), "extracted"
        except Exception:
            return None, "fail"
    return None, "fail"


def evaluate(content):
    obj, how = extract(content)
    if not isinstance(obj, dict):
        return False, False, None, "fail"
    s = obj.get("sentiment")
    valid = s in VALID
    return valid, (how == "raw" and valid), s, how


def run(name, **kw):
    n = len(DATA)
    valid = raw = correct = errors = 0
    lat = []
    fails = []
    for text, label in DATA:
        try:
            content, dt = call(text=text, **kw)
        except Exception as e:  # noqa: BLE001
            errors += 1
            fails.append((text, label, None, f"ERR:{e}"))
            continue
        lat.append(dt)
        v, rv, got, how = evaluate(content)
        valid += v
        raw += rv
        ok = v and got == label
        correct += ok
        if not ok:
            fails.append((text, label, got, content.replace("\n", " ")[:70]))
    return {"name": name, "n": n, "valid": valid / n, "raw": raw / n,
            "acc": correct / n, "lat": (sum(lat) / len(lat)) if lat else 0,
            "errors": errors, "fails": fails}


def pct(x):
    return f"{x*100:5.1f}%"


CONDITIONS = [
    ("A) 0.8B think=false +JSONプロンプト", dict(model="qwen3.5:0.8b", think=False, json_prompt=True)),
    ("B) 0.8B think=false +format文法", dict(model="qwen3.5:0.8b", think=False, fmt=SCHEMA)),
    ("C) coder 非thinking +format文法", dict(model="qwen2.5-coder:latest", fmt=SCHEMA, timeout=90)),
]


def main():
    print("[warmup]...", flush=True)
    try:
        call("qwen3.5:0.8b", "テスト", think=False, json_prompt=True)
        call("qwen2.5-coder:latest", "テスト", fmt=SCHEMA, timeout=90)
    except Exception as e:  # noqa: BLE001
        print(f"接続エラー: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for name, kw in CONDITIONS:
        print(f"[run] {name}", flush=True)
        results.append(run(name, **kw))

    print("\n" + "=" * 62)
    print(f"  PoC 結果 (n={results[0]['n']}, temp=0, num_ctx=4096)")
    print("=" * 62)
    print(f"  {'条件':<34}{'有効':>7}{'素直':>7}{'正答':>7}{'遅延s':>6}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['name']:<32}{pct(r['valid']):>7}{pct(r['raw']):>7}{pct(r['acc']):>7}{r['lat']:>6.2f}")
    print("=" * 62)

    for r in results:
        if r["fails"]:
            print(f"\n[{r['name']}] 不一致/無効 {len(r['fails'])}件:")
            for text, gold, got, info in r["fails"][:10]:
                print(f"  - 「{text}」 gold={gold} got={got} | {info}")

    with open("results/poc_results.json", "w", encoding="utf-8") as fp:
        json.dump([{k: v for k, v in r.items() if k != "fails"} for r in results],
                  fp, ensure_ascii=False, indent=2)
    print("\n保存: poc_results.json")


if __name__ == "__main__":
    main()
