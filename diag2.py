#!/usr/bin/env python3
"""解決策の特定: thinking予算を増やす / lenientパース / 非thinkingモデル。"""
import json, re, time, urllib.request

URL = "http://localhost:11434/api/chat"
SCHEMA = {"type": "object",
          "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                         "reason": {"type": "string"}},
          "required": ["sentiment", "reason"]}


def call(model, content, fmt=None, think=None, num_predict=None, timeout=60):
    opts = {"num_ctx": 4096, "temperature": 0}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "stream": False, "options": opts}
    if fmt is not None:
        body["format"] = fmt
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"err": str(e), "dt": round(time.time() - t0, 1)}
    dt = round(time.time() - t0, 1)
    m = d.get("message", {})
    c = m.get("content", "")
    th = m.get("thinking") or ""
    raw_json = lenient_json = False
    try:
        json.loads(c); raw_json = True
    except Exception:
        pass
    mt = re.search(r"\{.*\}", c, re.DOTALL)
    if mt:
        try:
            json.loads(mt.group(0)); lenient_json = True
        except Exception:
            pass
    return {"dt": dt, "done": d.get("done_reason"), "eval": d.get("eval_count"),
            "raw_json": raw_json, "lenient_json": lenient_json, "think_len": len(th),
            "content": c.replace("\n", " ")[:140]}


CLS = "次の文をpositive/negative/neutralに分類: 「待ち時間が長すぎて最悪だった」"
CLS_JSON = CLS + '\n{"sentiment":"...","reason":"..."} のJSONだけ返す。'

print("### T6 qwen3.5:0.8b  format + think ON + num_predict=1024")
print("   ", call("qwen3.5:0.8b", CLS, fmt=SCHEMA, num_predict=1024, timeout=90))

print("\n### T7 qwen3.5:0.8b  think=false + no format + JSONプロンプト (lenient baseline)")
print("   ", call("qwen3.5:0.8b", CLS_JSON, think=False, num_predict=256))

print("\n### T8 qwen2.5-coder (非thinking) + format=schema")
print("   ", call("qwen2.5-coder:latest", CLS, fmt=SCHEMA, num_predict=256, timeout=90))

print("\n### T9 qwen2.5-coder (非thinking) + format=schema  別の文(positive)")
print("   ", call("qwen2.5-coder:latest", "次の文をpositive/negative/neutralに分類: 「対応が丁寧で大満足」", fmt=SCHEMA, num_predict=256, timeout=90))
