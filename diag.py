#!/usr/bin/env python3
"""format / think の組み合わせを切り分けて、効く構成を特定する。"""
import json, time, urllib.request

URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:0.8b"
SCHEMA = {"type": "object",
          "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                         "reason": {"type": "string"}},
          "required": ["sentiment", "reason"]}
MSG = [{"role": "user", "content": "次の文を分類: 「最高だった」"}]


def call(cfg, timeout=30):
    body = {"model": MODEL, "messages": MSG, "stream": False,
            "options": {"num_ctx": 4096, "temperature": 0}}
    body.update(cfg)
    if "num_predict" in cfg:
        body["options"]["num_predict"] = cfg.pop("num_predict")
        body.update({"options": body["options"]})
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"err": str(e), "dt": round(time.time() - t0, 1)}
    dt = round(time.time() - t0, 1)
    msg = d.get("message", {})
    content = msg.get("content", "")
    thinking = msg.get("thinking") or ""
    is_json = False
    try:
        json.loads(content); is_json = True
    except Exception:
        pass
    return {"dt": dt, "done": d.get("done_reason"), "eval": d.get("eval_count"),
            "is_json": is_json, "think_len": len(thinking),
            "content": content.replace("\n", " ")[:120]}


TESTS = [
    ("1 think=false, no format", {"think": False}),
    ("2 think=false, format=schema", {"think": False, "format": SCHEMA}),
    ('3 think=false, format="json"', {"think": False, "format": "json"}),
    ("4 format=schema, num_predict=256 (think default)", {"format": SCHEMA, "options": {"num_ctx": 4096, "temperature": 0, "num_predict": 256}}),
    ("5 /no_think in prompt + format", {"format": SCHEMA, "messages": [{"role": "user", "content": "/no_think 次の文を分類: 「最高だった」"}]}),
]

for name, cfg in TESTS:
    print(f"\n### {name}")
    print("   ", call(dict(cfg)))
