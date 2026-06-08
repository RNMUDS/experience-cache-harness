"""ollama クライアント — 小型モデル向けに調整済み。

PoC (poc-findings.md) の教訓を実装で固定する:
  - 既定の巨大文脈は厳禁 → `num_ctx` を必ず明示（小さく）。
  - 0.8B は常に `think:false`（高速・決定論）。thinking 暴走でハングするため。
  - 出力契約は文法に依存しない。プロンプト指示 + 寛容パース + 構造化エラーで保証する。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://localhost:11434/api/chat"


@dataclass(frozen=True)
class ChatResult:
    """1 回のチャット呼び出しの結果（不変）。"""

    content: str
    latency: float
    model: str
    ok: bool = True
    error: str | None = None
    done_reason: str | None = None


@dataclass(frozen=True)
class OllamaClient:
    """ollama /api/chat の薄いラッパ。小型モデル向け既定値を内蔵。"""

    url: str = DEFAULT_URL
    num_ctx: int = 4096
    temperature: float = 0.0
    num_predict: int = 256
    timeout: float = 90.0

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        think: bool | None = False,
        fmt: dict[str, Any] | None = None,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        """1 ターンのチャット。例外は握りつぶさず ChatResult.error に載せて返す（呼び出し側が回復判断）。"""
        options = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "num_predict": self.num_predict if num_predict is None else num_predict,
        }
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "options": options}
        if fmt is not None:
            body["format"] = fmt
        if think is not None:
            body["think"] = think

        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout if timeout is None else timeout) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ChatResult("", time.time() - t0, model, ok=False, error=f"{type(exc).__name__}: {exc}")

        content = payload.get("message", {}).get("content", "") or ""
        return ChatResult(
            content=content,
            latency=time.time() - t0,
            model=model,
            ok=True,
            done_reason=payload.get("done_reason"),
        )


@dataclass
class CachingClient:
    """応答メモ化クライアント。temp=0 は決定論なので、同一プロンプトは 1 回だけ計算する。

    これは研究対象の「経験キャッシュ」とは別物（再現性と高速化のための応答キャッシュ）。
    ディスク永続化で再実行を即時化できる。
    """

    inner: OllamaClient
    store: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: str | None = None
    hits: int = 0
    misses: int = 0

    @staticmethod
    def _key(model: str, messages: list[dict[str, str]], num_predict: int | None) -> str:
        payload = json.dumps([model, messages, num_predict], ensure_ascii=False, sort_keys=True)
        return payload

    def chat(self, model: str, messages: list[dict[str, str]], *, think: bool | None = False,
             fmt: dict[str, Any] | None = None, num_predict: int | None = None,
             timeout: float | None = None) -> ChatResult:
        key = self._key(model, messages, num_predict)
        cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            return ChatResult(cached["content"], cached.get("latency", 0.0), model, ok=cached.get("ok", True),
                              error=cached.get("error"), done_reason=cached.get("done_reason"))
        self.misses += 1
        res = self.inner.chat(model, messages, think=think, fmt=fmt, num_predict=num_predict, timeout=timeout)
        if res.ok:  # 失敗はキャッシュしない（再試行余地を残す）
            self.store[key] = {"content": res.content, "ok": True, "done_reason": res.done_reason,
                               "latency": res.latency}
        return res

    def real_latency(self, model: str, messages: list[dict[str, str]], num_predict: int | None) -> float:
        """キャッシュ済みの実測レイテンシ（あれば）。指標用。"""
        cached = self.store.get(self._key(model, messages, num_predict))
        return cached.get("latency", 0.0) if cached else 0.0

    def load(self) -> None:
        if self.path:
            p = Path(self.path)
            if p.exists():
                self.store = json.loads(p.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self.path:
            Path(self.path).write_text(json.dumps(self.store, ensure_ascii=False), encoding="utf-8")


def extract_json(text: str) -> dict[str, Any] | None:
    """寛容な JSON 抽出。素直に loads → 失敗したら最初の {...} を救出。失敗時 None。"""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def warmup(client: OllamaClient, models: list[str]) -> dict[str, bool]:
    """各モデルを 1 回叩いてロード済みにする。{model: ok} を返す。"""
    status: dict[str, bool] = {}
    for model in models:
        res = client.chat(model, [{"role": "user", "content": "ping"}], num_predict=4)
        status[model] = res.ok
    return status


# 設定の派生（不変オブジェクトの安全な複製）
def with_overrides(client: OllamaClient, **changes: Any) -> OllamaClient:
    """クライアント設定を一部だけ変えた新しいインスタンスを返す（ミューテーションしない）。"""
    return replace(client, **changes)
