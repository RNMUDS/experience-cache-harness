"""ラベル抽出 — 寛容パース + enum 検証。文法に依存せずハーネス側で構造を保証する（タスク非依存）。"""
from __future__ import annotations

from collections.abc import Sequence

from .llm import extract_json


def parse_label(content: str, labels: Sequence[str]) -> tuple[str | None, bool, bool]:
    """(label, valid, ambiguous) を返す。

    valid     = labels 内のラベルを取得できた。
    ambiguous = JSON で素直に取れず、フォールバック走査に頼った/失敗した（エスカレーション判断用）。
    """
    valid_set = set(labels)
    obj = extract_json(content)
    if isinstance(obj, dict):
        # LLM 出力は信頼しない: label が list/dict/数値で返ることがあるため str に限定して検証。
        label = obj.get("label")
        if isinstance(label, str) and label in valid_set:
            return label, True, False
        # 互換: 旧 "sentiment" キー
        label = obj.get("sentiment")
        if isinstance(label, str) and label in valid_set:
            return label, True, False
    lowered = content.lower()
    found = [lab for lab in labels if lab.lower() in lowered]
    if len(found) == 1:
        return found[0], True, True
    return None, False, True
