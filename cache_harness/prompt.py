"""プロンプト組立 — 唯一の真実の場所。タスク(言語/ラベル)で書式を切替。アブレーション各アームは
「何を渡すか」だけで差をつける。0.8B の文脈逼迫を避けるため exemplar を文字数予算で上限管理。
exemplar 順序は「最類似を最後（クエリ直前）」に置く（recency バイアスを味方に）。
"""
from __future__ import annotations

from collections.abc import Sequence

from .task import Task

_FRAME = {
    "ja": {"rules": "判定の手引き:", "ex": "例:", "shot": '文:「{t}」→ {l}', "q": '文:「{q}」'},
    "en": {"rules": "Guidelines:", "ex": "Examples:", "shot": 'Text: "{t}" -> {l}', "q": 'Text: "{q}"'},
}


def _frame(task: Task) -> dict[str, str]:
    return _FRAME.get(task.lang, _FRAME["ja"])


def _rules_block(rules: Sequence[str], fr: dict[str, str]) -> str:
    if not rules:
        return ""
    return fr["rules"] + "\n" + "\n".join(f"- {r}" for r in rules) + "\n"


def _exemplars_block(exemplars: Sequence[tuple[str, str]], max_chars: int, fr: dict[str, str]) -> str:
    """exemplars は (text,label) の昇順（最類似が末尾）想定。予算内で末尾優先に残す。"""
    if not exemplars:
        return ""
    lines: list[str] = []
    used = 0
    for text, label in reversed(exemplars):
        line = fr["shot"].format(t=text, l=label)
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    lines.reverse()
    return (fr["ex"] + "\n" + "\n".join(lines) + "\n") if lines else ""


def build_messages(
    query: str,
    task: Task,
    *,
    rules: Sequence[str] = (),
    exemplars: Sequence[tuple[str, str]] = (),
    max_exemplar_chars: int = 1200,
) -> list[dict[str, str]]:
    fr = _frame(task)
    parts = [task.instruction, ""]
    rb = _rules_block(rules, fr)
    if rb:
        parts.append(rb)
    eb = _exemplars_block(exemplars, max_exemplar_chars, fr)
    if eb:
        parts.append(eb)
    parts.append(fr["q"].format(q=query))
    parts.append(task.suffix())
    return [{"role": "user", "content": "\n".join(parts)}]
