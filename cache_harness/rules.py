"""ルール蒸留（タスク非依存）— **train の誤分類からのみ**手引きを生成（test/ルーブリックは不可視）。

汎用化: ラベルや言語を決め打ちせず、混同ペアごとに「真の X を Y と誤る」を train から発見し、
X を Y から識別する手がかり語をデータ駆動で抽出して短い手引きにする。
これにより感情/トピック/質問種別など任意の分類へ横展開でき、採点基準漏洩の懸念も小さい。
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .task import Task

_JA_STOP = ("です", "ます", "まし", "した", "ありま", "ていま", "います", "おりま",
            "になっ", "ていた", "ました", "という", "ことが", "れていま", "から")
_EN_STOP = {"the", "and", "for", "are", "was", "you", "that", "this", "with", "what",
            "have", "has", "his", "her", "its", "from", "they", "their", "about", "which",
            "who", "whom", "does", "did", "can", "will", "would", "could", "should"}


def _has_content(tok: str) -> bool:
    """漢字・カタカナ・数字を含むか（純ひらがな/記号の機能的断片を除外し、内容語的な手がかりに絞る）。"""
    return any("一" <= c <= "鿿" or "゠" <= c <= "ヿ" or c.isdigit() for c in tok)


def _tokens(text: str, lang: str) -> list[str]:
    if lang == "en":
        return [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 3 and w not in _EN_STOP]
    grams = [text[i : i + 3] for i in range(len(text) - 2)] if len(text) >= 3 else [text]
    return [g for g in grams if _has_content(g)]


def mine_cues(pos: Sequence[str], neg: Sequence[str], lang: str, *, top: int = 6, min_df: int = 2) -> list[str]:
    """真クラス(pos)に多く、対照クラス(neg)に少ない識別的トークンを返す。"""
    pos_df: Counter[str] = Counter()
    for t in pos:
        pos_df.update(set(_tokens(t, lang)))
    neg_df: Counter[str] = Counter()
    for t in neg:
        neg_df.update(set(_tokens(t, lang)))
    scored: list[tuple[float, int, str]] = []
    for tok, df in pos_df.items():
        if df < min_df or not tok.strip():
            continue
        if lang != "en" and any(s in tok for s in _JA_STOP):
            continue
        score = df / (1 + neg_df.get(tok, 0))
        if score > 1.5:
            scored.append((score, df, tok))
    scored.sort(reverse=True)
    return [tok for _, _, tok in scored[:top]]


def distill_rules(train_eval: Sequence[tuple[str, str, str]], task: Task, *, max_rules: int = 4) -> tuple[str, ...]:
    """train_eval=[(text,gold,pred)] から混同ペア×手がかりの手引きを生成。"""
    errors = [(t, g, p) for (t, g, p) in train_eval if g != p and p]
    if not errors:
        return ()
    pairs = Counter((g, p) for (_, g, p) in errors)
    rules: list[str] = []
    for (gold, pred), _ in pairs.most_common():
        pos = [t for (t, g, _p) in train_eval if g == gold]
        neg = [t for (t, g, _p) in train_eval if g == pred]
        cues = mine_cues(pos, neg, task.lang)
        if not cues:
            continue
        joined = "、".join(cues) if task.lang != "en" else ", ".join(cues)
        if task.lang == "en":
            rules.append(f'If it contains [{joined}], prefer "{gold}" (often misread as "{pred}").')
        else:
            rules.append(f'[{joined}] を含むなら「{gold}」の可能性が高い（「{pred}」と誤りやすい）。')
        if len(rules) >= max_rules:
            break
    return tuple(rules)


def oracle_rules_jp() -> tuple[str, ...]:
    """自作JP感情の比較用 oracle（人手ルーブリック。漏洩検出の上限参照）。"""
    return (
        "満足・感動・称賛・おすすめなど良い評価を述べる文は positive。",
        "不満・落胆・怒り・故障・劣悪など悪い評価を述べる文は negative。",
        "評価や感情の語を含まず、客観的事実（時間・価格・寸法・容量・日時・場所・数量など）"
        "を述べるだけの文は neutral。",
    )
