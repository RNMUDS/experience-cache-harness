"""タスク抽象 — 分類タスクを言語非依存・ラベル非依存に表現する。

ハーネス本体（検索/キャッシュ/ルール/分類器）は本抽象にのみ依存し、感情/トピック/質問種別など
任意の few-shot 分類へ横展開できる。プロンプト書式は言語(lang)で切替。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    labels: tuple[str, ...]
    instruction: str
    lang: str = "ja"  # ja | en

    def suffix(self) -> str:
        if len(self.labels) > 15:  # 多クラスは列挙しない（文脈逼迫回避）
            return ('Return ONLY JSON: {"label": "<one of the listed labels>"}.'
                    if self.lang == "en" else '{"label": "<候補のいずれか>"} のJSON「のみ」を返してください。')
        opts = "|".join(self.labels)
        if self.lang == "en":
            return f'Return ONLY JSON: {{"label": "{opts}"}}.'
        return f'{{"label": "{opts}"}} のJSON「のみ」を返してください。'


# 自作JP感情（controlled。テーマ構造つきで LOTO/汎化検査に使う）
SENTIMENT_JP = Task(
    "jp_sentiment_authored",
    ("positive", "negative", "neutral"),
    "次の日本語の文を positive / negative / neutral のいずれかに分類してください。",
    lang="ja",
)

# 公開ベンチ
TASKS: dict[str, Task] = {
    "sst2": Task("sst2", ("positive", "negative"),
                 "Classify the sentiment of the sentence as positive or negative.", lang="en"),
    "agnews": Task("agnews", ("world", "sports", "business", "scitech"),
                   "Classify the news text topic as world, sports, business, or scitech.", lang="en"),
    "trec": Task("trec", ("abbr", "entity", "description", "human", "location", "number"),
                 "Classify what the question asks about: abbr, entity, description, human, location, or number.",
                 lang="en"),
    "jp_sentiment": Task("jp_sentiment", ("negative", "neutral", "positive"),
                         "次の日本語の文を negative / neutral / positive のいずれかに分類してください。", lang="ja"),
}
