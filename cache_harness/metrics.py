"""評価指標 — 精度・クラス別 recall・混同行列・遅延・LLM 呼数。

「精度が上がったか」を正直に測るための集計層。記憶と汎化を分けて見るため
`novel`（キャッシュに近傍が無い新規入力か）フラグも集計対象に持つ。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Outcome:
    """1 件の判定結果（不変）。"""

    text: str
    gold: str
    pred: str | None
    valid: bool          # enum 内の妥当ラベルが取れたか
    latency: float
    llm_calls: int       # この 1 件で消費した LLM 呼び出し回数（短絡=0）
    route: str           # short_circuit | student | teacher | nn_copy | ...
    novel: bool = True   # キャッシュに高類似の近傍が無い新規入力か（編集距離=検索と独立に判定）
    sim: float = 0.0     # 検索の最類似コサイン（層別用）
    edit_sim: float = 0.0  # train への最大編集距離比（検索と独立な記憶尺度）

    @property
    def correct(self) -> bool:
        return self.valid and self.pred == self.gold


def _wilson_low(correct: int, total: int, z: float = 1.96) -> float:
    """Wilson 信頼区間の下限（n が小さくても破綻しない）。"""
    if total == 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - margin) / denom


def _wilson_high(correct: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = correct / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center + margin) / denom


@dataclass
class Report:
    name: str
    n: int
    accuracy: float
    acc_ci: tuple[float, float]
    valid_rate: float
    per_class_recall: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_f1: dict[str, float]
    macro_f1: float
    confusion: dict[str, dict[str, int]]
    mean_latency: float
    median_latency: float
    p90_latency: float
    total_llm_calls: int
    novel_accuracy: float        # 新規入力のみの精度（汎化の指標）
    novel_n: int
    route_counts: dict[str, int]

    def summary_line(self, width: int = 30) -> str:
        lo, hi = self.acc_ci
        neu = self.per_class_recall.get("neutral", 0.0)
        return (
            f"  {self.name:<{width}}"
            f"{self.accuracy * 100:6.1f}%"
            f" [{lo * 100:4.1f},{hi * 100:5.1f}]"
            f"{self.macro_f1 * 100:7.1f}"
            f"{neu * 100:8.1f}%"
            f"{self.novel_accuracy * 100:8.1f}%"
            f"{self.median_latency:7.2f}s"
            f"{self.total_llm_calls:6d}"
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def aggregate(name: str, outcomes: list[Outcome]) -> Report:
    n = len(outcomes)
    empty = Report(name, 0, 0.0, (0.0, 0.0), 0.0, {}, {}, {}, 0.0, {}, 0.0, 0.0, 0.0, 0, 0.0, 0, {})
    if n == 0:
        return empty

    correct = sum(o.correct for o in outcomes)
    valid = sum(o.valid for o in outcomes)

    gold_total: Counter[str] = Counter(o.gold for o in outcomes)
    pred_total: Counter[str] = Counter(o.pred for o in outcomes if o.pred)
    hit: Counter[str] = Counter(o.gold for o in outcomes if o.correct)  # gold==pred の数
    classes = sorted(gold_total)

    recall = {c: hit[c] / gold_total[c] if gold_total[c] else 0.0 for c in classes}
    precision = {c: hit[c] / pred_total[c] if pred_total[c] else 0.0 for c in classes}
    f1 = {
        c: (2 * precision[c] * recall[c] / (precision[c] + recall[c])) if (precision[c] + recall[c]) else 0.0
        for c in classes
    }
    macro_f1 = sum(f1.values()) / len(f1) if f1 else 0.0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for o in outcomes:
        confusion[o.gold][o.pred or "∅"] += 1

    novel = [o for o in outcomes if o.novel]
    novel_correct = sum(o.correct for o in novel)
    latencies = [o.latency for o in outcomes]

    return Report(
        name=name,
        n=n,
        accuracy=correct / n,
        acc_ci=(_wilson_low(correct, n), _wilson_high(correct, n)),
        valid_rate=valid / n,
        per_class_recall=recall,
        per_class_precision=precision,
        per_class_f1=f1,
        macro_f1=macro_f1,
        confusion={g: dict(row) for g, row in confusion.items()},
        mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        median_latency=_percentile(latencies, 0.5),
        p90_latency=_percentile(latencies, 0.9),
        total_llm_calls=sum(o.llm_calls for o in outcomes),
        novel_accuracy=(novel_correct / len(novel)) if novel else 0.0,
        novel_n=len(novel),
        route_counts=dict(Counter(o.route for o in outcomes)),
    )


def header() -> str:
    return (
        f"  {'条件':<30}{'正答':>7}{'  95%CI':>12}{'mF1':>7}{'neu再現':>8}{'新規':>8}{'遅延':>7}{'LLM':>6}\n"
        "  " + "-" * 84
    )


def mcnemar_better(base: list[Outcome], variant: list[Outcome]) -> dict[str, float | int]:
    """対応ありの McNemar 検定（同一テスト集合での base vs variant）。

    b = base 正解 / variant 不正解, c = base 不正解 / variant 正解。
    片側 p（variant が改善している証拠）を二項検定で近似。
    """
    by_text_base = {o.text: o.correct for o in base}
    b = c = 0
    for o in variant:
        bc = by_text_base.get(o.text)
        if bc is None:
            continue
        if bc and not o.correct:
            b += 1
        elif (not bc) and o.correct:
            c += 1
    n = b + c
    # 片側二項検定 P(X >= c | p=0.5): 改善方向の有意性
    p = 1.0 if n == 0 else sum(math.comb(n, i) for i in range(c, n + 1)) / (2 ** n)
    return {"b_base_only": b, "c_variant_only": c, "p_one_sided": p}
