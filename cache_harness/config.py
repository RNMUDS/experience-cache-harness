"""ハーネス設定 — 全ノブを 1 つの不変 Settings に集約（ハードコード禁止）。

レビュー指摘への対応: tau / k / escalation 閾値は **dev でのみ**調整し、ここで凍結してから
test を 1 回だけ測る。test を見てチューニングしない（HP リーク防止）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

LABELS: tuple[str, str, str] = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class Settings:
    # モデル
    student_model: str = "qwen3.5:0.8b"
    teacher_model: str = "qwen2.5-coder:latest"
    embed_model: str = "nomic-embed-text"
    # ollama 呼び出し（PoC 由来の安全値）
    num_ctx: int = 4096
    temperature: float = 0.0
    student_num_predict: int = 128
    teacher_num_predict: int = 200
    # 検索
    retriever: str = "char_ngram"  # char_ngram | embedding
    ngram_sizes: tuple[int, ...] = (2, 3)
    k: int = 4
    balance_kshot: bool = True       # k-shot のクラス均衡（多数ラベルバイアス対策）
    min_exemplar_sim: float = 0.0    # この類似度未満の exemplar は注入しない（0.8B の文脈ノイズ抑制）
    # 短絡 / エスカレーション（dev で調整して凍結）
    tau_short_circuit: float = 0.92  # この類似度以上なら LLM を呼ばず即答
    tau_escalate: float = 0.30       # 最類似がこれ未満なら教師へ（冷スタート/無関係文脈対策）
    tau_nn: float = 0.60             # 近傍ルート: 最類似がこれ以上なら NN 多数決を採用（小型LLMより信頼）
    # 新規性（検索指標と独立: 編集距離比）
    dup_edit_ratio: float = 0.80     # difflib 比がこれ以上なら near-duplicate（記憶）扱い
    # プロンプト予算
    max_prompt_chars: int = 1600     # 0.8B の文脈逼迫を避ける上限（num_ctx 内に収める）
    max_rules: int = 4

    def derive(self, **changes: object) -> "Settings":
        """一部だけ変えた新インスタンス（ミューテーションしない）。"""
        return replace(self, **changes)
