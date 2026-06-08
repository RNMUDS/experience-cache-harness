#!/usr/bin/env python3
"""原稿ワークフロー出力(JSON)から安定章を抽出し paper.md に組版（エージェントのメタ前後書きを除去）。"""
from __future__ import annotations

import json
from pathlib import Path

SRC = "/private/tmp/claude-501/-Users-rn-Documents-Harness/353d1fc0-fd54-4bb4-9f3d-1c7a1e06a4df/tasks/wabvdmgru.output"
META_MARKERS = ("I have written", "I wrote", "**Report", "Key facts", "Key grounding",
                "Key points", "A few notes", "I now have", "I'll write", "I have all grounded")


def clean(text: str) -> str:
    lines = text.splitlines()
    # 最初の見出しまで読み飛ばす
    start = next((i for i, l in enumerate(lines) if l.lstrip().startswith("#")), 0)
    lines = lines[start:]
    # 末尾メタ説明を除去（メタマーカ行以降を切る）
    end = len(lines)
    for i, l in enumerate(lines):
        if any(l.lstrip().startswith(m) or m in l[:40] for m in META_MARKERS):
            end = i
            break
    body = "\n".join(lines[:end]).rstrip()
    # 末尾の孤立した "---" を除去
    while body.endswith("---"):
        body = body[:-3].rstrip()
    return body


def main() -> None:
    data = json.loads(Path(SRC).read_text(encoding="utf-8"))["result"]
    title = ("# Experience-Cache Harness: Making Frozen Sub-1B Language Models "
             "Generalize Without Fine-Tuning\n")
    parts = [title,
             clean(data["intro"]),
             "## 2. Related Work\n" + _strip_h1(clean(data["related"])),
             "## 3. Method\n" + _strip_h1(clean(data["method"])),
             "## 4. Experimental Setup\n" + _strip_h1(clean(data["setup"])),
             "## 5. Results\n\n_(自動生成: 実験結果から後段で挿入)_\n",
             "## 6. Discussion\n\n_(後段)_\n",
             "## 7. Conclusion\n\n_(後段)_\n",
             "## References\n\n_(\\cite{} キーを BibTeX へマップ)_\n"]
    Path("paper.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print("paper.md written:", len(Path('paper.md').read_text(encoding='utf-8')), "chars")


def _strip_h1(text: str) -> str:
    """章本文の先頭にある重複 H1/タイトル見出しを除去（番号は assemble 側で付与済み）。"""
    lines = text.splitlines()
    out = [l for l in lines if not (l.strip().startswith("# ") or
           l.strip() in ("# RELATED WORK", "# METHOD", "# Experimental Setup"))]
    return "\n".join(out).strip()


if __name__ == "__main__":
    main()
