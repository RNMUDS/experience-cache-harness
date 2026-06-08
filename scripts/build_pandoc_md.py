#!/usr/bin/env python3
"""paper.md を pandoc 用に前処理 → paper_pandoc.md。

\cite{a,b} → [@a; @b]、ぶら下がり \ref をテキスト化、画像に幅指定を付与。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import re
from pathlib import Path

src = Path("paper/paper.md").read_text(encoding="utf-8")


def cite_repl(m: str) -> str:
    keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
    return "[" + "; ".join("@" + k for k in keys) + "]"


# \cite{...} -> [@...; @...]
out = re.sub(r"\\cite\{([^}]*)\}", cite_repl, src)
# 参照表現をテキスト化（markdown には \label が無いため）
out = re.sub(r"Table~\\ref\{[^}]*\}", "the table", out)
out = re.sub(r"(Fig\.|Figure)~?\\ref\{[^}]*\}", "the figure", out)
out = re.sub(r"Section~\\ref\{[^}]*\}", "the relevant section", out)
out = re.sub(r"§~?\\ref\{[^}]*\}", "the relevant section", out)
out = re.sub(r"\\ref\{[^}]*\}", "", out)
# 画像: 幅指定 + 各画像を独立段落に（連続画像が横並びになって溢れるのを防ぐ）
lines = out.split("\n")
fixed = []
img = re.compile(r"^!\[[^\]]*\]\(figs/[^)]+\)")
for ln in lines:
    if img.match(ln.strip()):
        if fixed and fixed[-1].strip() != "":
            fixed.append("")  # 直前が空行でなければ空行を挿入
        fixed.append(re.sub(r"(\)){?[^}]*}?$", r"\1{width=68%}", ln.strip()))
        fixed.append("")      # 画像の後に必ず空行
    else:
        fixed.append(ln)
out = "\n".join(fixed)

Path("paper/paper_pandoc.md").write_text(out, encoding="utf-8")
print("paper_pandoc.md written:", len(out), "chars;",
      "cites:", out.count("[@"), "images:", out.count("{width=82%}"))
