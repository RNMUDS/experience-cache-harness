#!/usr/bin/env python3
"""IEEEtran 2段組版を生成 — paper.md から title/abstract を抽出し、本文を pandoc(latex,natbib)へ。
longtable を table*+tabular(全幅)に、figure を figure*(全幅)に変換して2段組で破綻しないようにする。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
import re
import subprocess
from pathlib import Path

_UNI = {"≈": r"$\approx$", "∼": r"$\sim$", "τ": r"$\tau$", "×": r"$\times$", "≥": r"$\ge$",
        "≤": r"$\le$", "≠": r"$\ne$", "→": r"$\rightarrow$", "±": r"$\pm$", "–": "--", "—": "---",
        "“": "``", "”": "''", "‘": "`", "’": "'", "…": r"\ldots{}", "≫": r"$\gg$", "≪": r"$\ll$",
        "ϕ": r"$\phi$", "φ": r"$\phi$", "θ": r"$\theta$", "λ": r"$\lambda$", "μ": r"$\mu$",
        "∈": r"$\in$", "∪": r"$\cup$", "∅": r"$\emptyset$", "≈": r"$\approx$",
        "⊥": r"$\bot$", "⊤": r"$\top$", "∧": r"$\wedge$", "∨": r"$\vee$", "¬": r"$\neg$",
        "∀": r"$\forall$", "∃": r"$\exists$", "≡": r"$\equiv$", "←": r"$\leftarrow$",
        "⇒": r"$\Rightarrow$", "↦": r"$\mapsto$", "∝": r"$\propto$", "•": r"$\bullet$",
        "≥": r"$\ge$", "→": r"$\rightarrow$", "ρ": r"$\rho$", "θ": r"$\theta$", "Δ": r"$\Delta$",
        "−": r"$-$", "✓": r"\checkmark", "✗": r"$\times$", "†": r"$\dagger$", "•": r"$\bullet$",
        "↔": r"$\leftrightarrow$", "⁻": r"\textsuperscript{-}", "⁰": r"\textsuperscript{0}",
        "¹": r"\textsuperscript{1}", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
        "⁴": r"\textsuperscript{4}", "⁵": r"\textsuperscript{5}", "⁶": r"\textsuperscript{6}",
        "⁷": r"\textsuperscript{7}", "⁸": r"\textsuperscript{8}", "⁹": r"\textsuperscript{9}"}


def sanitize(s: str) -> str:
    for k, v in _UNI.items():
        s = s.replace(k, v)
    return s


md = Path("paper/paper.md").read_text(encoding="utf-8")
lines = md.splitlines()

# --- title（最初の '# '）---
title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Experience-Cache Harness")

# --- abstract（## Abstract 〜 次の '## '）---
abs = []
in_abs = False
for l in lines:
    if l.strip().lower().startswith("## abstract"):
        in_abs = True; continue
    if in_abs and l.startswith("## "):
        break
    if in_abs:
        abs.append(l)
abstract = " ".join(x.strip() for x in abs if x.strip())

# --- 本文 md（title 行と Abstract 節を除去）---
body_lines, skip_abs = [], False
for l in lines:
    if l.startswith("# ") and "Experience-Cache Harness" in l:
        continue
    if l.strip().lower().startswith("## abstract"):
        skip_abs = True; continue
    if skip_abs and l.startswith("## ") and not l.strip().lower().startswith("## abstract"):
        skip_abs = False
    if skip_abs:
        continue
    body_lines.append(l)

# 連続画像を独立段落に分離（横並び溢れ防止）
sep, img = [], re.compile(r"^!\[[^\]]*\]\(figs/")
for l in body_lines:
    if img.match(l.strip()) and sep and sep[-1].strip() != "":
        sep.append("")
    sep.append(l)
    if img.match(l.strip()):
        sep.append("")
Path("paper/_body.md").write_text("\n".join(sep), encoding="utf-8")

# --- pandoc: body.md -> latex fragment (natbib citations) ---
subprocess.run(["pandoc", "paper/_body.md", "-t", "latex", "--natbib", "--wrap=preserve",
                "--syntax-highlighting=none", "--shift-heading-level-by=-1", "-o", "paper/_body.tex"], check=True)
tex = Path("paper/_body.tex").read_text(encoding="utf-8")


def conv_longtable(m: str) -> str:
    """longtable→ full-width table*+tabular。多行colspec/minipageセルを剥がし単純 l/r 列に。foot順序も補正。"""
    block = m.group(0)
    head = re.search(r"\\toprule.*?\n(.*?)\\\\", block, re.DOTALL)
    ncols = (head.group(1).count("&") + 1) if head else 3
    spec = "l" + "r" * (ncols - 1)
    # pandoc が付けた \caption{...}（\label を含む）を抽出して退避（番号・タイトル保持）
    cap_m = re.search(r"\\caption\{(?:[^{}]|\{[^{}]*\})*\}", block)
    cap = cap_m.group(0) if cap_m else ""
    if cap:
        block = re.sub(re.escape(cap) + r"\s*\\tabularnewline", "", block)
        block = block.replace(cap, "")
    # begin{longtable}{...多行colspec...} を \toprule 直前まで丸ごと除去
    block = re.sub(r"\\begin\{longtable\}\[\]\{.*?(?=\\toprule)", "@@BEGIN@@\n", block, count=1, flags=re.DOTALL)
    # 重複ヘッダ + longtable 専用コマンドを除去
    block = re.sub(r"\\endfirsthead.*?\\endhead", "", block, flags=re.DOTALL)
    for tok in (r"\endhead", r"\endlastfoot", r"\endfoot"):
        block = block.replace(tok, "")
    block = block.replace(r"\noalign{}", "")
    # minipage セル包みを除去（中身は残す。幅引数のネスト {} に対応）
    block = re.sub(r"\\begin\{minipage\}\[[bt]\]\{(?:[^{}]|\{[^{}]*\})*\}\s*(\\raggedright|\\centering)?\s*", "", block)
    block = block.replace(r"\end{minipage}", "")
    # \bottomrule は longtable では本文より前に来るので全削除し、末尾に1つ付け直す
    block = block.replace(r"\bottomrule", "")
    # 列が少なければカラム内 table、多ければ全幅 table*。はみ出す時だけ adjustbox で縮小。
    if ncols <= 5:
        env, target = "table", r"\columnwidth"
    else:
        env, target = "table*", r"\textwidth"
    cap_line = ("\n" + cap) if cap else ""
    begin = (r"\begin{%s}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}" % env + cap_line + "\n" +
             r"\begin{adjustbox}{max width=%s}" % target + "\n" + r"\begin{tabular}{" + spec + "}")
    end = (r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{adjustbox}" + "\n" + r"\end{%s}" % env)
    block = block.replace(r"\end{longtable}", end)
    block = block.replace("@@BEGIN@@", begin)
    return block


WIDE_FIGS = {"finetune_scale", "multitask", "embedders", "finetune", "sub1b", "cachecontent", "router"}  # 横長: 全幅 figure*


def conv_figure(m: str) -> str:
    """画像 figure を、狭いものはカラム内(\\columnwidth)、横長のものは全幅 figure*(\\textwidth) に。"""
    blk = m.group(0)
    name_m = re.search(r"figs/([a-z_0-9]+)\.png", blk)
    name = name_m.group(1) if name_m else ""
    wide = name in WIDE_FIGS
    width = r"\textwidth" if wide else r"\columnwidth"
    blk = re.sub(r"\\includegraphics\[[^\]]*\]", lambda _m: r"\includegraphics[width=" + width + "]", blk)
    if wide:
        blk = blk.replace(r"\begin{figure}", r"\begin{figure*}[tbp]").replace(r"\end{figure}", r"\end{figure*}")
    else:
        blk = blk.replace(r"\begin{figure}", r"\begin{figure}[tbp]")
    return blk


# 画像 figure を カラム内/全幅 に振り分け（verbatim を figure* 化する前に実施）
tex = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", conv_figure, tex, flags=re.DOTALL)
# 表を変換（列数で カラム内 table / 全幅 table*、はみ出しは adjustbox 縮小）
tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", conv_longtable, tex, flags=re.DOTALL)
# アルゴリズム擬似コードは全幅 figure* + \scriptsize（横に長いので）
tex = tex.replace(r"\begin{verbatim}", r"\begin{figure*}[tbp]\scriptsize\begin{verbatim}")
tex = tex.replace(r"\end{verbatim}", r"\end{verbatim}\end{figure*}")
tex = sanitize(tex)
Path("paper/_body_ieee.tex").write_text(tex, encoding="utf-8")

# --- main.tex ---
main = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{graphicx,amsmath,amssymb,bm,booktabs,array,url,hyperref}
\usepackage{morefloats}\extrafloats{200}
\usepackage{adjustbox}
\usepackage{dblfloatfix}
\usepackage[section]{placeins}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\hypersetup{hidelinks}
\sloppy
\emergencystretch=3em
\renewcommand{\floatpagefraction}{0.7}
\setcounter{totalnumber}{10}\setcounter{topnumber}{6}\setcounter{bottomnumber}{6}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\real}[1]{#1}
\title{%s}
\author{\IEEEauthorblockN{Ryota Nakamura}\IEEEauthorblockA{Musashino University\\ryota.nakamura@ds.musashino-u.ac.jp}}
\begin{document}
\maketitle
\begin{abstract}
%s
\end{abstract}
\input{_body_ieee.tex}
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
""" % (sanitize(title), sanitize(abstract))
Path("paper/paper_ieee.tex").write_text(main, encoding="utf-8")
print("paper_ieee.tex written. title len", len(title), "abstract len", len(abstract),
      "longtables converted:", tex.count(r"\begin{table*}"), "figures:", tex.count(r"\begin{figure*}"))
