#!/usr/bin/env python3
"""IEEE Access 公式クラス(ieeeaccess.cls)版を生成。

build_ieee を import すると IEEE 変換が走り、title/abstract/sanitize と _body_ieee.tex(本文)が用意される。
それを ieeeaccess の前付けで包む。ヘッダーロゴ(logo.png 等)はクラスが要求するため空プレースホルダを別途用意。
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])  # run relative to repo root
from pathlib import Path

import build_ieee  # 実行で IEEE 変換が走り _body_ieee.tex を生成、title/abstract/sanitize を公開

S = build_ieee.sanitize
title = S(build_ieee.title)
abstract = S(build_ieee.abstract)
KEYWORDS = ("Small language models, on-device inference, experience cache, "
            "retrieval-augmented classification, query routing, selective prediction, "
            "cost-efficient inference, edge AI.")

doc = r"""\documentclass{ieeeaccess}
\usepackage{graphicx,amsmath,amssymb,bm,booktabs,array,textcomp,url}
\usepackage{adjustbox}
\usepackage[hidelinks]{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\sloppy
\emergencystretch=3em
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\real}[1]{#1}

\history{Date of submission June 8, 2026.}
\doi{10.1109/ACCESS.2026.DOI}

\title{%s}
\author{\uppercase{Ryota Nakamura}\authorrefmark{1}}
\address[1]{Faculty of Data Science, Musashino University, Tokyo 135-8181, Japan (e-mail: ryota.nakamura@ds.musashino-u.ac.jp)}
\corresp{Corresponding author: Ryota Nakamura (e-mail: ryota.nakamura@ds.musashino-u.ac.jp).}
\markboth{R. Nakamura: Cache the Known, Ask Only the Novel}{R. Nakamura: Cache the Known, Ask Only the Novel}

\begin{document}
\maketitle

\begin{abstract}
%s
\end{abstract}

\begin{IEEEkeywords}
%s
\end{IEEEkeywords}

\input{_body_ieee.tex}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
""" % (title, abstract, KEYWORDS)

Path("paper/paper_access.tex").write_text(doc, encoding="utf-8")
print("paper_access.tex written.")
