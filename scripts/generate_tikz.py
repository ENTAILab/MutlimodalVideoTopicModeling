#!/usr/bin/env python3
"""Generate a LaTeX file with a TikZ visualization of collapsed topic clusters.

Usage:
  python3 scripts/generate_tikz.py --top 20

Outputs: data/output/collapsed_timeline.tex
"""
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).parent.parent / "data" / "output"
INPUT = ROOT / "collapsed_timeline.json"
OUT = ROOT / "collapsed_timeline.tex"

TEX_HEADER = r"""% Auto-generated TikZ timeline of collapsed topics
\documentclass[11pt]{article}
\usepackage[margin=1cm]{geometry}
\usepackage{tikz}
\usepackage{fontspec}
\setmainfont{DejaVu Sans}
\begin{document}
\centering
\section*{Collapsed Topic Timeline (Top clusters)}
\vspace{6pt}
\begin{tikzpicture}[x=1cm, y=1cm]
"""

TEX_FOOTER = r"""
\end{tikzpicture}
\end{document}
"""


def safe_tex(s: str) -> str:
    if not s:
        return ""
    return s.replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")


def render(clusters, max_width_cm=12.0):
    # compute max total_count
    max_count = max((c.get("total_count", 0) for c in clusters), default=1)
    lines = [TEX_HEADER]
    # vertical layout: one row per cluster
    y = 0
    row_h = 0.7
    for i, c in enumerate(clusters, 1):
        width = (c.get("total_count", 0) / max_count) * max_width_cm if max_count>0 else 0.1
        width = max(0.2, width)
        name = safe_tex(str(c.get("cluster_name") or f"cluster_{c.get('cluster_id')}"))
        words = safe_tex(" ".join((c.get("rep_words") or [])[:6]))
        count = c.get("total_count", 0)
        # draw rectangle and label
        lines.append(f"  % cluster {i}")
        lines.append(f"  \draw[fill=blue!35] (0, -{y}) rectangle ({width:.2f}, -{y+row_h});")
        lines.append(f"  \node[anchor=west] at ({width+0.2:.2f}, -{y+row_h/2}) {{{name} -- {count} -- {words}}};")
        y += row_h + 0.15
    lines.append(TEX_FOOTER)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20, help="Top N clusters to render")
    args = parser.parse_args()

    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    if not data:
        print("No clusters in input")
        return
    top = sorted(data, key=lambda x: x.get("total_count", 0), reverse=True)[: args.top]
    tex = render(top)
    OUT.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT} with top {len(top)} clusters")


if __name__ == '__main__':
    main()
