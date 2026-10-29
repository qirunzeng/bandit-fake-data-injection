#!/usr/bin/env python3
"""Build standalone vector PDFs from a corrected report using pdfLaTeX."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ("ucb-cost-horizon", "ucb-selection-ratio",
         "ts-cost-horizon", "ts-selection-ratio")


def build(report):
    engine = shutil.which("pdflatex")
    if not engine:
        raise SystemExit("PDF export requires pdfLaTeX with TikZ and PGFPlots.")
    shutil.copyfile(ROOT / "experiments/plot-style.tex", report / "fig/plot-style.tex")
    preamble = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{tikz,pgfplots}
\pgfplotsset{compat=1.18}
\input{fig/plot-style}
\begin{document}
\newsavebox{\exportbox}
\setlength{\linewidth}{206.4pt}
"""
    end = r"""\unskip}
\pdfpagewidth=\wd\exportbox
\pdfpageheight=\dimexpr\ht\exportbox+\dp\exportbox\relax
\pdfhorigin=0pt
\pdfvorigin=0pt
\shipout\box\exportbox
\end{document}
"""
    with tempfile.TemporaryDirectory(prefix="bandit-figures-") as temporary:
        temporary = Path(temporary)
        for name in (*PLOTS, "legend"):
            body = r"{\scriptsize\fdiLegendContents}" if name == "legend" else rf"\input{{fig/{name}}}"
            tex = temporary / (name + ".tex")
            tex.write_text(preamble + "\\sbox{\\exportbox}{%\n" + body + end)
            result = subprocess.run([engine, "-halt-on-error", "-interaction=nonstopmode",
                                     "-no-shell-escape", f"-output-directory={temporary}", str(tex)],
                                    cwd=report, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(result.stdout[-5000:])
            shutil.copyfile(temporary / (name + ".pdf"), report / "fig" / (name + ".pdf"))
            print("Built", name + ".pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results/report")
    build(parser.parse_args().report.resolve())
