#!/usr/bin/env bash
# Build the PDF and the arXiv upload tarball. arXiv does not run BibTeX,
# so the generated dee.bbl must be uploaded with the .tex.
set -euo pipefail
cd "$(dirname "$0")"
python3 figures/make_figures.py
latexmk -pdf -interaction=nonstopmode dee.tex
rm -f arxiv.tar.gz
tar czf arxiv.tar.gz dee.tex dee.bbl figures/*.pdf
echo "wrote arxiv.tar.gz:"; tar tzf arxiv.tar.gz
