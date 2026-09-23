#!/usr/bin/env bash
# One command: numbers -> Fig 5 data -> figures -> PAPER_DELTA.md   (no API calls)
set -e
cd "$(dirname "$0")/.."
python3 analysis/recompute.py
python3 analysis/build_fig5_data.py
python3 analysis/make_figs.py
python3 analysis/paper_delta.py
echo; echo "outputs: analysis/out/{recompute.json,fig5_data.json,fig_stats.json,PAPER_DELTA.md,figs/}"
