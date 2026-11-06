#!/usr/bin/env bash
# Rebuild the n=50 latent-vs-waveform LaTeX table once both run_latent.sh
# and run_waveform.sh have completed (or partially completed -- the
# aggregator intersects the two bundles, so the table reflects whichever
# pairs are present in BOTH).
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO"

PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

# 1. Markdown roll-up + ASCII tables (latent vs waveform across codecs).
"$PY" -u 0_all_combined/scripts/compare_s1_bundleA_latent_vs_waveform.py

# 2. Drop-in LaTeX replacement for paper-neurips/tables/tab_latent_vs_waveform.tex
#    (matches the existing simple 14-column booktabs style, just with the
#    new n and updated cell numbers).
"$PY" -u 0_all_combined/n50_latent_vs_waveform/build_n50_table.py

# 3. Also produce the richer multicolumn variant from build_s1_codec_latex.py
#    (kept for the s1_latent_vs_waveform_codecs.tex include path).
"$PY" -u 0_all_combined/scripts/build_s1_codec_latex.py
