#!/bin/bash
# Phase B (full attack) -> C (quick eval) -> D (table) for Qwen2.5-Omni
# Phase A calibration skipped; eps=0.05 picked by user (see snr_sweep.json note).
set -euo pipefail
cd ${REPO_ROOT}

MODEL=qwen25_omni
EPS=0.05
LOG_DIR=0_all_combined/scripts
LOG_B="$LOG_DIR/run_s3a_waveform_${MODEL}.log"
LOG_C="$LOG_DIR/quick_eval_s3a_${MODEL}.log"
LOG_D="$LOG_DIR/build_table_s3a_${MODEL}.log"

export PYTHONUNBUFFERED=1

echo "[pipeline] === Phase B: full 40-pair attack (${MODEL}, eps=${EPS}) ==="
conda run -n qwen-omni --no-capture-output python \
    0_all_combined/scripts/run_s3a_waveform_attack.py \
    --target-model "$MODEL" \
    --eps "$EPS" \
    --steps 1000 2>&1 | tee "$LOG_B"

echo "[pipeline] === Phase C: 12-channel quick eval (${MODEL}, eps=${EPS}) ==="
conda run -n qwen-omni --no-capture-output python \
    0_all_combined/scripts/quick_eval_waveform_bundle.py \
    --model "$MODEL" \
    --eps "$EPS" 2>&1 | tee "$LOG_C"

echo "[pipeline] === Phase D: comparison table (${MODEL}) ==="
conda run -n codec-attack --no-capture-output python \
    0_all_combined/scripts/build_waveform_comparison_table.py \
    --model "$MODEL" 2>&1 | tee "$LOG_D"

echo "[pipeline] === DONE ==="
