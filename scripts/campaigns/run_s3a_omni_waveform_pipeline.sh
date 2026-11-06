#!/bin/bash
# Chained Phase A (calibration) -> B (full attack) -> C (quick eval) -> D (table)
# for Qwen2.5-Omni waveform-vs-latent on S3a.
set -euo pipefail
cd ${REPO_ROOT}

MODEL=qwen25_omni
LOG_DIR=0_all_combined/scripts
mkdir -p "$LOG_DIR"
LOG_A="$LOG_DIR/snr_calibrate_s3a_${MODEL}.log"
LOG_B="$LOG_DIR/run_s3a_waveform_${MODEL}.log"
LOG_C="$LOG_DIR/quick_eval_s3a_${MODEL}.log"
LOG_D="$LOG_DIR/build_table_s3a_${MODEL}.log"

CALIB_JSON="0_all_combined/results/03a_ai_detection_bypass/waveform_comparison/calibration/${MODEL}/snr_sweep.json"

echo "[pipeline] === Phase A: calibration (${MODEL}) ==="
conda run -n qwen-omni --no-capture-output python \
    0_all_combined/scripts/snr_calibrate_s3a_waveform.py \
    --target-model "$MODEL" \
    --eps 0.025 0.035 0.05 0.075 0.1 0.2 \
    --steps 500 2>&1 | tee "$LOG_A"

if [[ ! -f "$CALIB_JSON" ]]; then
    echo "[pipeline] FATAL: calibration did not produce $CALIB_JSON" >&2
    exit 1
fi

EPS_STAR=$(python -c "import json; print(json.load(open('$CALIB_JSON'))['recommended_eps'])")
echo "[pipeline] recommended eps: $EPS_STAR"

echo "[pipeline] === Phase B: full 40-pair attack (${MODEL}, eps=$EPS_STAR) ==="
conda run -n qwen-omni --no-capture-output python \
    0_all_combined/scripts/run_s3a_waveform_attack.py \
    --target-model "$MODEL" \
    --eps "$EPS_STAR" \
    --steps 1000 2>&1 | tee "$LOG_B"

echo "[pipeline] === Phase C: 12-channel quick eval (${MODEL}, eps=$EPS_STAR) ==="
conda run -n qwen-omni --no-capture-output python \
    0_all_combined/scripts/quick_eval_waveform_bundle.py \
    --model "$MODEL" \
    --eps "$EPS_STAR" 2>&1 | tee "$LOG_C"

echo "[pipeline] === Phase D: comparison table (${MODEL}) ==="
conda run -n codec-attack --no-capture-output python \
    0_all_combined/scripts/build_waveform_comparison_table.py \
    --model "$MODEL" 2>&1 | tee "$LOG_D"

echo "[pipeline] === DONE ==="
