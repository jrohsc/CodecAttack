#!/usr/bin/env bash
# Waveform-space attack + codec sidecar eval on all 50 S1 carriers (qwen2_audio).
# Uses run_s1_waveform_bundleA_multibitrate.py at eps=0.075 (SNR-matched
# to the latent eps=1.0 bundle, ~5.8 dB clean SNR).
#
# Resumable: both stages skip work that already exists on disk
#   - run_s1_waveform_bundleA_multibitrate.py skips when the *_attacked.wav exists.
#   - eval_mp3_aac_bundle.py skips when the *_codecs.json exists.
# So you can Ctrl-C this script at any point and re-run it to pick up where
# it left off. Same for the latent companion script (run_latent.sh).
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO"

PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
SCEN_DIR=01_finance_voice_agent
EPS=0.075
BUNDLE="0_all_combined/results_codec_robust_waveform/${SCEN_DIR}/qwen2_audio/eps_${EPS}"
LOG_DIR="0_all_combined/n50_latent_vs_waveform/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ts() { date "+%F %H:%M:%S"; }

echo "=== [$(ts)] WAVEFORM n=50  bundle=$BUNDLE ==="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "=== [$(ts)] STAGE 1: attack ==="
"$PY" -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --eps "$EPS" \
    2>&1 | tee -a "$LOG_DIR/waveform_attack.log"

echo
echo "=== [$(ts)] STAGE 2: MP3+AAC codec sidecar eval ==="
"$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio \
    2>&1 | tee -a "$LOG_DIR/waveform_sidecar.log"

echo
echo "=== [$(ts)] WAVEFORM DONE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l \
    | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l \
    | xargs -I{} echo "  codec sidecars: {}"
