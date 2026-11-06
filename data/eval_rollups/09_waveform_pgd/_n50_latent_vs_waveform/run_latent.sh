#!/usr/bin/env bash
# Latent-space attack + codec sidecar eval on all 50 S1 carriers (qwen2_audio).
#
# Resumable: both stages skip work that already exists on disk
#   - run_s1_attack_bundleA.py skips a pair when its *_attacked.wav exists.
#   - eval_mp3_aac_bundle.py skips a wav when its *_codecs.json exists.
# So you can Ctrl-C this script at any point and re-run it to pick up where
# it left off. Same for the waveform companion script (run_waveform.sh).
#
# Designed to be run from the codec_attack project root.
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO"

PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
SCEN_DIR=01_finance_voice_agent
EPS=1.0
BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/qwen2_audio/eps_${EPS}"
LOG_DIR="0_all_combined/n50_latent_vs_waveform/logs"
mkdir -p "$LOG_DIR"

# Single GPU available (A100 80GB). Pin to it explicitly so two parallel
# runs don't both try GPU 0 unintentionally on a multi-GPU host.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CODEC_ATTACK_PY="$PY"

ts() { date "+%F %H:%M:%S"; }

echo "=== [$(ts)] LATENT n=50  bundle=$BUNDLE ==="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "=== [$(ts)] STAGE 1: attack ==="
"$PY" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen2_audio --eps "$EPS" \
    2>&1 | tee -a "$LOG_DIR/latent_attack.log"

echo
echo "=== [$(ts)] STAGE 2: MP3+AAC codec sidecar eval ==="
"$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio \
    2>&1 | tee -a "$LOG_DIR/latent_sidecar.log"

echo
echo "=== [$(ts)] LATENT DONE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l \
    | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l \
    | xargs -I{} echo "  codec sidecars: {}"
