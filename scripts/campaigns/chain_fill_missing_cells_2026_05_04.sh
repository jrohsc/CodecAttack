#!/usr/bin/env bash
# Fill the missing cells in the S1 latent-vs-waveform comparison table:
#   - MP3 / AAC @ {16, 24} kbps for both bundles (held-out, needs compute)
#   - Opus @ 96 kbps for both bundles (held-out, needs compute)
#
# Existing sidecars already cover MP3/AAC @ {32,64,96,128,192} and Opus @
# {16,24,32,64,128,192}. eval_mp3_aac_bundle.py is now merge-safe but we
# still re-run the full grid with --overwrite so all 7 bitrates are written
# in one consistent batch (cheap, ~10 min/bundle).
#
# All four jobs use qwen2_audio + the codec-attack env, one GPU, serial.

set -euo pipefail

ROOT=${REPO_ROOT}/0_all_combined
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python

LATENT_DIR=results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0
WAVEFORM_DIR=results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075

MP3_GRID=16,24,32,64,96,128,192
AAC_GRID=16,24,32,64,96,128,192
OPUS_NEW=96

cd "$ROOT"

echo "=========================================================="
echo "[$(date '+%F %T')] Step 1/4: MP3+AAC full grid -- LATENT bundle"
echo "=========================================================="
"$PY" scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$LATENT_DIR" \
    --target-model qwen2_audio \
    --mp3-bitrates "$MP3_GRID" \
    --aac-bitrates "$AAC_GRID" \
    --overwrite

echo "=========================================================="
echo "[$(date '+%F %T')] Step 2/4: MP3+AAC full grid -- WAVEFORM bundle"
echo "=========================================================="
"$PY" scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$WAVEFORM_DIR" \
    --target-model qwen2_audio \
    --mp3-bitrates "$MP3_GRID" \
    --aac-bitrates "$AAC_GRID" \
    --overwrite

echo "=========================================================="
echo "[$(date '+%F %T')] Step 3/4: Opus 96 kbps -- LATENT bundle"
echo "=========================================================="
"$PY" scripts/eval_opus_bundle.py \
    --bundle-dir "$LATENT_DIR" \
    --target-model qwen2_audio \
    --opus-bitrates "$OPUS_NEW"

echo "=========================================================="
echo "[$(date '+%F %T')] Step 4/4: Opus 96 kbps -- WAVEFORM bundle"
echo "=========================================================="
"$PY" scripts/eval_opus_bundle.py \
    --bundle-dir "$WAVEFORM_DIR" \
    --target-model qwen2_audio \
    --opus-bitrates "$OPUS_NEW"

echo "=========================================================="
echo "[$(date '+%F %T')] DONE"
echo "=========================================================="
