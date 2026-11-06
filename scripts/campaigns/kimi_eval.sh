#!/bin/bash
# Wrapper for running cross_model_transfer_eval.py with --target-model kimi_audio.
# Kimi is the fourth ARCHITECTURE held-out victim (inference only, no gradients),
# and its env needs three fixes that no other evaluator does:
#
#  1. PYTHONNOUSERSITE=1 -- ~/.local/lib/python3.12 holds a stray torch 2.11+cu130
#     that shadows the env's torch 2.8+cu126, breaking flash_attn's C ABI
#     (undefined symbol c10::cuda::c10_cuda_check_implementation).
#  2. LD_LIBRARY_PATH over the env's nvidia/*/lib -- flash_attn needs libcudart.so.12.
#  3. A PATH shim for ffmpeg -- the kimi-audio env's own ffmpeg is broken
#     (libx264.so.138 missing), so every opus channel died with exit 127.
#     demo_ota.apply_opus_compression resolves ffmpeg via shutil.which.
#
# Usage: bash kimi_eval.sh --bundle-dir <dir> [--prompt-text ...] [--opus-bitrates ...]
set -euo pipefail
cd ${REPO_ROOT}

E=${WORK_ROOT}/miniconda3/envs/kimi-audio
CA=${WORK_ROOT}/miniconda3/envs/codec-attack
SHIM=${REPO_ROOT}/.kimi_ffmpeg_shim
mkdir -p "$SHIM"
[[ -e "$SHIM/ffmpeg" ]] || ln -s "$CA/bin/ffmpeg" "$SHIM/ffmpeg"
[[ -e "$SHIM/ffprobe" ]] || ln -s "$CA/bin/ffprobe" "$SHIM/ffprobe"

exec env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
    PATH="$SHIM:$PATH" \
    LD_LIBRARY_PATH="$(ls -d $E/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}" \
    "$E/bin/python" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
    --target-model kimi_audio "$@"
