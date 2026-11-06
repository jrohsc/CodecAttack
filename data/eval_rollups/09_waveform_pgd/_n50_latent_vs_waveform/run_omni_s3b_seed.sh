#!/usr/bin/env bash
# Run Qwen2.5-Omni S3b (copyright bypass) latent attack for one seed, eps=1.0,
# multi_bitrate EoT, random-init. Then MP3/AAC sidecar eval. Resumable.
#
# Usage: bash run_omni_s3b_seed.sh <SEED>
set -uo pipefail
SEED="${1:?usage: run_omni_s3b_seed.sh <SEED>}"
REPO=${REPO_ROOT}
cd "$REPO"
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BUNDLE=0_all_combined/results_codec_robust/03b_copyright_bypass/qwen25_omni/eps_1.0_multibitrate_seed${SEED}
PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"
ts(){ date "+%F %H:%M:%S"; }

echo "=== [$(ts)] OMNI S3b seed=$SEED  GPU=$CUDA_VISIBLE_DEVICES ==="
echo "=== [$(ts)] ATTACK (eps=1.0, 1000 steps, n=45, rand-init) ==="
$OMNI -u 0_all_combined/scripts/run_s3_attack.py \
    --sub-scenario 03b --target-model qwen25_omni --eps 1.0 \
    --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval

echo "=== [$(ts)] SIDECAR (MP3+AAC) ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni --prompt-text "$PROMPT"

echo "=== [$(ts)] OMNI S3b seed=$SEED DONE ==="
printf "  %s wavs=%s codecs=%s\n" "${BUNDLE##*results_}" \
  "$(ls $BUNDLE/audio/*_attacked.wav 2>/dev/null|wc -l)" \
  "$(ls $BUNDLE/audio/*_codecs.json 2>/dev/null|wc -l)"
