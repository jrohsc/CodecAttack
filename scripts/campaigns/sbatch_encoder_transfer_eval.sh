#!/bin/bash
# Cross-model transfer eval for the encoder-transfer arms.
#
# Reuses 0_all_combined/scripts/cross_model_transfer_eval.py unchanged, so the
# criterion (exact substring / WER<=0.5) and the channel grid are byte-identical
# to the published 0/1566 cells. Each evaluator needs its own conda env.
#
# Usage: sbatch --export=ALL,ARM=enc_targetinit sbatch_encoder_transfer_eval.sh
#
#SBATCH --job-name=enc_tx_eval
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -o ${REPO_ROOT}/logs/enc_tx_eval_%A_%a.log
#SBATCH --array=0-3

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ARM="${ARM:?set ARM=<arm dir name>}"
BUNDLE="0_all_combined/results_encoder_transfer/${ARM}"

MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
ENVS=(  codec-attack qwen-omni  qwen-omni      flamingo3)

i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$i]}
ENV=${ENVS[$i]}

echo "=== transfer-eval arm=$ARM evaluator=$MODEL env=$ENV ==="
conda activate "$ENV"

# qwen2_audio is the attacker here: that column is the white-box diagonal, the
# rest are transfer. Same script either way.
python 0_all_combined/scripts/cross_model_transfer_eval.py \
    --bundle-dir "$BUNDLE" \
    --target-model "$MODEL" \
    ${PROMPT_TEXT:+--prompt-text "$PROMPT_TEXT"} \
    --opus-bitrates 32,64 \
    --mp3-bitrates 64 \
    --aac-bitrates 64

echo "=== done $ARM / $MODEL ==="
