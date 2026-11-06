#!/bin/bash
# 2x2 ablation isolating the two levers the audio-transfer literature uses and
# our published S1 runs did not:
#
#   loss = ce   -> CE on LM output tokens      (what we published; 0/1566 transfer)
#   loss = enc  -> per-frame cosine on Whisper encoder latents  (U-TLSA, 2512.23881)
#   init = zeros  -> published init
#   init = target -> scaled spoken-target init (ZQ-Attack, 2406.19311)
#
# arm 0: ce  + zeros   <- baseline control, must reproduce the published white-box rate
# arm 1: ce  + target  <- ZQ init alone
# arm 2: enc + zeros   <- encoder objective alone
# arm 3: enc + target  <- both (the predicted-transferable configuration)
# arm 4: enc_ce + target <- hybrid, keeps the white-box rate while adding encoder pull
#
#SBATCH --job-name=enc_tx
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 20:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH -o ${REPO_ROOT}/logs/enc_tx_%A_%a.log

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

LOSSES=(ce  ce     enc   enc    enc_ce)
INITS=( zeros target zeros target target)
NAMES=(ce_zeros ce_targetinit enc_zeros enc_targetinit encce_targetinit)

i=$SLURM_ARRAY_TASK_ID
LOSS=${LOSSES[$i]}
INIT=${INITS[$i]}
NAME=${NAMES[$i]}

echo "=== arm $i: loss=$LOSS init=$INIT name=$NAME ==="
python encoder_transfer_attack.py \
    --loss "$LOSS" \
    --init "$INIT" \
    --init-scale 0.1 \
    --eps 1.0 \
    --steps 1000 \
    --check-every 100 \
    --n "${N_PAIRS:-10}" \
    --source-model qwen2_audio \
    --prompt-mode "${PROMPT_MODE:-transcribe}" \
    --out-dir "0_all_combined/results_encoder_transfer/${NAME}"

echo "=== arm $i done; wav count: $(ls 0_all_combined/results_encoder_transfer/${NAME}/audio/*_attacked.wav 2>/dev/null | wc -l) ==="
