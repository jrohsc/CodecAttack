#!/bin/bash
# Re-run of the encoder-objective arms with the frame mask applied.
#
# The first encoder arms (enc_zeros, enc_targetinit, encce_targetinit) optimized
# a mean per-frame cosine over all 750 tower frames. Only ~16.5% of those frames
# carry target content: a 4.9s spoken target inside a 10s carrier, both padded to
# a 30s window. Measured on those arms (diagnose_encoder_objective.py, n=6):
#
#   content frames  cos 0.4797   <- below the 0.4831 MINIMUM of the cosine
#                                   between two unrelated commands
#   target padding  cos 0.8745   <- paid the optimizer to mute the carrier tail
#   pad vs pad      cos 0.9729   <- free
#   reported mean   cos 0.8697
#
# So those arms never aligned content at all, and their 0/50 says nothing about
# whether encoder-space attacks transfer. These arms fix that:
#
#   --frame-mask target   cosine over the target's own frames only
#   --carrier-mode match  carrier = target + 0.5s, so the pad-vs-speech region
#                         that rewarded muting shrinks to ~0
#
# arm 0: enc    + target-init + mask  <- the clean test of the transfer hypothesis
# arm 1: enc_ce + target-init + mask  <- hybrid, best shot at a working attack
#
# Steps raised 1000 -> 2500: in the 300-step smoke the masked cosine was still
# climbing at the last step (0.24 clean -> 0.77) with target words already
# surfacing in the transcript.
#
# Per-clip results are cached, so a preemption resumes rather than restarting --
# but the bundle is only complete when the wav count equals --n. Check it before
# scoring anything.
#
#SBATCH --job-name=enc_tx_mask
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 20:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH -o ${REPO_ROOT}/logs/enc_tx_mask_%A_%a.log

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOSSES=(enc            enc_ce)
NAMES=( enc_mask_match encce_mask_match)

i=$SLURM_ARRAY_TASK_ID
LOSS=${LOSSES[$i]}
NAME=${NAMES[$i]}

echo "=== masked arm $i: loss=$LOSS name=$NAME ==="
python encoder_transfer_attack.py \
    --loss "$LOSS" \
    --init target \
    --init-scale 0.1 \
    --eps 1.0 \
    --steps "${STEPS:-2500}" \
    --check-every 250 \
    --frame-mask target \
    --carrier-mode match \
    --n "${N_PAIRS:-50}" \
    --source-model qwen2_audio \
    --prompt-mode "${PROMPT_MODE:-transcribe}" \
    --out-dir "0_all_combined/results_encoder_transfer/${NAME}"

echo "=== masked arm $i done; wav count: $(ls 0_all_combined/results_encoder_transfer/${NAME}/audio/*_attacked.wav 2>/dev/null | wc -l) ==="
