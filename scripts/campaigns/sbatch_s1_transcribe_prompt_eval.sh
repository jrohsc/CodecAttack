#!/bin/bash
# EXPERIMENT #1, phase 2: evaluate the transcribe-prompt-optimized bundles
# (sbatch_s1_attack_transcribe_prompt.sh) against all four evaluators under the
# SAME transcribe prompt they were optimized with.
#
# Read the output as a matrix:
#   diagonal (attacker == evaluator) = white-box. If this is NOT high, the
#     experiment failed at the optimization step and the transfer cells are
#     uninterpretable again -- check before reading anything else.
#   off-diagonal = transfer, now measured against a 98.5% ceiling instead of 7%.
#
# 12 tasks = 3 attacker bundles x 4 evaluators.
# Usage: sbatch --dependency=afterok:<attack_jobid> sbatch_s1_transcribe_prompt_eval.sh
#
#SBATCH --job-name=s1_txprompt_ev
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -o ${REPO_ROOT}/logs/s1_txprompt_ev_%A_%a.log
#SBATCH --array=0-11

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

PROMPT="What does the person say in this audio?"
ROOT=0_all_combined/results_transcribe_prompt/01_finance_voice_agent

ATTACKERS=(qwen2_audio qwen25_omni audio_flamingo)
MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
ENVS=(  codec-attack qwen-omni  qwen-omni      flamingo3)

i=$SLURM_ARRAY_TASK_ID
ATK=${ATTACKERS[$((i / 4))]}
j=$((i % 4))
MODEL=${MODELS[$j]}
ENV=${ENVS[$j]}
BUNDLE="$ROOT/$ATK/eps_1.0"

ROLE="transfer"
[[ "$ATK" == "$MODEL" ]] && ROLE="WHITE-BOX DIAGONAL (control)"

NWAV=$(ls "$BUNDLE"/audio/*attacked*.wav 2>/dev/null | wc -l)
echo "=== txprompt eval: attacker=$ATK evaluator=$MODEL [$ROLE] wavs=$NWAV ==="
if [[ "$NWAV" -lt 50 ]]; then
    echo "WARNING: bundle has $NWAV/50 wavs -- partial bundle (SLURM preemption?)." >&2
fi

conda activate "$ENV"
python 0_all_combined/scripts/cross_model_transfer_eval.py \
    --bundle-dir "$BUNDLE" \
    --target-model "$MODEL" \
    --prompt-text "$PROMPT" \
    --out-tag transcribe \
    --opus-bitrates 32,64

echo "=== done $ATK -> $MODEL ==="
