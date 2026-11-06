#!/bin/bash
# Code-path control for the prompt-sensitivity result.
#
# sbatch_s1_transfer_noprompt.sh found the white-box DIAGONAL collapses under the
# transcribe prompt: Qwen2-Audio 0/50 clean (published bank-prompt value: 88),
# AF3 0/50 (published: 48), Omni 5/50 (published: 84).
#
# But those two numbers come from different code paths -- the published ones from
# the attack driver's own scoring (rescore_uncompressed.py), the new ones from
# cross_model_transfer_eval.py. This job re-measures the diagonal through the SAME
# script at the SAME bank prompt, so the only variable left between the two cells
# is the prompt itself. Without this, "the attack is prompt-bound" is confounded
# with "the two scorers disagree".
#
# 3 cells: each attacker bundle self-evaluated at the bank prompt.
# Writes transfer_eval_<model>__bankprompt.json.
#
# Usage: sbatch sbatch_s1_diagonal_bankprompt.sh
#
#SBATCH --job-name=s1_diag_bank
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH -o ${REPO_ROOT}/logs/s1_diag_bank_%A_%a.log
#SBATCH --array=0-2

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent
MODELS=(qwen2_audio qwen25_omni audio_flamingo)
ENVS=(  codec-attack qwen-omni  flamingo3)

i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$i]}
ENV=${ENVS[$i]}

# No --prompt-text: the script's default IS the bank prompt, matching how these
# wavs were optimized and how the published cells were scored.
echo "=== S1 diagonal @ BANK prompt: $MODEL (env=$ENV) ==="
conda activate "$ENV"

python 0_all_combined/scripts/cross_model_transfer_eval.py \
    --bundle-dir "$ROOT/$MODEL/eps_1.0" \
    --target-model "$MODEL" \
    --out-tag bankprompt \
    --opus-bitrates 32,64

echo "=== done $MODEL ==="
