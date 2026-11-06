#!/bin/bash
# PHASE 0 of the eps-ladder transfer probe -- the free rung.
#
# Why this exists
# ---------------
# Every cross-model transfer cell we have ever run reads 0. The working
# diagnosis is CAPACITY, not optimizer: at eps=1.0 the perturbation has just
# enough budget to control ONE model under ONE prompt. Two independent facts
# support that:
#
#   1. The 2-model ensemble (qwen2_audio + qwen25_omni, eps=1.0) drops the
#      IN-ENSEMBLE Omni from ~100% single-model white-box to 11/50 clips (22%).
#      Held-out AF3 and Omni-3B are 0/650 cells.
#   2. The attack is prompt-bound: the white-box diagonal collapses 88 -> 0
#      when the evaluator prompt is swapped (sbatch_s1_transfer_noprompt.sh),
#      and the code-path control (sbatch_s1_diagonal_bankprompt.sh) confirmed
#      that collapse is the prompt, not a scorer disagreement.
#
# If capacity is the binding constraint, transfer should be monotone in eps.
# The eps=1.5 S1 bundles ALREADY EXIST (50/50 ok wavs each, all three models,
# built by scripts/chain_s1_eps1.5.sh) and have never been transfer-evaluated
# -- only neural_delivery sidecars were run on them. So the 1.0 -> 1.5 rung
# costs inference only, zero attack GPU. Run it before spending anything on
# the eps in {2,4,8} attack sweep (Phase 1).
#
# Design notes
# ------------
# * Bank prompt (the script default) -- NOT the transcribe prompt. These wavs
#   were optimized under the bank prompt, and the published eps=1.0 cells were
#   scored under it. Changing eps and prompt at once would confound the rung.
# * --opus-bitrates 32,64 matches the published eps=1.0 transfer cells exactly,
#   so the two rungs are directly comparable cell-for-cell.
# * The attacker's own column is the WHITE-BOX DIAGONAL and is a required
#   control: if the diagonal degrades at eps=1.5 (over-optimization), then a
#   flat 0 in the transfer columns is uninterpretable.
# * Untagged output (transfer_eval_<model>.json) inside the eps_1.5 bundle,
#   matching the convention of the published eps_1.0 cells. Verified 2026-08-14
#   that no transfer_eval_* file exists in any eps_1.5 bundle, so nothing is
#   clobbered. The published eps_0.5 / eps_1.0 trees are NOT touched.
#
# 12 cells = 3 attacker bundles x 4 evaluators.
#
# Usage: sbatch sbatch_s1_eps15_transfer.sh
#
#SBATCH --job-name=s1_eps15_tx
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40&bf16
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -o ${REPO_ROOT}/logs/s1_eps15_tx_%A_%a.log
#SBATCH --array=0-11

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent

ATTACKERS=(qwen2_audio qwen25_omni audio_flamingo)
MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
ENVS=(  codec-attack qwen-omni  qwen-omni      flamingo3)

i=$SLURM_ARRAY_TASK_ID
ATK=${ATTACKERS[$((i / 4))]}
j=$((i % 4))
MODEL=${MODELS[$j]}
ENV=${ENVS[$j]}
BUNDLE="$ROOT/$ATK/eps_1.5"

ROLE="transfer"
[[ "$ATK" == "$MODEL" ]] && ROLE="WHITE-BOX DIAGONAL (control)"

# Fail loudly rather than silently evaluating a truncated bundle.
NWAV=$(ls "$BUNDLE"/audio/*.wav 2>/dev/null | wc -l)
if [[ "$NWAV" -ne 50 ]]; then
    echo "FATAL: $BUNDLE has $NWAV wavs, expected 50 -- refusing to score a partial bundle."
    exit 2
fi

echo "=== S1 eps=1.5 transfer: attacker=$ATK evaluator=$MODEL [$ROLE] env=$ENV ==="
echo "=== bundle=$BUNDLE (n=$NWAV wavs) prompt=BANK (script default)"
conda activate "$ENV"

python 0_all_combined/scripts/cross_model_transfer_eval.py \
    --bundle-dir "$BUNDLE" \
    --target-model "$MODEL" \
    --opus-bitrates 32,64

echo "=== done $ATK -> $MODEL ==="
