#!/bin/bash
#SBATCH --job-name=s1e2_q2fill
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=2:00:00
#SBATCH --output=${REPO_ROOT}/logs/s1e2_q2fill_%j.out
#SBATCH --error=${REPO_ROOT}/logs/s1e2_q2fill_%j.err
#
# The published S1 ensemble-2 (qwen2+omni7b) bundles have evals for omni7b, AF3 and
# omni3b but NOT for qwen2_audio -- the PRIMARY member, i.e. the one white-box number
# the cell exists to report. Backfills it for all three seeds under the bank prompt
# (cross_model_transfer_eval.py's default, matching the sibling transfer_eval files).
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
B=0_all_combined/results_codec_robust_ensemble_victim/01_finance_voice_agent/ensemble
for S in 1 7 42; do
  echo "=== [$(date -Is)] seed=$S  wavs=$(ls $B/eps_1_seed$S/audio/*.wav 2>/dev/null | wc -l)"
  $PY 0_all_combined/scripts/cross_model_transfer_eval.py \
      --bundle-dir "$B/eps_1_seed$S" --target-model qwen2_audio
done
echo "=== [$(date -Is)] ens2 qwen2 backfill DONE"
