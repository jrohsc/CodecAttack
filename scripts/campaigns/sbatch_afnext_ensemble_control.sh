#!/bin/bash
#SBATCH --job-name=afnext-ens-control
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=${REPO_ROOT}/results/afnext_control_%j.log

set -euo pipefail
cd ${WORK_ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Qualcomm/model_overlays/gemma4_venv/bin/python \
  Qualcomm/codec_attack/0_all_combined/scripts/eval_afnext_ensemble.py \
  --control-wav ${MODEL_ROOT}/desta/DeSTA2.5-Audio/assets/audios/1003_IWL_ANG_XX.wav
