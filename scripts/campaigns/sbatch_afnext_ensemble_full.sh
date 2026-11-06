#!/bin/bash
#SBATCH --job-name=afnext-ens-full
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=${REPO_ROOT}/results/afnext_full_%j.log

set -euo pipefail
cd ${WORK_ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Qualcomm/model_overlays/gemma4_venv/bin/python \
  Qualcomm/codec_attack/0_all_combined/scripts/eval_afnext_ensemble.py \
  --n-targets 112 --seeds 1,7,42 \
  --channels clean,opus64,opus128,mp3_64,mp3_128,aac_64,aac_128 \
  --delivery-root Qualcomm/codec_attack/results/ensemble_shared112_delivery \
  --output Qualcomm/codec_attack/results/afnext_ensemble_matched_full.json
