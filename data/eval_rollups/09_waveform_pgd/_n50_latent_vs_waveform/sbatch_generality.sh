#!/bin/bash
# Generality sweep: re-optimize against several deployment prompts/tool schemas.
#   sbatch --export=ALL,FAMILY=tools --job-name=gen_tools sbatch_generality.sh
#   sbatch --export=ALL,FAMILY=ats   --job-name=gen_ats   sbatch_generality.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd ${REPO_ROOT}
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
$PY -u 0_all_combined/scripts/generality_sweep.py --family "${FAMILY:?set FAMILY=tools|ats}"
