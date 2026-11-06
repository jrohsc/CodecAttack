#!/bin/bash
#SBATCH --job-name=grm-guard3
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h100|a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm_guard3_%j.out
#SBATCH --error=logs/slurm_guard3_%j.out
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python -u score_guard3.py results
