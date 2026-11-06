#!/bin/bash
# Extra crafting workers for the e3phi seed 7/42 shards (see steal_e3phi.py).
# Each array task claims clips from the tail of the slowest shards, one per
# run_benchmark call, and publishes them into the main bundles.
#SBATCH --job-name=e3phi_steal
#SBATCH --partition=gpu-preempt,superpod-a100,gpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e3phi_steal/slurm_%x_%A_%a.out
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
exec ${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python -u \
  0_all_combined/n50_latent_vs_waveform/steal_e3phi.py worker --gpu inherit
