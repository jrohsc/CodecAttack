#!/bin/bash
# ============================================================================
# SLURM job: encoder-residual R analysis WITH MP3 added (AC Q5 "why MP3 >> AAC").
# Inference-only on already-trained headline perturbations. Computes, per carrier,
#   R_codec(br) = ||h(codec(x_atk)) - h(x_atk)|| / ||h(x_atk) - h(x_clean)||
# for codec in {AAC, MP3}. Predicted: R_MP3 low everywhere; R_AAC high on S1 speech.
#
# Runs qwen2_audio on S1 (speech; AAC collapses) and S3a (music; AAC survives)
# — the contrast that explains the asymmetry. Separate node so it does NOT
# contend with the codec-EoT attack on the local GPU.
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=../n50_latent_vs_waveform/logs/%x_%j.out
#SBATCH --error=../n50_latent_vs_waveform/logs/%x_%j.err

set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
PY=$(which python)
echo "=== R-analysis (AAC+MP3) qwen2_audio | job $SLURM_JOB_ID on $SLURM_NODELIST ==="
for SCEN in S1 S3a; do
    echo "--- $SCEN ---"
    $PY -u 0_all_combined/scripts/encoder_residual_aac.py --model qwen2_audio --scenario "$SCEN"
done
echo "=== DONE ==="
