#!/bin/bash
# ============================================================================
# SLURM job: ENSEMBLE-VICTIM transfer probe (AC Q1, optional).
#
# Optimizes ONE latent perturbation against {qwen2_audio + qwen25_omni}
# simultaneously (gradient stitching, both in-process in codec-attack env),
# holding out audio_flamingo. Goal: does co-optimizing on 2 models produce ANY
# non-zero transfer to the held-out 3rd, vs the measured 0/1026 single-model
# transfer (tab:cross_model)?
#
# Needs an 80GB A100: 2x7B weights (~30GB) + backward exceeds the 40GB nodes
# (which OOM'd locally). --constraint=a100-80g is REQUIRED. On 80GB no gradient
# checkpointing is needed, so ENSEMBLE_GRAD_CKPT stays unset (fast path).
#
# CAVEATS (this is a DIRECTIONAL PRELIMINARY, not a headline):
#   - native ensemble harness = music carriers + transcribe-mode agent commands,
#     NOT the S1 finance / eps=1.0 / codec-EoT setup of tab:cross_model.
#   - optimizes on clean audio (no multi-bitrate Opus EoT in the loss).
#   If this shows signal, THEN invest in the S1+EoT-comparable version.
#
# Low priority (gpu-preempt) so it does NOT jump the required Q2/Q3/Q5 jobs.
#
#   sbatch --job-name=ens_transfer sbatch_ensemble.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

echo "=========================================="
echo "  ENSEMBLE-VICTIM transfer probe (Q1)"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
python -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0),"| mem",round(torch.cuda.get_device_properties(0).total_memory/1e9,1),"GB")'
echo "  ensemble = qwen2_audio + qwen25_omni   held-out = audio_flamingo"
echo "=========================================="

# Ensemble attack over 1 music carrier x all agent commands (~preliminary n).
# ENSEMBLE_GRAD_CKPT unset -> full-speed path (80GB fits without checkpointing).
python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni \
    --music jazz_1

echo "=== ensemble attack done. adversarial wavs: ==="
find results* -path "*ensemble*" -name "*_attacked.wav" -newermt "-10 hours" 2>/dev/null | head
echo "NEXT (separate step, flamingo3 env): eval these wavs on held-out audio_flamingo"
echo "  + on qwen2_audio/qwen25_omni for white-box-ensemble confirmation."
echo "=== DONE ==="
