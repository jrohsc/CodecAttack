#!/bin/bash
# ============================================================================
# SLURM batch script: Fill in the missing watermark table experiments.
#
# Missing experiments (all on saycan dataset, christmas_jazz_1):
#   1. Latent attack eps=1.0  → watermark detection
#   2. Waveform attack eps=0.1 → watermark detection
#   3. Waveform attack eps=0.5 → watermark detection
#
# Existing results:
#   - Latent eps=0.3 (1/81 destroyed), eps=0.5 (38/81 destroyed)
#   - Waveform eps=0.01 (0/81 destroyed)
#   - Codec roundtrip eps=0 (0/16 destroyed) — clean baseline done
#
# Usage:
#   sbatch --export=EXP=latent_1.0 sbatch_watermark_table.sh
#   sbatch --export=EXP=waveform_0.1 sbatch_watermark_table.sh
#   sbatch --export=EXP=waveform_0.5 sbatch_watermark_table.sh
#
# Or submit all 3:
#   for EXP in latent_1.0 waveform_0.1 waveform_0.5; do
#     sbatch --export=EXP=$EXP sbatch_watermark_table.sh
#   done
# ============================================================================

#SBATCH --job-name=wm-table
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/wm_table_%x_%j.out
#SBATCH --error=logs/wm_table_%x_%j.err

EXP="${EXP:-latent_1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# ---- Activate conda ----
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Experiment: $EXP"
echo "  Python:    $(which python)"
echo "============================================"

case "$EXP" in
    latent_1.0)
        echo "Running: Latent attack eps=1.0 watermark detection (saycan, christmas_jazz_1)"
        python run_watermark_experiments.py \
            --experiment attack_detection \
            --dataset saycan \
            --music christmas_jazz_1 \
            --eps 1.0 \
            --steps 300
        ;;
    waveform_0.1)
        echo "Running: Waveform attack eps=0.1 watermark detection (saycan, christmas_jazz_1)"
        python run_watermark_experiments.py \
            --experiment waveform_attack_detection \
            --dataset saycan \
            --music christmas_jazz_1 \
            --eps 0.1 \
            --steps 300
        ;;
    waveform_0.5)
        echo "Running: Waveform attack eps=0.5 watermark detection (saycan, christmas_jazz_1)"
        python run_watermark_experiments.py \
            --experiment waveform_attack_detection \
            --dataset saycan \
            --music christmas_jazz_1 \
            --eps 0.5 \
            --steps 300
        ;;
    *)
        echo "ERROR: Unknown experiment '$EXP'. Choose: latent_1.0 | waveform_0.1 | waveform_0.5"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  FINISHED: $EXP"
echo "============================================"
