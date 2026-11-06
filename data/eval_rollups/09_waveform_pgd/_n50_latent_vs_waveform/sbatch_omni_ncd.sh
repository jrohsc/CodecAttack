#!/bin/bash
# ============================================================================
# SLURM job: held-out NEURAL-CODEC delivery eval (DAC + EnCodec) on one
# Qwen2.5-Omni S1 seed bundle. Inference-only (no re-optimization): feeds the
# already-attacked wavs through the neural codec round-trip and re-queries Omni.
# Gives DAC-delivery ASR per seed -> 3-seed error bars for the held-out DAC col.
#
# Meant to run AFTER the matching omni_s1_s<SEED> attack finishes:
#   sbatch --dependency=afterok:<attack_jobid> --export=ALL,SEED=1 \
#          --job-name=omni_ncd_s1 sbatch_omni_ncd.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:?set SEED}"
BUNDLE="0_all_combined/results_codec_robust/01_finance_voice_agent/qwen25_omni/eps_1.0_seed${SEED}"
echo "=== OMNI held-out DAC/EnCodec delivery eval  seed=$SEED  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "  bundle: $BUNDLE"
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs present: {}"

$OMNI -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni --codecs dac,encodec

echo "=== DONE  wrote neural_delivery_eval_qwen25_omni.json in $BUNDLE ==="
