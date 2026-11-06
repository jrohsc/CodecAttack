#!/bin/bash
# ============================================================================
# SLURM job: one Qwen2.5-Omni S1 (finance voice agent) latent multi-seed
# bundle, eps=1.0, multi_bitrate Opus EoT, random-init + MP3/AAC held-out
# sidecar. Resumable (both stages skip pairs already on disk). Requeue-safe.
#
# This is the MAIN-MODEL version of the S1 latent seed sweep (Qwen2-Audio is
# in the appendix). Produces the Opus + MP3(held-out) + AAC(held-out) table
# with 3-seed error bars on the flagship model.
#
# Examples:
#   sbatch --export=ALL,SEED=1  --job-name=omni_s1_s1  sbatch_omni_s1.sh
#   sbatch --export=ALL,SEED=7  --job-name=omni_s1_s7  sbatch_omni_s1.sh
#   sbatch --export=ALL,SEED=42 --job-name=omni_s1_s42 sbatch_omni_s1.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:?set SEED=<int>}"
SCEN=01_finance_voice_agent
BUNDLE="0_all_combined/results_codec_robust/${SCEN}/qwen25_omni/eps_1.0_seed${SEED}"

echo "=========================================="
echo "  OMNI S1 latent seed sweep:  SEED=$SEED"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
echo "  bundle: $BUNDLE"
echo "=========================================="
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

echo "=== STAGE 1: latent attack (eps=1.0, seed=$SEED, rand-init, multi_bitrate EoT) ==="
$OMNI -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen25_omni --eps 1.0 --seed "$SEED" --rand-init \
    --channel-mode multi_bitrate

echo "=== STAGE 2: MP3+AAC held-out sidecar eval ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni

echo "=== DONE  seed=$SEED  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
