#!/bin/bash
# ============================================================================
# SLURM job: EnCodec-EoT seed-7 attack, STAGE 1 ONLY, restricted to a disjoint
# carrier shard so several workers can fill one bundle in parallel.
#
#   sbatch --export=ALL,SEED=7,CARRIERS="a.wav b.wav" --job-name=enc_s7_A \
#          sbatch_encodec_eot_shard.sh
#
# Why stage 1 only: stages 2-3 (MP3/AAC sidecar, neural delivery eval) must see
# the COMPLETE 50-clip bundle. They stay with the full-bundle job
# (sbatch_encodec_eot.sh), which is run last as a sweeper -- its stage 1 skips
# every pair already on disk, so it costs nothing for shards already finished.
#
# Shards MUST be disjoint. The driver's skip-if-exists check only sees files
# that are already written, not pairs another worker is mid-flight on.
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-7}"
EPS=1.0
: "${CARRIERS:?set CARRIERS to a space-separated carrier list}"

echo "=========================================="
echo "  EnCodec-EoT shard  SEED=$SEED EPS=$EPS"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
echo "  carriers ($(echo $CARRIERS | wc -w)): $CARRIERS"
echo "=========================================="
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

# Fail fast on the bad-node mode that killed job 62233188 (No CUDA GPUs available).
$OMNI -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "FATAL: no usable CUDA device on $SLURM_NODELIST -- requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

echo "=== STAGE 1 (shard): latent attack, EoT=encodec ==="
$OMNI -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen25_omni --eps "$EPS" --seed "$SEED" --rand-init \
    --train-codec encodec --carriers $CARRIERS

BUNDLE="0_all_combined/results_codec_robust_encodec/01_finance_voice_agent/qwen25_omni/eps_${EPS}_seed${SEED}"
echo "=== SHARD DONE seed=$SEED ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  bundle now has {} / 50 attacked wavs"
