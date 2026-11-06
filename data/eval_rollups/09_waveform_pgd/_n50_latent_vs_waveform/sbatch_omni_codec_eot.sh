#!/bin/bash
# ============================================================================
# SLURM job: Qwen2.5-Omni S1 codec-EoT ablation cell — train the EoT proxy
# against a chosen codec (AAC / MP3 / DAC), then MP3/AAC held-out sidecar.
# Single-seed (the ablation is a contrast, not an error-bar claim); the Opus
# headline + seed sweep carry the error bars.
#
#   TRAIN_CODEC  aac | mp3 | dac         (which codec the EoT proxy trains on)
#   SEED         integer                 (default 1)
#
# Examples:
#   sbatch --export=ALL,TRAIN_CODEC=aac,SEED=1 --job-name=omni_ce_aac sbatch_omni_codec_eot.sh
#   sbatch --export=ALL,TRAIN_CODEC=mp3,SEED=1 --job-name=omni_ce_mp3 sbatch_omni_codec_eot.sh
#   sbatch --export=ALL,TRAIN_CODEC=dac,SEED=1 --job-name=omni_ce_dac sbatch_omni_codec_eot.sh
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

TRAIN_CODEC="${TRAIN_CODEC:?set TRAIN_CODEC=aac|mp3|dac|ensemble}"
SEED="${SEED:-1}"
case "$TRAIN_CODEC" in
  aac) RESULTS_DIR=results_codec_robust_aac ;;
  mp3) RESULTS_DIR=results_codec_robust_mp3 ;;
  dac) RESULTS_DIR=results_codec_robust_dac ;;
  ensemble) RESULTS_DIR=results_codec_robust_ensemble ;;
  *) echo "ERROR: TRAIN_CODEC must be aac|mp3|dac|ensemble"; exit 1 ;;
esac
SCEN=01_finance_voice_agent
BUNDLE="0_all_combined/${RESULTS_DIR}/${SCEN}/qwen25_omni/eps_1.0_seed${SEED}"

echo "=========================================="
echo "  OMNI S1 codec-EoT:  TRAIN_CODEC=$TRAIN_CODEC SEED=$SEED"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
echo "  bundle: $BUNDLE"
echo "=========================================="
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

echo "=== STAGE 1: latent attack (eps=1.0, seed=$SEED, rand-init, train-codec=$TRAIN_CODEC) ==="
$OMNI -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen25_omni --eps 1.0 --seed "$SEED" --rand-init \
    --train-codec "$TRAIN_CODEC"

echo "=== STAGE 2: MP3+AAC held-out sidecar eval ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni

echo "=== DONE  TRAIN_CODEC=$TRAIN_CODEC seed=$SEED  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
