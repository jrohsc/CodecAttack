#!/bin/bash
# ============================================================================
# SLURM job: Qwen2.5-Omni S1 WAVEFORM-space baseline (for the Q2 latent-vs-
# waveform contrast on the MAIN model). Uses the ported waveform driver
# (--target-model qwen25_omni, add-only). Then MP3/AAC held-out sidecar.
#
#   LP    0 | 400        (0 = unconstrained waveform; 400 = band-limited <=400 Hz)
#   SEED  integer        (default 1)
#
# Examples:
#   sbatch --export=ALL,LP=400,SEED=1 --job-name=omni_wav_lp400 sbatch_omni_waveform.sh
#   sbatch --export=ALL,LP=0,SEED=1   --job-name=omni_wav       sbatch_omni_waveform.sh
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
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LP="${LP:-0}"
SEED="${SEED:-1}"
EPS=0.075
LP_FLAG=""; LP_TAG=""
if [ "$LP" != "0" ]; then LP_FLAG="--lowpass-hz $LP"; LP_TAG="_lp${LP}"; fi
BUNDLE="0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni/eps_${EPS}${LP_TAG}_seed${SEED}"

echo "=== OMNI S1 waveform  LP=$LP SEED=$SEED  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "  bundle: $BUNDLE"
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

echo "=== STAGE 1: waveform attack (eps=$EPS lp=$LP seed=$SEED rand-init, qwen25_omni) ==="
$OMNI -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --target-model qwen25_omni --eps "$EPS" --seed "$SEED" --rand-init $LP_FLAG

echo "=== STAGE 2: MP3+AAC held-out sidecar ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni

echo "=== DONE  LP=$LP seed=$SEED  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
