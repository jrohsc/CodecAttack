#!/bin/bash
# SNR calibration for the S3a waveform arm -- one job per model.
#
#   sbatch --export=ALL,MODEL=qwen2_audio,REFSNR=7.823 \
#          --job-name=calib_s3a_q2a sbatch_calib_s3a.sh
#
# REFSNR is the mean latent SNR over seeds {1,7,42}, NOT seed 1 alone. Seed 1 is
# systematically the lowest-SNR seed, so matching it biases eps' high and pushes
# the other two seeds toward the 1 dB gate -- exactly how Omni S3b failed.
# Measured 2026-08-09 with measure_bundle_snr.py --loader load_music:
#   qwen2_audio     7.721 / 7.844 / 7.905  -> 7.823
#   qwen25_omni     8.276 / 8.324 / 8.389  -> 8.330
#   audio_flamingo  7.816 / 7.881 / 7.926  -> 7.874
#
# The eps grid must BRACKET the reference on both sides. S3b's grid topped out
# at 0.075 (8.8-10.6 dB, all ABOVE its 7.4-8.8 dB references), so every S3b eps'
# was an extrapolation and Omni's missed by >1 dB. 0.15 here is expected to land
# below 7 dB, making the fit an interpolation.
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

MODEL="${MODEL:?set MODEL=qwen2_audio|qwen25_omni|audio_flamingo}"
REFSNR="${REFSNR:?set REFSNR=<mean latent SNR over seeds 1,7,42>}"

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio)    PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni)    PY="$ENVROOT/qwen-omni/bin/python" ;;
    audio_flamingo) PY="$ENVROOT/flamingo3/bin/python" ;;
    *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Seed 1 only names the bundle the loader/n come from; REFSNR overrides its mean.
REF=0_all_combined/results_codec_robust/03a_ai_detection_bypass/${MODEL}/eps_1.0_multibitrate_seed1

echo "=== calib s3a x $MODEL  ref=${REFSNR} dB ==="
echo "    job $SLURM_JOB_ID on $SLURM_NODELIST GPU=$CUDA_VISIBLE_DEVICES"

$PY -u 0_all_combined/scripts/snr_calibrate_generic.py \
    --scenario s3a --target-model "$MODEL" \
    --ref-bundle "$REF" --ref-snr-override "$REFSNR" \
    --loader load_music \
    --eps 0.03 0.05 0.075 0.10 0.15 --steps 300 --k 3

echo "=== DONE -> 0_all_combined/calibration/snr_${MODEL}_s3a.json ==="
