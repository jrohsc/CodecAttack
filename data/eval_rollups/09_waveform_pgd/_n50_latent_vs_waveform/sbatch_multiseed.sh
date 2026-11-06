#!/bin/bash
# ============================================================================
# SLURM job: one multi-seed S1 bundle (attack + MP3/AAC sidecar eval).
#
# Parametrized via --export:
#   SPACE   latent | waveform            (which attack)
#   SEED    integer                      (RNG seed; routes to eps_..._seed<S>)
#   LP      0 | 400                       (waveform lowpass-Hz; 0 = none)
#
# Each job is INDEPENDENT (own per-seed bundle dir) and RESUMABLE (both the
# attack and the sidecar skip pairs already on disk). Safe to requeue.
#
# Examples:
#   sbatch --export=SPACE=latent,SEED=7      --job-name=ms_lat_s7      sbatch_multiseed.sh
#   sbatch --export=SPACE=waveform,SEED=42   --job-name=ms_wav_s42     sbatch_multiseed.sh
#   sbatch --export=SPACE=waveform,SEED=42,LP=400 --job-name=ms_wavlp400_s42 sbatch_multiseed.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=60G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

PY=${WORK_ROOT}/miniconda3/bin/python
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
PY=$(which python)

SPACE="${SPACE:?set SPACE=latent|waveform}"
SEED="${SEED:?set SEED=<int>}"
LP="${LP:-0}"
SCEN=01_finance_voice_agent

echo "=========================================="
echo "  multiseed bundle:  SPACE=$SPACE SEED=$SEED LP=$LP"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  GPU=$CUDA_VISIBLE_DEVICES"
echo "  python: $PY"
python -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'
echo "=========================================="

if [ "$SPACE" = "latent" ]; then
    EPS=1.0
    BUNDLE="0_all_combined/results_codec_robust/${SCEN}/qwen2_audio/eps_${EPS}_seed${SEED}"
    echo "=== STAGE 1: latent attack (eps=$EPS seed=$SEED) ==="
    python -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model qwen2_audio --eps "$EPS" --seed "$SEED" --rand-init
elif [ "$SPACE" = "waveform" ]; then
    EPS=0.075
    LP_FLAG=""
    LP_TAG=""
    if [ "$LP" != "0" ]; then LP_FLAG="--lowpass-hz $LP"; LP_TAG="_lp${LP}"; fi
    BUNDLE="0_all_combined/results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_${EPS}${LP_TAG}_seed${SEED}"
    echo "=== STAGE 1: waveform attack (eps=$EPS lp=$LP seed=$SEED) ==="
    python -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
        --eps "$EPS" --seed "$SEED" --rand-init $LP_FLAG
else
    echo "ERROR: SPACE must be latent|waveform"; exit 1
fi

echo "=== STAGE 2: MP3+AAC codec sidecar eval on $BUNDLE ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio

echo "=== DONE  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
