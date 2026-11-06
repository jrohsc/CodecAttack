#!/bin/bash
# Local (non-SLURM) driver for the seed-7 S1 multiseed bundles.
# Runs on whatever GPU is visible in this session (interactive A100).
# Mirrors sbatch_multiseed.sh for SEED=7, both SPACEs. Fully resumable:
# both the attack and the sidecar skip pairs already on disk.
set -uo pipefail

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

REPO=${REPO_ROOT}
cd "$REPO"
SEED=7
SCEN=01_finance_voice_agent

echo "============================================================"
echo "  LOCAL seed-7 run  host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
python -c 'import torch;print("  gpu:",torch.cuda.get_device_name(0))'
echo "============================================================"

LAT_BUNDLE="0_all_combined/results_codec_robust/${SCEN}/qwen2_audio/eps_1.0_seed${SEED}"
WAV_BUNDLE="0_all_combined/results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_seed${SEED}"

echo "=== STAGE 1a: LATENT attack (eps=1.0 seed=$SEED) ==="
python -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
    --target-model qwen2_audio --eps 1.0 --seed "$SEED" --rand-init

echo "=== STAGE 1b: LATENT codec sidecar eval ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$LAT_BUNDLE" --target-model qwen2_audio

echo "=== STAGE 2a: WAVEFORM attack (eps=0.075 seed=$SEED) ==="
python -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --eps 0.075 --seed "$SEED" --rand-init

echo "=== STAGE 2b: WAVEFORM codec sidecar eval ==="
python -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$WAV_BUNDLE" --target-model qwen2_audio

echo "=== SEED 7 LOCAL RUN DONE ==="
for b in "$LAT_BUNDLE" "$WAV_BUNDLE"; do
  echo "  $b : attacked=$(ls "$b/audio/"*_attacked.wav 2>/dev/null | wc -l | tr -d ' ')/50  codec=$(ls "$b/audio/"*_codecs.json 2>/dev/null | wc -l | tr -d ' ')/50"
done
