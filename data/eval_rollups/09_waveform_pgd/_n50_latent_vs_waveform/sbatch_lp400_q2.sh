#!/bin/bash
# Q2 adaptive baseline: finish the zeros-init lp400 waveform bundle (resume
# 34/50 -> 50/50) + MP3/AAC sidecar. Matches the headline latent/waveform
# (both zeros-init) for a clean 3-way decomposition. No seed, no rand-init.
#SBATCH --job-name=lp400_q2
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
BUNDLE=0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075_lp400
echo "=== lp400 Q2 baseline  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "=== ATTACK (resume; lowpass=400Hz, zeros-init) ==="
$PY -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --eps 0.075 --lowpass-hz 400
echo "=== SIDECAR (MP3+AAC) ==="
$PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio
echo "=== DONE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
