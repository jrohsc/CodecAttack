#!/bin/bash
# ============================================================================
# SLURM job: Qwen2.5-Omni S3b (copyright bypass) WAVEFORM baseline, one seed.
#
# Companion to sbatch_omni_s3b.sh (the latent arm). Same carriers, same 1000
# steps, same multi-bitrate Opus EoT, same rand-init; only the perturbation
# domain differs. EPS must be the SNR-calibrated value matching the S3b latent
# arm's clean-channel SNR (8.82 dB measured on eps_1.0_multibitrate_seed1).
#
#   sbatch --export=ALL,SEED=1,EPS=0.15  --job-name=omni_s3bw_s1  sbatch_omni_s3b_waveform.sh
#   sbatch --export=ALL,SEED=7,EPS=0.15  --job-name=omni_s3bw_s7  sbatch_omni_s3b_waveform.sh
#   sbatch --export=ALL,SEED=42,EPS=0.15 --job-name=omni_s3bw_s42 sbatch_omni_s3b_waveform.sh
#
# Resumable: the driver skips pairs whose *_attacked.wav already exists and the
# sidecar skips clips that already have *_codecs.json. Safe to requeue.
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

SEED="${SEED:?set SEED}"
EPS="${EPS:?set EPS (SNR-calibrated waveform budget)}"
BUNDLE=0_all_combined/results_codec_robust_waveform/03b_copyright_bypass/qwen25_omni/eps_${EPS}_seed${SEED}
PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

echo "=== omni S3b WAVEFORM seed=$SEED eps=$EPS  job $SLURM_JOB_ID on $SLURM_NODELIST GPU=$CUDA_VISIBLE_DEVICES ==="
$OMNI -c 'import torch;print("torch",torch.__version__,"gpu",torch.cuda.get_device_name(0))'

echo "=== ATTACK (waveform, 1000 steps, n=45, multi_bitrate EoT, rand-init) ==="
$OMNI -u 0_all_combined/scripts/run_waveform_bundleA_multibitrate.py \
    --scenario s3b --target-model qwen25_omni \
    --eps "$EPS" --seed "$SEED" --rand-init

echo "=== SIDECAR (MP3+AAC) ==="
$OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen25_omni --prompt-text "$PROMPT"

echo "=== DONE seed=$SEED ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
