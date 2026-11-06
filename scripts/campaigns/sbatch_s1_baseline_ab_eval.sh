#!/bin/bash
# Score reusable arms A/B now; the final A/B/C/D evaluator resumes this file.
#SBATCH --job-name=s1_base_ab
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
FINAL="$ROOT/reviewer_control_outputs/s1_baseline_full40_seed1"
cd "$ROOT"
mkdir -p "$FINAL/manifests" "$FINAL/eval" "$REPO/logs"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_base_ab_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"

"$CA" -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation \
  --method latent_codecattack \
  --bundle results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0 \
  --output "$FINAL/manifests/armA_latent.jsonl"
"$CA" -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation \
  --method original_waveform_published \
  --bundle results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.06082_norm_seed1 \
  --output "$FINAL/manifests/armB_original_waveform.jsonl"
"$CA" -u -m reviewer_controls.evaluate \
  --manifest "$FINAL/manifests/armA_latent.jsonl" "$FINAL/manifests/armB_original_waveform.jsonl" \
  --output "$FINAL/eval/arms_ABCD.jsonl" --resume

