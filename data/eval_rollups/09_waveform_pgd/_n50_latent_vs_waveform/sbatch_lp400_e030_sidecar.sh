#!/bin/bash
# ============================================================================
# MP3/AAC sidecar for the SNR-matched lp400 arm (eps'=0.30, Qwen2.5-Omni, S1).
#
# WHY: all three seed bundles have 50/50 attacked wavs and Opus+white-box
# scores, but ZERO *_codecs.json -- so the MP3/AAC columns of the lp400 row in
# Table 2 are currently unbacked. This is inference-only; no attack re-runs.
#
# Prompt: eval_mp3_aac_bundle.py's BANK_PROMPT default is byte-identical to the
# one run_s1_waveform_bundleA_multibitrate.py attacked under (verified), and
# these are 01_finance bundles, so NO --prompt-text passthrough is needed.
# Passing the wrong prompt here is the known silent-failure mode (reads 0/N).
#
# Env: qwen-omni (target model is qwen25_omni). Script is idempotent -- it
# skips clips whose sidecar exists, so --requeue resumes for free.
#
#   sbatch --job-name=lp400_e030_sidecar sbatch_lp400_e030_sidecar.sh
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

BASE=0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni

echo "=== lp400 eps'=0.30 MP3/AAC sidecar  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "=== mp3 grid 32,64,96,128,192 | aac grid 32,64,96,128,192 (defaults) ==="

for SEED in 1 7 42; do
  BUNDLE="$BASE/eps_0.3_lp400_seed${SEED}"
  echo
  echo "=== seed $SEED : $BUNDLE ==="
  if [ ! -d "$BUNDLE/audio" ]; then
    echo "  MISSING BUNDLE -- skipping"; continue
  fi
  $OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
      --bundle-dir "$BUNDLE" --target-model qwen25_omni
  echo "  rc=$?"
done

echo
echo "=== FINAL COUNTS (wavs must equal sidecars) ==="
for SEED in 1 7 42; do
  BUNDLE="$BASE/eps_0.3_lp400_seed${SEED}"
  W=$(ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l)
  C=$(ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l)
  echo "  seed${SEED}: attacked_wavs=$W  codec_sidecars=$C"
done
echo "=== DONE ==="
