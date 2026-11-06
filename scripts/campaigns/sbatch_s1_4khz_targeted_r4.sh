#!/bin/bash
# Distortion-only correction of the five remaining held-out SNR failures.
#SBATCH --job-name=s1_4k_r4
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
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
OUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
cd "$ROOT"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_4k_r4_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
"$CA" -u -m reviewer_controls.run_attack \
  --scenario s1 --split evaluation --method bandlimited_4khz \
  --epsilon 0.166 --alpha-ratio 0.02 --steps 1000 --seed 1 \
  --epsilon-map "$OUT/frozen/epsilon_bandlimited_4khz.evaluation.r4.json" \
  --example-ids-file "$OUT/frozen/bandlimited_4khz.evaluation.bad_r4.txt" \
  --run-tag snr_correction_r4 --resume --output-root "$OUT"
