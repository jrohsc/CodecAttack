#!/bin/bash
# Evaluate two distortion-gated repairs after the 35 fixed cases.
#SBATCH --job-name=s1_d_early2
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_baseline_full40_seed1"
cd "$ROOT"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_d_early2_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
"$CA" -u -m reviewer_controls.evaluate \
  --manifest "$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1/frozen/bandlimited_4khz.evaluation.early2.jsonl" \
  --output "$OUT/eval/armD.jsonl" --resume
