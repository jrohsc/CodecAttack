#!/bin/bash
# Full S1 <=4 kHz waveform control: 10 development + 40 held-out examples.
# For each split, run a distortion probe, solve per-example SNR budgets without
# using attack outcomes, rerun at matched budgets, verify, and score the saved
# and reloaded PCM artifacts.
#SBATCH --job-name=s1_4khz_50
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
REF="$ROOT/reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl"
CONTROL_SPLIT="${CONTROL_SPLIT:-all}"
TASK_RUN_ID="${SLURM_JOB_ID:-local_$$}"

cd "$ROOT"
mkdir -p "$OUT/frozen" "$OUT/eval" "$REPO/logs"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_4khz_full50_${TASK_RUN_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"

run_split () {
  local split=$1
  local probe="$OUT/s1/$split/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_probe/manifest.jsonl"
  local matched="$OUT/s1/$split/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snrmatched/manifest.jsonl"
  local epsmap="$OUT/frozen/epsilon_bandlimited_4khz.$split.json"

  echo "=== [$(date -u +%FT%TZ)] $split distortion probe"
  "$CA" -u -m reviewer_controls.run_attack \
    --scenario s1 --split "$split" --method bandlimited_4khz \
    --epsilon 0.166 --alpha-ratio 0.02 --steps 1000 --seed 1 \
    --run-tag probe --resume --output-root "$OUT"

  echo "=== [$(date -u +%FT%TZ)] $split per-example SNR budgets"
  "$CA" -m reviewer_controls.calibrate_epsilon \
    --probe "$probe" --reference "$REF" --mode per_example --output "$epsmap"

  echo "=== [$(date -u +%FT%TZ)] $split calibrated attack"
  "$CA" -u -m reviewer_controls.run_attack \
    --scenario s1 --split "$split" --method bandlimited_4khz \
    --epsilon 0.166 --alpha-ratio 0.02 --steps 1000 --seed 1 \
    --epsilon-map "$epsmap" --run-tag snrmatched \
    --resume --output-root "$OUT"

  echo "=== [$(date -u +%FT%TZ)] $split realized-SNR verification"
  "$CA" -m reviewer_controls.calibrate_epsilon \
    --probe "$probe" --reference "$REF" --verify "$matched" \
    --mode per_example \
    --output "$OUT/frozen/epsilon_bandlimited_4khz.$split.verified.json"
}

case "$CONTROL_SPLIT" in
  development)
    run_split development
    DEV="$OUT/s1/development/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snrmatched/manifest.jsonl"
    "$CA" -u -m reviewer_controls.evaluate \
      --manifest "$DEV" --output "$OUT/eval/development.jsonl" --resume
    ;;
  evaluation)
    run_split evaluation
    EVAL="$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snrmatched/manifest.jsonl"
    "$CA" -u -m reviewer_controls.evaluate \
      --manifest "$EVAL" --output "$OUT/eval/evaluation.jsonl" --resume
    ;;
  all)
    run_split development
    run_split evaluation
    DEV="$OUT/s1/development/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snrmatched/manifest.jsonl"
    EVAL="$OUT/s1/evaluation/bandlimited_4khz/eps0.166_ar0.02_steps1000_seed1_snrmatched/manifest.jsonl"
    "$CA" -u -m reviewer_controls.evaluate \
      --manifest "$DEV" "$EVAL" --output "$OUT/eval/all50.jsonl" --resume
    ;;
  *)
    echo "ERROR: CONTROL_SPLIT must be development, evaluation, or all" >&2
    exit 2
    ;;
esac

echo "=== [$(date -u +%FT%TZ)] complete"
