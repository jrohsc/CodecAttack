#!/bin/bash
# Missing arm C only: tuned unrestricted waveform, frozen S1 held-out split.
# Required exports: PHASE={probe,matched}, SHARD_ID={0,1,2,3}.
# The learning rate (alpha/epsilon=0.02) was frozen before held-out evaluation.
# Epsilon calibration is distortion-only and is produced by
# sbatch_s1_tuned_full40_calibrate.sh between the two phases.
#SBATCH --job-name=s1_tuned40
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PHASE="${PHASE:?set PHASE to probe or matched}"
SHARD_ID="${SHARD_ID:?set SHARD_ID to 0, 1, 2, or 3}"
case "$PHASE" in probe|matched) ;; *) echo "invalid PHASE=$PHASE" >&2; exit 2 ;; esac
case "$SHARD_ID" in 0|1|2|3) ;; *) echo "invalid SHARD_ID=$SHARD_ID" >&2; exit 2 ;; esac

REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_tuned_full40_seed1"
EPSMAP="$OUT/frozen/epsilon_tuned_waveform.evaluation.json"
TASK_RUN_ID="${SLURM_JOB_ID:-local_$$}"

cd "$ROOT"
mkdir -p "$OUT/frozen" "$REPO/logs"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_tuned40_${PHASE}_sh${SHARD_ID}_${TASK_RUN_ID}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"

extra=(--run-tag "$PHASE")
if [[ "$PHASE" == matched ]]; then
  [[ -f "$EPSMAP" ]] || { echo "missing frozen epsilon map: $EPSMAP" >&2; exit 1; }
  extra+=(--epsilon-map "$EPSMAP")
fi

echo "=== [$(date -u +%FT%TZ)] arm=C split=evaluation phase=$PHASE shard=$SHARD_ID/4 alpha/epsilon=0.02 seed=1"
"$CA" -u -m reviewer_controls.run_attack \
  --scenario s1 --split evaluation --method tuned_waveform \
  --epsilon 0.06082 --alpha-ratio 0.02 --steps 1000 --seed 1 \
  --shard-id "$SHARD_ID" --num-shards 4 --resume \
  --output-root "$OUT" "${extra[@]}"

