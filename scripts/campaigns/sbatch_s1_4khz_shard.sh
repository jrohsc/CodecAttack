#!/bin/bash
# Resume arm D with disjoint ID files. Required: PHASE, SHARD_ID, IDS_FILE.
#SBATCH --job-name=s1_4k_shard
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
PHASE="${PHASE:?probe or matched}"
SHARD_ID="${SHARD_ID:?0..3}"
IDS_FILE="${IDS_FILE:?newline-delimited stable IDs}"
case "$PHASE" in probe|matched) ;; *) exit 2 ;; esac
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_4khz_full50_seed1"
EPSMAP="$OUT/frozen/epsilon_bandlimited_4khz.evaluation.sharded.json"
cd "$ROOT"
export PATH="$(dirname "$CA"):$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
export TMPDIR="$REPO/tmp_s1_4k_${PHASE}_sh${SHARD_ID}_${SLURM_JOB_ID:-local_$$}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
mkdir -p "$TMPDIR"
extra=()
if [[ "$PHASE" == matched ]]; then extra+=(--epsilon-map "$EPSMAP"); fi
"$CA" -u -m reviewer_controls.run_attack \
  --scenario s1 --split evaluation --method bandlimited_4khz \
  --epsilon 0.166 --alpha-ratio 0.02 --steps 1000 --seed 1 \
  --example-ids-file "$IDS_FILE" --run-tag "sharded_${PHASE}_${SHARD_ID}" \
  --resume --output-root "$OUT" "${extra[@]}"

