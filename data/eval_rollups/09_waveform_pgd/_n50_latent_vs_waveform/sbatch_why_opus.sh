#!/bin/bash
# Signal-level "why Opus" analysis. CPU-only; requests a GPU slot only to get a
# node allocation, and runs the codecs on CPU (--device cpu) as validated locally.
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="${OUT:?set OUT}"
# ARMS optional: comma list of arm names to restrict to (empty/unset = all arms,
# the previous behaviour). Used to re-run just the EnCodec arm after seed 7 lands.
ARMS="${ARMS:-}"
echo "=== why-opus analysis  job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local} ==="
echo "    out=$OUT  arms=${ARMS:-<all>}"
$PY -u 0_all_combined/scripts/analyze_why_opus.py \
    --seeds 1,7,42 --neural dac,encodec --device cpu --out "$OUT" \
    ${ARMS:+--arms "$ARMS"}
