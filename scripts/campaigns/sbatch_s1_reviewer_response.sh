#!/bin/bash
# Assemble the final report only after both result builders succeed.
#SBATCH --job-name=s1_review_report
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs"
cd "$ROOT"; export PYTHONPATH="$ROOT:$REPO"
"$PY" -m reviewer_controls.build_reviewer_response \
  --baseline-json "$OUT/s1_baseline_full40_seed1/baseline_report.json" \
  --baseline-markdown "$OUT/s1_baseline_full40_seed1/baseline_report.md" \
  --ensemble-json "$OUT/s1_ensemble_n123_seed1/ensemble_report.json" \
  --ensemble-markdown "$OUT/s1_ensemble_n123_seed1/ensemble_report.md" \
  --audit "$OUT/PROTOCOL_AUDIT_20260921.md" \
  --split-manifest "$OUT/S1_SPLIT_MANIFEST.csv" \
  --job-ledger "$OUT/JOB_LEDGER_20260921.json" \
  --reproduce "$OUT/REPRODUCE_20260921.md" \
  --output "$OUT/REVIEWER_RESPONSE_REPORT.md"
