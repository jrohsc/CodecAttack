#!/bin/bash
# Post-hoc exact-metric audit of the completed N=1/N=2/N=3 S1 bundles.
#SBATCH --job-name=s1_ens_report
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO=${REPO_ROOT}
ROOT="$REPO/0_all_combined"
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OUT="$ROOT/reviewer_control_outputs/s1_ensemble_n123_seed1"
cd "$ROOT"
mkdir -p "$OUT" "$REPO/logs"
export PYTHONPATH="$ROOT:$REPO"

"$PY" -m reviewer_controls.build_ensemble_report \
  --n1 results_s1_opuseot_eps10_pool_scaling/01_finance_voice_agent/n1_qwen2/eps_10_seed1 \
  --n2 results_s1_opuseot_eps10_pool_scaling/01_finance_voice_agent/n2_qwen2_omni/eps_10_seed1 \
  --n3 results_s1_ens3_eps10_codec_compare/01_finance_voice_agent/opus_eot/eps_10_seed1 \
  --output-json "$OUT/ensemble_report.json" \
  --output-markdown "$OUT/ensemble_report.md"

