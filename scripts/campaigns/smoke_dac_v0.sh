#!/usr/bin/env bash
# DAC smoke v0: one-pair end-to-end attack on Qwen2.5-Omni / S3b pair 0.
# See 0_all_combined/results_DAC/PLAN_smoke.md for full context.
#
# Usage:
#   bash 0_all_combined/scripts/smoke_dac_v0.sh
#
# Prereqs:
#   - codec-attack conda env with descript-audio-codec installed (task 1)
#   - eps_dac_v0.txt in results_DAC/.../smoke/ (task 4)
set -eu

ROOT="${REPO_ROOT}"
LOG_DIR="$ROOT/0_all_combined/scripts/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/smoke_dac_v0_${TS}.log"

cd "$ROOT"

# Activate the env. `conda activate` requires the conda init hook to be sourced.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack

echo "=== smoke_dac_v0 starting at $(date) ===" | tee "$LOG"
python "$ROOT/0_all_combined/scripts/smoke_dac_v0.py" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== exit $RC ===" | tee -a "$LOG"
exit "$RC"
