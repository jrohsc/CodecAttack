#!/usr/bin/env bash
# MIMI smoke v0: one-pair end-to-end attack on Qwen2.5-Omni / S3b pair 0.
# See 0_all_combined/results_MIMI/README.md for context.
#
# Usage:
#   bash 0_all_combined/scripts/smoke_mimi_v0.sh
#
# Prereqs:
#   - codec-attack conda env with transformers MimiModel available
#   - eps_mimi_v0.txt in results_MIMI/.../smoke/
set -eu

ROOT="${REPO_ROOT}"
LOG_DIR="$ROOT/0_all_combined/scripts/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/smoke_mimi_v0_${TS}.log"

cd "$ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack

echo "=== smoke_mimi_v0 starting at $(date) ===" | tee "$LOG"
python "$ROOT/0_all_combined/scripts/smoke_mimi_v0.py" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== exit $RC ===" | tee -a "$LOG"
exit "$RC"
