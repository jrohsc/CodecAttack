#!/usr/bin/env bash
# MIMI EoT full: 45-pair x 15-channel attack on Qwen2.5-Omni / S3b copyright bypass.
# See 0_all_combined/results_OTHER_NEURAL_CODECS/results_DAC/README.md for design (MIMI mirrors DAC).
#
# Usage:
#   bash 0_all_combined/scripts/run_mimi_full.sh                  # full mode
#   bash 0_all_combined/scripts/run_mimi_full.sh --mode smoke     # 5-pair smoke
#   bash 0_all_combined/scripts/run_mimi_full.sh --pairs 0        # single-pair sanity
set -eu

ROOT="${REPO_ROOT}"
LOG_DIR="$ROOT/0_all_combined/scripts/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/run_mimi_full_${TS}.log"

# Default to full mode; pass-through any user-supplied flags
ARGS=("--mode" "full")
if [ "$#" -gt 0 ]; then
    ARGS=("$@")
fi

cd "$ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack

echo "=== run_mimi_full starting at $(date) ===" | tee "$LOG"
echo "    args: ${ARGS[*]}" | tee -a "$LOG"
python -u "$ROOT/0_all_combined/scripts/run_mimi_full.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== exit $RC at $(date) ===" | tee -a "$LOG"
exit "$RC"
