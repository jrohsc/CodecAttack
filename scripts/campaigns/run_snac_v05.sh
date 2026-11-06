#!/usr/bin/env bash
# SNAC v0.5: 5-pair x 4-channel multi-pair sanity run, mirroring DAC v0.5.
# eps_snac=2.0 (smoke-validated on a_thousand_years_lyrical / Qwen2-Audio).
#
# Usage:
#   bash 0_all_combined/scripts/run_snac_v05.sh
set -eu

ROOT="${REPO_ROOT}"
LOG_DIR="$ROOT/0_all_combined/scripts/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/run_snac_v05_${TS}.log"

cd "$ROOT"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack

echo "=== run_snac_v05 starting at $(date) ===" | tee "$LOG"
python -u "$ROOT/0_all_combined/scripts/run_snac_v05.py" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== exit $RC ===" | tee -a "$LOG"
exit "$RC"
