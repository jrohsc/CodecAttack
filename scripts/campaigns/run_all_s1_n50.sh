#!/usr/bin/env bash
# Sequential driver for all 9 S1 cells (3 models x 3 eps) at n=50.
# Calls run_s1_cell.sh for each cell in turn. Skip-existing logic in the
# attack/sidecar drivers makes this idempotent -- re-running only does the
# missing work.
#
# Usage:
#   bash 0_all_combined/scripts/run_all_s1_n50.sh                # all 9 cells
#   bash 0_all_combined/scripts/run_all_s1_n50.sh qwen2_audio    # one model, all eps
#   EPS_LIST="0.5 1.0" bash 0_all_combined/scripts/run_all_s1_n50.sh
#
# To run two cells in parallel from separate terminals instead, prefer the
# single-cell launcher:
#   bash 0_all_combined/scripts/run_s1_cell.sh qwen2_audio    1.0
#   bash 0_all_combined/scripts/run_s1_cell.sh audio_flamingo 1.0
# (different models in different envs share the one A100; two 7B attacks fit.)
set -uo pipefail

cd ${REPO_ROOT}

MODEL_LIST="${1:-qwen2_audio qwen25_omni audio_flamingo}"
EPS_LIST="${EPS_LIST:-0.5 1.0 1.5}"

LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/run_all_s1_n50.master.log"

echo "[$(date +%F\ %H:%M:%S)] START -- models='$MODEL_LIST' eps='$EPS_LIST'" | tee -a "$MASTER_LOG"

FAILED=()
for MODEL in $MODEL_LIST; do
    for EPS in $EPS_LIST; do
        echo "" | tee -a "$MASTER_LOG"
        echo "[$(date +%F\ %H:%M:%S)] CELL  model=$MODEL  eps=$EPS" | tee -a "$MASTER_LOG"
        bash 0_all_combined/scripts/run_s1_cell.sh "$MODEL" "$EPS" 2>&1 | tee -a "$MASTER_LOG"
        rc=${PIPESTATUS[0]}
        if (( rc != 0 )); then
            echo "[$(date +%F\ %H:%M:%S)] CELL FAILED rc=$rc -- continuing" | tee -a "$MASTER_LOG"
            FAILED+=("$MODEL/eps_$EPS")
        fi
    done
done

echo "" | tee -a "$MASTER_LOG"
if (( ${#FAILED[@]} == 0 )); then
    echo "[$(date +%F\ %H:%M:%S)] ALL CELLS OK" | tee -a "$MASTER_LOG"
else
    echo "[$(date +%F\ %H:%M:%S)] FAILED CELLS: ${FAILED[*]}" | tee -a "$MASTER_LOG"
fi

echo ""
echo "Next steps after attacks finish:"
echo "  1. Rebuild rollups:    python 0_all_combined/scripts/rollup_results.py"
echo "  2. Bump table to n=50: edit 0_all_combined/scripts/build_main_results_tables.py"
echo "       SCENARIOS['S1']['n_target'] = 50"
echo "       drop the keep_s1_5voice filter (or leave a 5-voice ablation row)"
echo "  3. Regenerate table:   python 0_all_combined/scripts/build_main_results_tables.py --scenario S1"
