#!/bin/bash
# ============================================================================
# Print all experiment commands for the full benchmark grid.
#
# Full grid: 3 models × 3 eps × 4 carriers = 36 runs, 200 commands each.
#
# Usage:
#   ./run_all.sh              # Print all 36 commands
#   ./run_all.sh --pending    # Print only experiments not yet started/completed
#   ./run_all.sh --launch N   # Launch N experiments in background (tmux sessions)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Experiment grid ----
MODELS=("qwen2_audio" "audio_flamingo" "kimi_audio")
EPS_VALUES=("0.3" "0.5" "1.0")
CARRIERS=("empire_state_of_mind" "jazz_1" "calm_1" "classical_music_1")
MODE="qa"
STEPS="150"

# ---- Parse args ----
PENDING_ONLY=""
LAUNCH_N=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --pending|-p)   PENDING_ONLY="1"; shift;;
        --launch|-l)    LAUNCH_N="$2"; shift 2;;
        --help|-h)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pending, -p    Only show experiments that are not yet complete"
            echo "  --launch N, -l N Launch N experiments in tmux sessions"
            echo ""
            echo "Grid: ${#MODELS[@]} models × ${#EPS_VALUES[@]} eps × ${#CARRIERS[@]} carriers = $(( ${#MODELS[@]} * ${#EPS_VALUES[@]} * ${#CARRIERS[@]} )) total"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ---- Check completion status of a run ----
check_done() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "0"
        return
    fi
    find "$dir" -name "record.json" 2>/dev/null | wc -l
}

# ---- Print grid ----
TOTAL=0
DONE_TOTAL=0
PENDING_CMDS=()

echo "============================================================================"
echo "  FULL EXPERIMENT GRID: ${#MODELS[@]} models × ${#EPS_VALUES[@]} eps × ${#CARRIERS[@]} carriers = $(( ${#MODELS[@]} * ${#EPS_VALUES[@]} * ${#CARRIERS[@]} )) runs"
echo "  200 commands per run | mode=$MODE | steps=$STEPS"
echo "============================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo "--- $MODEL ---"
    for EPS in "${EPS_VALUES[@]}"; do
        for CARRIER in "${CARRIERS[@]}"; do
            TOTAL=$((TOTAL + 1))
            OUTPUT_DIR="results_${MODE}/benchmark_${MODEL}_eps${EPS}_${CARRIER}"
            DONE=$(check_done "$OUTPUT_DIR")
            DONE_TOTAL=$((DONE_TOTAL + DONE))

            if [ "$DONE" -ge 200 ]; then
                STATUS="DONE (${DONE}/200)"
            elif [ "$DONE" -gt 0 ]; then
                STATUS="PARTIAL (${DONE}/200)"
            else
                STATUS="PENDING"
            fi

            CMD="./run.sh -m $MODEL -c $CARRIER -e $EPS -s $STEPS"

            if [ -n "$PENDING_ONLY" ] && [ "$DONE" -ge 200 ]; then
                continue
            fi

            printf "  %-12s  eps=%-4s  %-25s  %s\n" "$STATUS" "$EPS" "$CARRIER" "$CMD"

            if [ "$DONE" -lt 200 ]; then
                PENDING_CMDS+=("$CMD")
            fi
        done
    done
    echo ""
done

echo "============================================================================"
echo "  Total: $TOTAL runs | Done: $(( DONE_TOTAL / 200 )) complete | Pending: ${#PENDING_CMDS[@]}"
echo "  Total experiments: $((TOTAL * 200)) | Completed: $DONE_TOTAL"
echo "============================================================================"

# ---- Launch mode ----
if [ -n "$LAUNCH_N" ]; then
    if ! command -v tmux &> /dev/null; then
        echo ""
        echo "tmux not found. Copy-paste these commands into separate terminals:"
        echo ""
        for i in $(seq 0 $((LAUNCH_N - 1))); do
            if [ $i -lt ${#PENDING_CMDS[@]} ]; then
                echo "  ${PENDING_CMDS[$i]}"
            fi
        done
        exit 0
    fi

    echo ""
    echo "Launching $LAUNCH_N experiments in tmux sessions..."
    for i in $(seq 0 $((LAUNCH_N - 1))); do
        if [ $i -lt ${#PENDING_CMDS[@]} ]; then
            SESSION="exp_${i}"
            echo "  tmux: $SESSION → ${PENDING_CMDS[$i]}"
            tmux new-session -d -s "$SESSION" "cd $SCRIPT_DIR && ${PENDING_CMDS[$i]}; echo 'DONE'; read"
        fi
    done
    echo ""
    echo "Monitor with: tmux ls"
    echo "Attach with:  tmux attach -t exp_0"
fi

# ---- If not launching, print copy-paste block ----
if [ -z "$LAUNCH_N" ] && [ ${#PENDING_CMDS[@]} -gt 0 ]; then
    echo ""
    echo "Copy-paste into separate terminals (add -g GPU_ID to pin GPUs):"
    echo ""
    for cmd in "${PENDING_CMDS[@]}"; do
        echo "  $cmd"
    done
fi
