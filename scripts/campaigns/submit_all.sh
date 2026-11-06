#!/bin/bash
# ============================================================================
# Submit all attack jobs to SLURM.
#
# Usage:
#   ./submit_all.sh              # Submit all 6 jobs (3 models × 2 eps)
#   ./submit_all.sh --dry-run    # Print commands without submitting
#   ./submit_all.sh --eps 0.5    # Only eps=0.5 (3 jobs)
#   ./submit_all.sh --model qwen2_audio  # Only one model (2 jobs)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# ---- Defaults ----
MODELS=("qwen2_audio" "audio_flamingo" "kimi_audio")
EPS_VALUES=("0.3" "0.5" "1.0")
MUSIC="empire_state_of_mind"
MODE="transcribe" # qa or transcribe
STEPS="150"
DRY_RUN=""
USE_SRUN=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-n)   DRY_RUN="1"; shift;;
        --srun)         USE_SRUN="1"; shift;;
        --model|-m)     MODELS=("$2"); shift 2;;
        --eps|-e)       EPS_VALUES=("$2"); shift 2;;
        --music|-c)     MUSIC="$2"; shift 2;;
        --mode)         MODE="$2"; shift 2;;
        --steps|-s)     STEPS="$2"; shift 2;;
        --help|-h)
            echo "Usage: ./submit_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n    Print sbatch commands without submitting"
            echo "  --model, -m      Single model (default: all three)"
            echo "  --eps, -e        Single eps value (default: 0.5 and 1.0)"
            echo "  --music, -c      Music carrier (default: empire_state_of_mind)"
            echo "  --mode           Prompt mode: qa | transcribe (default: qa)"
            echo "  --steps, -s      Attack steps (default: 150)"
            echo "  --srun           Use srun (interactive, sequential) instead of sbatch"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  Models:  ${MODELS[*]}"
echo "  Eps:     ${EPS_VALUES[*]}"
echo "  Music:   $MUSIC"
echo "  Mode:    $MODE"
echo "  Steps:   $STEPS"
echo "  Jobs:    $(( ${#MODELS[@]} * ${#EPS_VALUES[@]} ))"
echo "============================================"
echo ""

for EPS in "${EPS_VALUES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="ca_${MODEL}_e${EPS}_${MUSIC}"

        if [ -n "$USE_SRUN" ]; then
            CMD="srun --partition=gpu-preempt --account=${PI_GROUP} --gres=gpu:1 \"--constraint=a100|l40s|a40\" --mem=60G --cpus-per-task=8 --time=24:00:00 --job-name=$JOB_NAME --export=ALL,MODEL=$MODEL,EPS=$EPS,MUSIC=$MUSIC,MODE=$MODE,STEPS=$STEPS bash sbatch_attack.sh"
        else
            CMD="sbatch --job-name=$JOB_NAME --export=MODEL=$MODEL,EPS=$EPS,MUSIC=$MUSIC,MODE=$MODE,STEPS=$STEPS sbatch_attack.sh"
        fi

        if [ -n "$DRY_RUN" ]; then
            echo "[DRY RUN] $CMD"
        else
            echo "Running: $MODEL / eps=$EPS / $MUSIC"
            $CMD
        fi
    done
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      logs/"
