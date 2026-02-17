#!/bin/bash
# ============================================================================
# SLURM batch script for CodecAttack on UMass Unity (gpupod-l40s)
#
# Runs latent-space attack on a single (model, music, eps) configuration.
# Each job uses 1 L40S GPU and processes all 200 commands sequentially.
# Auto-resumes from previous progress if the output directory exists.
#
# Usage:
#   sbatch sbatch_attack.sh                          # defaults: qwen2_audio, empire_state_of_mind, eps=0.5, qa
#   sbatch --export=MODEL=audio_flamingo,EPS=0.5 sbatch_attack.sh
#   sbatch --export=MODEL=kimi_audio,EPS=1.0 sbatch_attack.sh
#
# Submit all 6 jobs at once:
#   for EPS in 0.5 1.0; do
#     for MODEL in qwen2_audio audio_flamingo kimi_audio; do
#       sbatch --export=MODEL=$MODEL,EPS=$EPS,MUSIC=empire_state_of_mind,MODE=qa sbatch_attack.sh
#     done
#   done
#
# Track progress:
#   ./run.sh --status results_qa/benchmark_qwen2_audio_eps0.5_empire_state_of_mind
#   squeue -u $USER                    # check running jobs
#   tail -f logs/ca_*_<JOBID>.out      # follow live output
#   ls logs/                           # all job logs
# ============================================================================

#SBATCH --job-name=codec-attack
#SBATCH --partition=gpupod-l40s
#SBATCH --account=pi_ahoumansadr_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Defaults (override via --export) ----
MODEL="${MODEL:-qwen2_audio}"
MUSIC="${MUSIC:-empire_state_of_mind}"
EPS="${EPS:-0.5}"
MODE="${MODE:-qa}"
STEPS="${STEPS:-150}"

# ---- Map model → conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["audio_flamingo"]="flamingo3"
    ["kimi_audio"]="kimi-audio"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'. Choose: qwen2_audio | audio_flamingo | kimi_audio"
    exit 1
fi

# ---- Setup ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

OUTPUT_DIR="${SCRIPT_DIR}/results_${MODE}/benchmark_${MODEL}_eps${EPS}_${MUSIC}"

# ---- Check existing progress ----
DONE=0
if [ -d "$OUTPUT_DIR" ]; then
    DONE=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
fi

echo "============================================"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURM_NODELIST"
echo "  GPU:       $CUDA_VISIBLE_DEVICES"
echo "  Model:     $MODEL"
echo "  Conda:     $CONDA_ENV"
echo "  Music:     $MUSIC"
echo "  EPS:       $EPS"
echo "  Mode:      $MODE"
echo "  Steps:     $STEPS"
echo "  Output:    $OUTPUT_DIR"
echo "  Progress:  $DONE / 200 completed"
echo "============================================"

if [ "$DONE" -ge 200 ]; then
    echo "All 200 experiments already completed. Nothing to do."
    exit 0
fi

# ---- Activate conda ----
source /work/pi_ahoumansadr_umass_edu/jroh/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# ---- Resume flag ----
RESUME_FLAG=""
if [ "$DONE" -gt 0 ]; then
    echo "Resuming from $DONE completed experiments..."
    RESUME_FLAG="--resume $OUTPUT_DIR"
fi

# ---- Run ----
python run_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --prompt-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --quiet \
    $RESUME_FLAG

# ---- Final status ----
FINAL=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  FINISHED: $MODEL / $MUSIC / eps=$EPS / $MODE"
echo "  Completed: $FINAL / 200"
echo "============================================"
