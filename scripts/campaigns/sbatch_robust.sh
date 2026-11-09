#!/bin/bash
# ============================================================================
# SLURM batch script for Channel-Robust CodecAttack on a SLURM GPU cluster
#
# Runs robust latent-space attack (EoT + channel augmentation) on a single
# (model, music, eps) configuration. Auto-resumes from previous progress.
#
# Usage:
#   sbatch sbatch_robust.sh                                    # defaults
#   sbatch --export=MUSIC=jazz_1,EPS=0.4 sbatch_robust.sh
#   sbatch --export=N_EOT=2,STEPS=150 sbatch_robust.sh        # quick test
#
# Track:
#   squeue -u $USER
#   tail -f logs/robust-attack_<JOBID>.out
# ============================================================================

#SBATCH --job-name=robust-attack
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- Defaults (override via --export) ----
MODEL="${MODEL:-qwen2_audio}"
MUSIC="${MUSIC:-jazz_1}"
EPS="${EPS:-0.4}"
MODE="${MODE:-qa}"
STEPS="${STEPS:-300}"
DATASET="${DATASET:-ours}"

# Robust-specific
N_EOT="${N_EOT:-4}"
SEVERITY="${SEVERITY:-1.0}"
CURRICULUM="${CURRICULUM:-1}"        # 1=enabled, 0=disabled
FREQ_SHAPING="${FREQ_SHAPING:-0}"    # 1=enabled, 0=disabled
QUICK="${QUICK:-0}"                  # 1=quick test (2 commands)

# ---- Map model → conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["qwen25_omni"]="qwen-omni"
    ["audio_flamingo"]="flamingo3"
    ["kimi_audio"]="kimi-audio"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'. Choose: qwen2_audio | qwen25_omni | audio_flamingo | kimi_audio"
    exit 1
fi

# ---- Setup ----
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

# Build output directory name
DATASET_TAG=""
if [ "$DATASET" != "ours" ]; then
    DATASET_TAG="_${DATASET}"
fi
OUTPUT_DIR="${SCRIPT_DIR}/results_airgap/benchmark_${MODEL}_eps${EPS}_${MUSIC}_eot${N_EOT}${DATASET_TAG}"

# ---- Check existing progress ----
DONE=0
if [ -d "$OUTPUT_DIR" ]; then
    DONE=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
fi

# Build flags
CURRICULUM_FLAG=""
if [ "$CURRICULUM" = "1" ]; then
    CURRICULUM_FLAG="--channel-curriculum"
fi

FREQ_SHAPING_FLAG=""
if [ "$FREQ_SHAPING" = "1" ]; then
    FREQ_SHAPING_FLAG="--freq-shaping"
fi

QUICK_FLAG=""
if [ "$QUICK" = "1" ]; then
    QUICK_FLAG="--quick"
fi

echo "============================================"
echo "  ROBUST ATTACK (SLURM)"
echo "============================================"
echo "  Job ID:      $SLURM_JOB_ID"
echo "  Node:        $SLURM_NODELIST"
echo "  GPU:         $CUDA_VISIBLE_DEVICES"
echo "  Model:       $MODEL"
echo "  Conda:       $CONDA_ENV"
echo "  Music:       $MUSIC"
echo "  Dataset:     $DATASET"
echo "  EPS:         $EPS"
echo "  Steps:       $STEPS"
echo "  Mode:        $MODE"
echo "  EoT samples: $N_EOT"
echo "  Severity:    $SEVERITY"
echo "  Curriculum:  $CURRICULUM"
echo "  Freq shape:  $FREQ_SHAPING"
echo "  Output:      $OUTPUT_DIR"
echo "  Progress:    $DONE completed"
echo "============================================"

if [ "$DONE" -ge 200 ]; then
    echo "All 200 experiments already completed. Nothing to do."
    exit 0
fi

# ---- Activate conda ----
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
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
python run_robust_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --prompt-mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --n-eot-samples "$N_EOT" \
    --channel-severity "$SEVERITY" \
    $CURRICULUM_FLAG \
    $FREQ_SHAPING_FLAG \
    $QUICK_FLAG \
    --quiet \
    $RESUME_FLAG

# ---- Final status ----
FINAL=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  FINISHED: robust / $MODEL / $MUSIC / eps=$EPS"
echo "  Completed: $FINAL / 200"
echo "============================================"
