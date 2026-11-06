#!/bin/bash
# ============================================================================
# SLURM batch script for AAC-Robust CodecAttack on UMass Unity
#
# Runs latent-space attack with real AAC codec in the training loop (STE)
# plus IR convolution and background noise. Results saved to results_aac/.
#
# Usage:
#   sbatch sbatch_aac_robust.sh                                    # defaults
#   sbatch --export=MUSIC=jazz_1,EPS=5.0 sbatch_aac_robust.sh
#   sbatch --export=MUSIC=calm_1,EPS=3.0,STEPS=150 sbatch_aac_robust.sh
#   sbatch --export=QUICK=1 sbatch_aac_robust.sh                   # quick test
#
# Track:
#   squeue -u $USER
#   tail -f logs/aac-attack_<JOBID>.out
# ============================================================================

#SBATCH --job-name=aac-attack
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
EPS="${EPS:-5.0}"
MODE="${MODE:-qa}"
STEPS="${STEPS:-300}"
DATASET="${DATASET:-saycan}"
WARMUP="${WARMUP:-0.3}"
AAC_TRAIN_BR="${AAC_TRAIN_BR:-96,128,192,256}"
AAC_EVAL_BR="${AAC_EVAL_BR:-128,192}"
INCLUDE_MP3="${INCLUDE_MP3:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-1}"
BG_NOISE_SNR="${BG_NOISE_SNR:-30.0}"
QUICK="${QUICK:-0}"
TIMESHIFT_ONLY="${TIMESHIFT_ONLY:-0}"
TIME_SHIFT_MS="${TIME_SHIFT_MS:-60.0}"
SOUNDCLOUD_MODE="${SOUNDCLOUD_MODE:-0}"

# ---- Map model -> conda env ----
declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["qwen25_omni"]="qwen-omni"
    ["audio_flamingo"]="flamingo3"
)

CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'. Choose: qwen2_audio | qwen25_omni | audio_flamingo"
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
MP3_TAG=""
if [ "$INCLUDE_MP3" = "1" ]; then
    MP3_TAG="_mp3"
fi
TSONLY_TAG=""
if [ "$TIMESHIFT_ONLY" = "1" ]; then
    TSONLY_TAG="_tsonly"
fi
SC_TAG=""
if [ "$SOUNDCLOUD_MODE" = "1" ]; then
    SC_TAG="_sc"
fi
OUTPUT_DIR="${SCRIPT_DIR}/results_aac/benchmark_${MODEL}_${MODE}_eps${EPS}_${MUSIC}${TSONLY_TAG}${SC_TAG}_${STEPS}steps_all${MP3_TAG}${DATASET_TAG}"

# ---- Check existing progress ----
DONE=0
if [ -d "$OUTPUT_DIR" ]; then
    DONE=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
fi

# Build flags
MP3_FLAG=""
if [ "$INCLUDE_MP3" = "1" ]; then
    MP3_FLAG="--include-mp3"
fi

NOISE_FLAG="--train-with-noise"
if [ "$TRAIN_NOISE" = "0" ]; then
    NOISE_FLAG="--no-train-noise"
fi

QUICK_FLAG=""
if [ "$QUICK" = "1" ]; then
    QUICK_FLAG="--quick"
fi

TSONLY_FLAG=""
if [ "$TIMESHIFT_ONLY" = "1" ]; then
    TSONLY_FLAG="--timeshift-only"
fi

SC_FLAG=""
if [ "$SOUNDCLOUD_MODE" = "1" ]; then
    SC_FLAG="--soundcloud-mode"
fi

echo "============================================"
echo "  AAC-ROBUST ATTACK (SLURM)"
echo "============================================"
echo "  Job ID:         $SLURM_JOB_ID"
echo "  Node:           $SLURM_NODELIST"
echo "  GPU:            $CUDA_VISIBLE_DEVICES"
echo "  Model:          $MODEL"
echo "  Conda:          $CONDA_ENV"
echo "  Music:          $MUSIC"
echo "  Dataset:        $DATASET"
echo "  EPS:            $EPS"
echo "  Steps:          $STEPS"
echo "  Mode:           $MODE"
echo "  Warmup:         $WARMUP"
echo "  AAC train BR:   $AAC_TRAIN_BR"
echo "  AAC eval BR:    $AAC_EVAL_BR"
echo "  Include MP3:    $INCLUDE_MP3"
echo "  Train noise:    $TRAIN_NOISE (SNR=${BG_NOISE_SNR}dB)"
echo "  Timeshift only: $TIMESHIFT_ONLY"
echo "  SoundCloud mode:$SOUNDCLOUD_MODE"
echo "  Time shift ms:  $TIME_SHIFT_MS"
echo "  Output:         $OUTPUT_DIR"
echo "  Progress:       $DONE completed"
echo "============================================"

if [ "$DONE" -ge 200 ]; then
    echo "All experiments already completed. Nothing to do."
    exit 0
fi

# ---- Activate conda ----
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)')"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# ---- Run ----
python run_aac_robust_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --prompt-mode "$MODE" \
    --warmup-ratio "$WARMUP" \
    --aac-train-bitrates "$AAC_TRAIN_BR" \
    --aac-eval-bitrates "$AAC_EVAL_BR" \
    $NOISE_FLAG \
    --bg-noise-snr-db "$BG_NOISE_SNR" \
    --time-shift-ms "$TIME_SHIFT_MS" \
    --output-dir "$OUTPUT_DIR" \
    $MP3_FLAG \
    $QUICK_FLAG \
    $TSONLY_FLAG \
    $SC_FLAG \
    --quiet

# ---- Final status ----
FINAL=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  FINISHED: aac-robust / $MODEL / $MUSIC / eps=$EPS"
echo "  Completed: $FINAL experiments"
echo "============================================"
