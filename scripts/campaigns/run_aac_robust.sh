#!/bin/bash
# ============================================================================
# Run AAC-Robust CodecAttack directly (no SLURM)
#
# Usage:
#   bash run_aac_robust.sh                                          # defaults
#   MUSIC=calm_1 EPS=3.0 bash run_aac_robust.sh
#   MUSIC=jazz_1 EPS=5.0 INCLUDE_MP3=1 bash run_aac_robust.sh
#   QUICK=1 bash run_aac_robust.sh                                  # quick test
# ============================================================================

# ---- Defaults (override via env vars) ----
MODEL="${MODEL:-audio_flamingo}"
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build output directory name
DATASET_TAG=""
if [ "$DATASET" != "ours" ]; then
    DATASET_TAG="_${DATASET}"
fi
MP3_TAG=""
if [ "$INCLUDE_MP3" = "1" ]; then
    MP3_TAG="_mp3"
fi
OUTPUT_DIR="${SCRIPT_DIR}/results_aac/benchmark_${MODEL}_${MODE}_eps${EPS}_${MUSIC}_aac${MP3_TAG}${DATASET_TAG}"

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

echo "============================================"
echo "  AAC-ROBUST ATTACK (direct)"
echo "============================================"
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
echo "  Output:         $OUTPUT_DIR"
echo "  Progress:       $DONE completed"
echo "============================================"

# ---- Activate conda ----
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
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
    --output-dir "$OUTPUT_DIR" \
    $MP3_FLAG \
    $QUICK_FLAG

echo ""
echo "============================================"
FINAL=$(find "$OUTPUT_DIR" -name "record.json" 2>/dev/null | wc -l)
echo "  DONE: $FINAL experiments in $OUTPUT_DIR"
echo "============================================"
