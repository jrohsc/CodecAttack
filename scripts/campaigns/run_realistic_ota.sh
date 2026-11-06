#!/bin/bash
# ============================================================================
# Run Realistic OTA attack directly (no SLURM)
#
# Uses measured SoundCloud+airgap transfer function for channel simulation.
# Includes loudness norm, speaker nonlinearity, AGC, and freq-shaped perturbation.
#
# Usage:
#   bash run_realistic_ota.sh                          # defaults
#   EPS=10.0 bash run_realistic_ota.sh                 # higher eps
#   QUICK=1 bash run_realistic_ota.sh                  # quick test (3 commands)
#   NO_FREQ_SHAPE=1 bash run_realistic_ota.sh          # ablation: no freq shaping
# ============================================================================

MODEL="${MODEL:-qwen2_audio}"
MUSIC="${MUSIC:-calm_1}"
EPS="${EPS:-10.0}"
MODE="${MODE:-qa}"
STEPS="${STEPS:-500}"
DATASET="${DATASET:-saycan}"
WARMUP="${WARMUP:-0.3}"
NOISE_SNR="${NOISE_SNR:-25.0}"
N_EOT="${N_EOT:-4}"
CHANNEL_RATIO="${CHANNEL_RATIO:-0.8}"
NO_FREQ_SHAPE="${NO_FREQ_SHAPE:-0}"
QUICK="${QUICK:-0}"

declare -A CONDA_ENVS=(
    ["qwen2_audio"]="codec-attack"
    ["qwen25_omni"]="qwen-omni"
    ["audio_flamingo"]="flamingo3"
)
CONDA_ENV="${CONDA_ENVS[$MODEL]}"
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: Unknown model '$MODEL'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FS_FLAG=""
if [ "$NO_FREQ_SHAPE" = "1" ]; then
    FS_FLAG="--no-freq-shape"
fi
QUICK_FLAG=""
if [ "$QUICK" = "1" ]; then
    QUICK_FLAG="--quick"
fi

echo "============================================"
echo "  REALISTIC OTA ATTACK"
echo "============================================"
echo "  Model:       $MODEL"
echo "  Music:       $MUSIC"
echo "  EPS:         $EPS"
echo "  Steps:       $STEPS"
echo "  Freq shape:  $([ "$NO_FREQ_SHAPE" = "1" ] && echo OFF || echo ON)"
echo "  EoT:         $N_EOT"
echo "  Ch ratio:    $CHANNEL_RATIO"
echo "  Noise SNR:   $NOISE_SNR dB"
echo "============================================"

source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

python run_realistic_ota_benchmark.py \
    --target-model "$MODEL" \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --prompt-mode "$MODE" \
    --warmup-ratio "$WARMUP" \
    --noise-snr-db "$NOISE_SNR" \
    --n-eot "$N_EOT" \
    --channel-ratio "$CHANNEL_RATIO" \
    $FS_FLAG \
    $QUICK_FLAG

echo ""
echo "DONE"
