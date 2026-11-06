#!/bin/bash
# Waveform-domain baseline attack benchmark
# Compares against latent-space attacks to show waveform perturbations
# do NOT survive Opus compression.
#
# Usage:
#   bash run_waveform_baseline.sh                              # Full run (qwen2_audio, empire_state_of_mind, saycan)
#   bash run_waveform_baseline.sh --quick                      # Quick 2-command test
#   bash run_waveform_baseline.sh --target-model qwen25_omni   # Use a different model
#
# Target model options (--target-model):
#   qwen2_audio   - Qwen2-Audio-7B-Instruct (default, conda: codec-attack)
#   qwen25_omni   - Qwen2.5-Omni-7B         (conda: qwen-omni)
#   audio_flamingo - Audio Flamingo 3         (conda: flamingo3)
#   kimi_audio    - Kimi-Audio-7B-Instruct   (conda: kimi-audio)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default settings
MUSIC="empire_state_of_mind"
DATASET="saycan"
EPS=0.5
STEPS=300
TARGET_MODEL="qwen2_audio"

# Parse --target-model from args to select the right conda env
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-model)
            TARGET_MODEL="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Select conda env based on target model
case "$TARGET_MODEL" in
    qwen2_audio)   CONDA_ENV="codec-attack" ;;
    qwen25_omni)   CONDA_ENV="qwen-omni" ;;
    audio_flamingo) CONDA_ENV="flamingo3" ;;
    *)
        echo "ERROR: Unknown model '$TARGET_MODEL'"
        echo "Options: qwen2_audio, qwen25_omni, audio_flamingo"
        exit 1
        ;;
esac

echo "============================================"
echo "Waveform Baseline Attack"
echo "============================================"
echo "Model:   $TARGET_MODEL"
echo "Conda:   $CONDA_ENV"
echo "Music:   $MUSIC"
echo "Dataset: $DATASET"
echo "Eps:     $EPS"
echo "Steps:   $STEPS"
echo "Extra:   ${EXTRA_ARGS[*]}"
echo "============================================"

conda run -n "$CONDA_ENV" python run_waveform_baseline.py \
    --music "$MUSIC" \
    --dataset "$DATASET" \
    --eps "$EPS" \
    --steps "$STEPS" \
    "${EXTRA_ARGS[@]}"
