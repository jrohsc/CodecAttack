#!/bin/bash
# SpecAugment OTA attack experiment
#
# Uses SpecAugment (random mel frequency band masking) + temporal shift +
# additive noise during optimization to achieve OTA robustness without
# needing accurate channel modeling (RIRs, bandpass, etc.).
#
# Inspired by "Attacker's Noise Can Manipulate Your Audio-based LLM in the
# Real World" (2025), but combined with our latent-space optimization for
# codec robustness — achieving robustness to BOTH codec and OTA channels.
#
# Usage:
#   bash run_spec_ota.sh              # Quick test (2 commands)
#   bash run_spec_ota.sh --full       # Full benchmark (30 SayCan commands)

set -e

# Default parameters (matching the paper's recipe)
EPS=${EPS:-5.0}
STEPS=${STEPS:-1000}
MUSIC=${MUSIC:-calm_1}
WARMUP=${WARMUP:-0.3}
TIME_SHIFT=${TIME_SHIFT:-100}
N_MASK=${N_MASK:-10}
MASK_SIZE=${MASK_SIZE:-50}
NOISE_EPS=${NOISE_EPS:-0.02}

# Parse --full flag
DATASET_ARGS="--quick"
for arg in "$@"; do
    if [ "$arg" == "--full" ]; then
        DATASET_ARGS="--full --dataset saycan"
    fi
done

echo "=== SpecAugment OTA Attack ==="
echo "eps=$EPS, steps=$STEPS, music=$MUSIC"
echo "SpecAugment: n_mask=$N_MASK, mask_size=$MASK_SIZE"
echo "Noise eps: $NOISE_EPS, Time shift: ${TIME_SHIFT}ms"
echo "Warmup ratio: $WARMUP"
echo "=============================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode spec_ota \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --warmup-ratio "$WARMUP" \
    --time-shift-ms "$TIME_SHIFT" \
    --spec-augment-n-mask "$N_MASK" \
    --spec-augment-mask-size "$MASK_SIZE" \
    --spec-augment-noise-eps "$NOISE_EPS" \
    --prompt-mode qa \
    $DATASET_ARGS \
    "$@"
