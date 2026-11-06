#!/bin/bash
# Test: spec_ota with gradient accumulation (EoT simulation)
#
# Runs 2 SayCan commands with grad-accum=5, 10000 total steps
# (= 2000 effective updates, each averaging 5 random augmentations)
#
# Estimated time: ~1.1 hrs per command, ~2.2 hrs total

set -e

EPS=3.0
STEPS=10000
MUSIC=calm_1
WARMUP=0.3
N_MASK=10
MASK_SIZE=50
NOISE_EPS=0.02
GRAD_ACCUM=5

echo "=== SpecAugment OTA + Gradient Accumulation Test ==="
echo "eps=$EPS, steps=$STEPS (effective: $((STEPS / GRAD_ACCUM)))"
echo "grad_accum=$GRAD_ACCUM"
echo "SpecAugment: n_mask=$N_MASK, mask_size=$MASK_SIZE"
echo "Noise eps: $NOISE_EPS"
echo "Warmup ratio: $WARMUP"
echo "====================================================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode spec_ota \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --warmup-ratio "$WARMUP" \
    --spec-augment-n-mask "$N_MASK" \
    --spec-augment-mask-size "$MASK_SIZE" \
    --spec-augment-noise-eps "$NOISE_EPS" \
    --grad-accum "$GRAD_ACCUM" \
    --prompt-mode qa \
    --dataset saycan \
    --commands saycan_000 saycan_001 \
    --output-dir results_spec_ota_gradaccum \
    "$@"
