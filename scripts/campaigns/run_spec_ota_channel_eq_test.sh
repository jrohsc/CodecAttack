#!/bin/bash
# Frequency-constrained spec_ota attack using measured channel response
#
# Three-pronged approach:
# 1. Bandpass on perturbation (500-8kHz) — only allow energy in speaker-friendly bands
# 2. Measured channel EQ filter — simulate the MacBook→iPhone freq response
# 3. SpecAugment + translation + noise — stochastic augmentation for robustness
#
# Uses --channel-only (no warmup) since every step needs channel augmentation
# Combined with grad-accum=5 for EoT-like robustness.
#
# Estimated time: ~1.1 hrs per command

set -e

EPS=3.0
STEPS=10000
MUSIC=calm_1
N_MASK=10
MASK_SIZE=50
NOISE_EPS=0.02
GRAD_ACCUM=5
CHANNEL_RESPONSE=results_spec_ota/measured_channel_response.npz

# Bandpass: constrain perturbation to bands the MacBook speaker reproduces well
# Channel analysis shows: bass (-2dB), mid (+6dB), upper-mid (+8dB), presence (+10dB)
BANDPASS_LOW=500
BANDPASS_HIGH=8000

echo "=== Frequency-Constrained SpecAugment OTA Attack ==="
echo "eps=$EPS, steps=$STEPS (effective: $((STEPS / GRAD_ACCUM)))"
echo "grad_accum=$GRAD_ACCUM"
echo "Bandpass: $BANDPASS_LOW - $BANDPASS_HIGH Hz"
echo "Channel response: $CHANNEL_RESPONSE"
echo "SpecAugment: n_mask=$N_MASK, mask_size=$MASK_SIZE"
echo "Mode: channel-only (no warmup)"
echo "====================================================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode spec_ota \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --channel-only \
    --bandpass-low "$BANDPASS_LOW" \
    --bandpass-high "$BANDPASS_HIGH" \
    --spec-augment-n-mask "$N_MASK" \
    --spec-augment-mask-size "$MASK_SIZE" \
    --spec-augment-noise-eps "$NOISE_EPS" \
    --grad-accum "$GRAD_ACCUM" \
    --channel-response "$CHANNEL_RESPONSE" \
    --prompt-mode qa \
    --dataset saycan \
    --commands saycan_000 saycan_001 \
    --output-dir results_spec_ota_channel_eq \
    "$@"
