#!/bin/bash
# Empirical OTA IR Experiment 1: Sanity check
#
# Uses ACTUAL measured IR + noise from MacBook Air -> iPhone 16 Pro
# instead of synthetic RIRs or SpecAugment proxies.
#
# Single command test: does loss converge through the real channel?

set -e

EPS=${EPS:-3.0}
STEPS=${STEPS:-1000}
MUSIC=${MUSIC:-calm_1}
WARMUP=${WARMUP:-0.3}
N_EOT=${N_EOT:-4}

echo "=== Empirical OTA IR Experiment 1 (Sanity Check) ==="
echo "eps=$EPS, steps=$STEPS, music=$MUSIC, EoT=$N_EOT"
echo "Channel: measured IR + 21 real noise residuals"
echo "===================================================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode empirical_ota \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --warmup-ratio "$WARMUP" \
    --n-eot-samples "$N_EOT" \
    --prompt-mode qa \
    --commands nav_turn_left nav_turn_right \
    --output-dir results_empirical_ota_IR/exp1_eps${EPS}_${MUSIC}_eot${N_EOT}_${STEPS}steps \
    "$@"
