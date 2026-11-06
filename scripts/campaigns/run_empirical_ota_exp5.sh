#!/bin/bash
# Experiment 5: Robust FIR + Phase Jitter + High EoT
#
# Key insight from exp3 acoustic analysis: physical channel doesn't just
# attenuate — it INVERTS phase (correlation = -0.32). Our FIR model only
# captures magnitude response, not phase destruction.
#
# This experiment adds:
#   - Random per-band phase rotation (±π radians = full phase randomization)
#   - Per-band magnitude jitter (±3dB)
#   - Global gain jitter (±3dB)
#   - Higher EoT (20 instead of 4) for stable gradients
#   - Time shift + additive noise + SpecAugment
#   - Real noise residuals
#
# Goal: force perturbation to be robust to arbitrary phase destruction,
# not just magnitude changes.

set -e

EPS="${EPS:-3.0}"
STEPS="${STEPS:-1000}"
MUSIC="${MUSIC:-calm_1}"
WARMUP="${WARMUP:-0.3}"
N_EOT="${N_EOT:-20}"
BAND_JITTER="${BAND_JITTER:-3.0}"
GAIN_JITTER="${GAIN_JITTER:-3.0}"
PHASE_JITTER="${PHASE_JITTER:-3.14159}"  # pi radians = full random phase

OUTPUT_DIR="results_empirical_ota_IR/5_exp_robust_fir_phase"

echo "=== Experiment 5: Robust FIR + Phase Jitter ==="
echo "eps=$EPS, steps=$STEPS, music=$MUSIC, EoT=$N_EOT"
echo "Channel: PSD FIR + band_jitter=±${BAND_JITTER}dB + gain_jitter=±${GAIN_JITTER}dB"
echo "         + phase_jitter=±${PHASE_JITTER}rad + SpecAugment + time_shift"
echo "Output: $OUTPUT_DIR"
echo "================================================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode empirical_ota \
    --robust-fir \
    --robust-fir-band-jitter-db "$BAND_JITTER" \
    --robust-fir-gain-jitter-db "$GAIN_JITTER" \
    --robust-fir-phase-jitter "$PHASE_JITTER" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --warmup-ratio "$WARMUP" \
    --n-eot-samples "$N_EOT" \
    --prompt-mode qa \
    --commands nav_turn_left nav_turn_right \
    --output-dir "$OUTPUT_DIR" \
    "$@"
