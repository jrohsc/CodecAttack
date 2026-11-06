#!/bin/bash
# Experiment 4: Robust FIR
#
# PSD-ratio FIR (from exp3) + stochastic augmentation to cover
# the ~40% channel variance that static FIR can't capture:
#   - Per-band random gain jitter (±3dB across 8 octave bands)
#   - Global gain jitter (±3dB, simulates distance/volume variation)
#   - Circular time shift (0-100ms, simulates encoder padding)
#   - SpecAugment on mel features (frequency masking for unknown channel effects)
#   - Additive uniform noise (eps=0.02)
#   - Real noise residuals (from empirical recordings)
#   - EoT x4 averaging
#
# Hypothesis: static FIR works (87.5% sim OTA) but physical fails because
# the channel is non-stationary. Adding stochastic augmentation on top of
# the FIR should produce perturbations robust to channel variation.

set -e

EPS=${EPS:-3.0}
STEPS=${STEPS:-1000}
MUSIC=${MUSIC:-calm_1}
WARMUP=${WARMUP:-0.3}
N_EOT=${N_EOT:-4}
BAND_JITTER=${BAND_JITTER:-3.0}
GAIN_JITTER=${GAIN_JITTER:-3.0}

OUTPUT_DIR="results_empirical_ota_IR/4_exp_robust_fir"

echo "=== Experiment 4: Robust FIR ==="
echo "eps=$EPS, steps=$STEPS, music=$MUSIC, EoT=$N_EOT"
echo "Channel: PSD FIR + band_jitter=±${BAND_JITTER}dB + gain_jitter=±${GAIN_JITTER}dB"
echo "         + SpecAugment + time_shift + uniform_noise + real_noise"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

conda run -n codec-attack --no-capture-output \
    python run_robust_benchmark.py \
    --channel-mode empirical_ota \
    --robust-fir \
    --robust-fir-band-jitter-db "$BAND_JITTER" \
    --robust-fir-gain-jitter-db "$GAIN_JITTER" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --music "$MUSIC" \
    --warmup-ratio "$WARMUP" \
    --n-eot-samples "$N_EOT" \
    --prompt-mode qa \
    --commands nav_turn_left nav_turn_right nav_go_straight nav_u_turn \
               nav_charging nav_corridor nav_parking nav_exit_building \
               nav_elevator nav_stairs \
    --output-dir "$OUTPUT_DIR" \
    "$@"
