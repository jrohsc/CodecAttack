#!/bin/bash
# Bandpass (300-4000Hz) + Speaker Nonlinearity attack
# Combines both fixes for physical OTA:
#   1. Bandpass constrains perturbation to speaker-reproducible frequencies
#   2. Speaker soft clipping teaches PGD to survive nonlinear distortion
#
# Output: results_empirical_ota_IR/exp_bandpass300_4000_nonlinear_eps3.0_1000steps/

set -e

OUTPUT_DIR="results_empirical_ota_IR/exp_bandpass300_4000_nonlinear_eps3.0_1000steps"

echo "=== Bandpass 300-4000Hz + Speaker Nonlinearity ==="
echo "Output: $OUTPUT_DIR"
echo ""

conda run -n codec-attack python run_robust_benchmark.py \
    --channel-mode empirical_ota \
    --music calm_1 \
    --dataset ours \
    --eps 3.0 \
    --alpha 0.2 \
    --steps 1000 \
    --n-eot-samples 4 \
    --prompt-mode qa \
    --bandpass-low-hz 300 \
    --bandpass-high-hz 4000 \
    --speaker-nonlinearity \
    --speaker-drive 2.0 \
    --speaker-mix 0.3 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
