#!/bin/bash
# Bandpass (300-4000Hz) + STRONG Speaker Nonlinearity
# drive=20, mix=1.0 → energy_ratio ~0.16 (matches real physical channel's ~0.13)
#
# Previous attempt: drive=2.0, mix=0.3 was too weak (0.7% change, essentially no-op)
# Real physical channel destroys 87% of perturbation energy → need aggressive distortion

set -e

OUTPUT_DIR="results_empirical_ota_IR/exp_bandpass300_4000_strong_nonlinear_eps3.0_1000steps"
mkdir -p "$OUTPUT_DIR"

echo "=== Bandpass 300-4000Hz + STRONG Speaker Nonlinearity ==="
echo "  drive=20.0, mix=1.0 (energy_ratio ~0.16, matches physical)"
echo "  Output: $OUTPUT_DIR"
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
    --speaker-drive 20.0 \
    --speaker-mix 1.0 \
    --output-dir "$OUTPUT_DIR" \
    --quick \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
