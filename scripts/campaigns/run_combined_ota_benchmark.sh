#!/bin/bash
# Combined OTA robustness experiment:
# yakura_ota (RIR + noise) + time_shift + freq_augment + gain attenuation
#
# This combines ALL channel augmentations we have:
# 1. Diverse RIR convolution (room acoustics) - from yakura_ota mode
# 2. Gaussian noise (SNR 20±5 dB) - from yakura_ota mode
# 3. Bandpass perturbation (1-4 kHz) - Yakura IJCAI'19
# 4. Time shift (0-100ms) - encoder padding robustness
# 5. Freq response augmentation (empirical speaker/mic FIR) - spectral reshaping
# 6. Random gain (-20 to -10 dB) - OTA volume loss
#
# 30 SayCan commands, eps=5.0, 1000 steps, QA mode

set -e

CONDA_ENV="codec-attack"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results_combined_ota"

echo "=== Combined OTA Robustness Benchmark ==="
echo "Output: ${OUTPUT_DIR}"
echo ""

conda run -n ${CONDA_ENV} python "${SCRIPT_DIR}/run_robust_benchmark.py" \
    --target-model qwen2_audio \
    --dataset saycan \
    --music calm_1 \
    --eps 5.0 \
    --steps 1000 \
    --prompt-mode qa \
    --channel-mode yakura_ota \
    --rir-dir "${SCRIPT_DIR}/data/rirs" \
    --bandpass-low-hz 1000 \
    --bandpass-high-hz 4000 \
    --noise-snr-db 20.0 \
    --noise-snr-std 5.0 \
    --n-eot-samples 4 \
    --time-shift-ms 100 \
    --freq-augment \
    --freq-augment-jitter 0.3 \
    --gain-range-db="-20,-10" \
    --warmup-ratio 0.3 \
    --output-dir "${OUTPUT_DIR}" \
    --commands saycan_000 saycan_001 saycan_002 saycan_003 saycan_004 \
              saycan_005 saycan_006 saycan_007 saycan_008 saycan_009 \
              saycan_010 saycan_011 saycan_012 saycan_013 saycan_014 \
              saycan_015 saycan_016 saycan_017 saycan_018 saycan_019 \
              saycan_020 saycan_021 saycan_022 saycan_023 saycan_024 \
              saycan_025 saycan_026 saycan_027 saycan_028 saycan_029 \
              saycan_030 saycan_031 saycan_032 saycan_033 saycan_034 \
              saycan_035 saycan_036 saycan_037 saycan_038 saycan_039 \
    "$@"

echo ""
echo "=== Done ==="
echo "Results in: ${OUTPUT_DIR}"
