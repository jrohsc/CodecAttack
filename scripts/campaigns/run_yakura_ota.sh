#!/bin/bash
# Yakura-style OTA attack: bandpass perturbation + diverse RIRs + Gaussian noise
#
# Step 1: Generate diverse synthetic RIRs (if not already done)
# Step 2: Run robust attack with yakura_ota channel mode
#
# Usage:
#   bash run_yakura_ota.sh                   # Quick test (1 music, 2 commands, 150 steps)
#   bash run_yakura_ota.sh --full            # Full benchmark
#   bash run_yakura_ota.sh --full --eps 15   # Full benchmark with higher eps

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RIR_DIR="${SCRIPT_DIR}/data/rirs"
N_RIRS=615

# ============================================================================
# Parse args
# ============================================================================
QUICK_MODE=true
EXTRA_ARGS=""
EPS=10.0
STEPS=300
MUSIC="jazz_1"
N_EOT=4
BPF_LOW=1000
BPF_HIGH=4000
NOISE_SNR=20.0
NOISE_STD=5.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            QUICK_MODE=false
            shift
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --music)
            MUSIC="$2"
            shift 2
            ;;
        --n-eot)
            N_EOT="$2"
            shift 2
            ;;
        --bpf-low)
            BPF_LOW="$2"
            shift 2
            ;;
        --bpf-high)
            BPF_HIGH="$2"
            shift 2
            ;;
        --noise-snr)
            NOISE_SNR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# ============================================================================
# Step 1: Generate RIRs if needed
# ============================================================================
if [ ! -d "${RIR_DIR}" ] || [ "$(ls -1 ${RIR_DIR}/*.wav 2>/dev/null | wc -l)" -lt "${N_RIRS}" ]; then
    echo "============================================"
    echo "Generating ${N_RIRS} diverse synthetic RIRs..."
    echo "============================================"
    python "${SCRIPT_DIR}/generate_rirs.py" \
        --output-dir "${RIR_DIR}" \
        --n-rirs "${N_RIRS}" \
        --sample-rate 24000 \
        --max-length-ms 500 \
        --seed 42
    echo ""
fi

echo "============================================"
echo "RIR directory: ${RIR_DIR} ($(ls -1 ${RIR_DIR}/*.wav 2>/dev/null | wc -l) files)"
echo "============================================"

# ============================================================================
# Step 2: Run Yakura OTA attack
# ============================================================================
echo ""
echo "============================================"
echo "Running Yakura-style OTA attack"
echo "  Band-pass: ${BPF_LOW}-${BPF_HIGH} Hz"
echo "  RIRs: ${RIR_DIR}"
echo "  Noise SNR: ${NOISE_SNR} ± ${NOISE_STD} dB"
echo "  EoT samples: ${N_EOT}"
echo "  Eps: ${EPS}"
echo "  Steps: ${STEPS}"
echo "  Music: ${MUSIC}"
echo "============================================"
echo ""

CMD="python ${SCRIPT_DIR}/run_robust_benchmark.py \
    --channel-mode yakura_ota \
    --bandpass-low-hz ${BPF_LOW} \
    --bandpass-high-hz ${BPF_HIGH} \
    --rir-dir ${RIR_DIR} \
    --noise-snr-db ${NOISE_SNR} \
    --noise-snr-std ${NOISE_STD} \
    --n-eot-samples ${N_EOT} \
    --eps ${EPS} \
    --steps ${STEPS} \
    --music ${MUSIC} \
    --warmup-ratio 0.3 \
    --channel-severity 1.0 \
    --channel-curriculum"

if [ "${QUICK_MODE}" = true ]; then
    CMD="${CMD} --quick"
fi

CMD="${CMD} ${EXTRA_ARGS}"

echo "Command: ${CMD}"
echo ""
eval ${CMD}
