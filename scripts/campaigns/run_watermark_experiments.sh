#!/bin/bash
# Run AudioSeal watermark defense experiments.
#
# Usage:
#   bash run_watermark_experiments.sh codec_roundtrip
#   bash run_watermark_experiments.sh attack_detection --music jazz_1 --num_commands 5
#   bash run_watermark_experiments.sh all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate codec-attack

cd "$SCRIPT_DIR"

EXPERIMENT="${1:-codec_roundtrip}"
shift || true

python run_watermark_experiments.py --experiment "$EXPERIMENT" "$@"
