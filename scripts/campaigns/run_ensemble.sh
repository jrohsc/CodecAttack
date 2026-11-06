#!/bin/bash
# Ensemble adversarial attack launcher.
#
# Runs the benchmark with ensemble mode enabled, optimizing a single
# perturbation against multiple Audio LLMs simultaneously.
#
# Usage:
#   ./run_ensemble.sh                                    # Full run (default models)
#   ./run_ensemble.sh --quick                            # Quick test
#   ./run_ensemble.sh --ensemble-models qwen2_audio,qwen25_omni  # Subset
#   ./run_ensemble.sh --music jazz_1 --category navigation       # Scoped

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate primary conda env (compatible with Qwen2-Audio + Qwen2.5-Omni)
CONDA_BASE="${CONDA_EXE%/bin/conda}"
if [ -z "$CONDA_BASE" ]; then
    CONDA_BASE="$HOME/miniconda3"
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate codec-attack

echo "=== Ensemble Adversarial Attack ==="
echo "Primary env: codec-attack ($(python --version))"
echo "Working dir: $SCRIPT_DIR"
echo ""

python "$SCRIPT_DIR/run_benchmark.py" \
    --ensemble \
    --prompt-mode transcribe \
    "$@"
