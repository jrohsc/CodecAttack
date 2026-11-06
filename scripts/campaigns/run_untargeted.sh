#!/bin/bash
# ============================================================================
# Run untargeted latent-space attack benchmark.
#
# Untargeted mode: maximize CE loss on the model's original output to disrupt
# its response. Success = WER(original, adversarial) > 0.5.
#
# Usage:
#   ./run_untargeted.sh                           # Full: 200 commands, eps=10.0
#   ./run_untargeted.sh --quick                   # Quick: 2 commands, 150 steps
#   ./run_untargeted.sh --eps 5.0                 # Custom epsilon
#   ./run_untargeted.sh --music jazz_1            # Different carrier
#   ./run_untargeted.sh --category navigation     # One category only
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
CONDA_ENV="codec-attack"
EPS="10.0"
STEPS="300"
MUSIC="empire_state_of_mind"
EXTRA_ARGS=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            EXTRA_ARGS="$EXTRA_ARGS --quick"
            STEPS="150"
            shift;;
        --eps|-e)
            EPS="$2"; shift 2;;
        --steps|-s)
            STEPS="$2"; shift 2;;
        --music|-m)
            MUSIC="$2"; shift 2;;
        --category|-c)
            EXTRA_ARGS="$EXTRA_ARGS --category $2"; shift 2;;
        --quiet)
            EXTRA_ARGS="$EXTRA_ARGS --quiet"; shift;;
        --help|-h)
            echo "Usage: ./run_untargeted.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick, -q           Quick test: 2 commands, 150 steps"
            echo "  --eps, -e EPS         Latent epsilon (default: 10.0)"
            echo "  --steps, -s STEPS     Optimization steps (default: 300)"
            echo "  --music, -m MUSIC     Music carrier (default: empire_state_of_mind)"
            echo "  --category, -c CAT    Filter to one category (navigation, manipulation, ...)"
            echo "  --quiet               Reduce per-step logging"
            echo ""
            echo "Available music carriers:"
            echo "  empire_state_of_mind, jazz_1, jazz_2, calm_1, calm_2,"
            echo "  christmas_jazz_1, christmas_jazz_2, classical_music_1, classical_music_2"
            exit 0
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

echo "============================================================================"
echo "  UNTARGETED LATENT-SPACE ATTACK"
echo "  Music: $MUSIC | eps: $EPS | steps: $STEPS"
echo "============================================================================"
echo ""

conda run -n "$CONDA_ENV" python run_benchmark.py \
    --untargeted \
    --music "$MUSIC" \
    --eps "$EPS" \
    --steps "$STEPS" \
    --prompt-mode transcribe \
    $EXTRA_ARGS
