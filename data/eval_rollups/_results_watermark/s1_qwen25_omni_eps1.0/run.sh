#!/usr/bin/env bash
# S1 watermark x codec-channel sweep on Qwen2.5-Omni at eps=1.0.
#
# Run with:
#   bash run.sh
#
# Resume-safe: re-run the same command to fill in any missing
# (carrier, channel) cells without re-doing finished work.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEC_DIR="$(cd "$HERE/../../.." && pwd)"

# AudioSeal 0.2.0 needs omegaconf>=2.0; the codec-attack env is pinned to 1.4.1.
# qwen-omni env has the right omegaconf and Qwen2.5-Omni; use it for both.
PY=${PYTHON_QWEN_OMNI}

if [[ ! -x "$PY" ]]; then
    echo "ERROR: missing python: $PY"
    exit 1
fi

# cd into codec_attack so demo_ota / aac_channel / config / latent_attack imports
# resolve their relative paths the same way other runners do.
cd "$CODEC_DIR"

echo "=========================================================="
echo "  S1 watermark x codec-channel sweep on Qwen2.5-Omni"
echo "=========================================================="
echo "  python   : $PY"
echo "  cwd      : $(pwd)"
echo "  out dir  : $HERE"
echo "----------------------------------------------------------"

exec "$PY" "$HERE/run_channel_sweep.py" \
    --out-dir "$HERE" \
    --max-pairs 50 \
    --steps 300 \
    --device cuda \
    --dtype bfloat16 \
    "$@"
