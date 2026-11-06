#!/usr/bin/env bash
# Run the eps sweep sequentially: jazz at eps=0.5 then eps=1.5, 6 word counts each.
# 12 cells total, ~2.6h on a single A100.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ABL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"
PY="${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ABL/scripts/chain_eps_sweep.${STAMP}.log"

WORDS=(w002 w005 w008 w015 w020 w032)

cd "$REPO"

{
    echo "[$(date '+%F %T')] eps sweep starting (jazz, w<=32)"

    for EPS in 0.5 1.5; do
        echo
        echo "=== eps=$EPS ==="
        "$PY" -u "$ABL/scripts/run_expA_eps_sweep.py" \
            --eps "$EPS" --carriers jazz --targets "${WORDS[@]}"
        RC=$?
        echo "[$(date '+%F %T')] eps=$EPS exit=$RC"
    done

    echo
    echo "[$(date '+%F %T')] eps sweep complete"
} 2>&1 | tee "$LOG"
