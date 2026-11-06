#!/usr/bin/env bash
# Wait for the calm-tail PID to finish, then launch AF3 replication.
# Single-GPU box → runs strictly sequentially.
#
# Usage: bash chain_calm_then_af3.sh <calm_pid>
set -u
CALM_PID="${1:?need calm_tail PID}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ABL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"
AF3_PY="${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="$ABL/scripts/chain_af3.${STAMP}.log"

cd "$REPO"

{
    echo "[$(date '+%F %T')] waiting for calm-tail PID=$CALM_PID"
    while kill -0 "$CALM_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] calm-tail finished, launching AF3 replication"

    # Verify calm tail produced rollups before starting AF3
    for w in w040 w057 w082 w131; do
        f="$ABL/expA_words/qwen2_audio_eps1.0_multibitrate/calm/$w/rollup.json"
        if [[ -f "$f" ]]; then
            echo "  calm/$w: OK"
        else
            echo "  calm/$w: MISSING (continuing anyway)"
        fi
    done

    "$AF3_PY" -u "$ABL/scripts/run_expA_af3.py"
    AF3_RC=$?
    echo "[$(date '+%F %T')] AF3 run exited rc=$AF3_RC"

    # Rebuild figure + table once both phases land
    if [[ -x ${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python ]]; then
        echo "[$(date '+%F %T')] rebuilding strict figure + table"
        ${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python \
            "$ABL/scripts/build_strict_capacity.py"
    fi
    echo "[$(date '+%F %T')] chain done"
} 2>&1 | tee "$CHAIN_LOG"
