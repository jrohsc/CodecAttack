#!/usr/bin/env bash
# Wait for the AF3 chain to finish, then retry the calm-tail (which OOMed on
# the first attempt). Single-GPU box → strictly sequential.
#
# Usage: bash retry_calm_after_af3.sh <chain_pid>
set -u
CHAIN_PID="${1:?need chain PID}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ABL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"
CA_PY="${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ABL/scripts/retry_calm.${STAMP}.log"

cd "$REPO"

{
    echo "[$(date '+%F %T')] waiting for AF3 chain PID=$CHAIN_PID"
    while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] AF3 chain finished, sleeping 30s for GPU cleanup"
    sleep 30

    echo "[$(date '+%F %T')] retrying calm tail w040/057/082/131"
    "$CA_PY" -u "$ABL/scripts/run_expA_words.py" \
        --carriers calm --targets w040 w057 w082 w131
    RC=$?
    echo "[$(date '+%F %T')] calm retry exit=$RC"

    echo "[$(date '+%F %T')] rebuilding strict figure + table"
    "$CA_PY" "$ABL/scripts/build_strict_capacity.py"
    echo "[$(date '+%F %T')] retry chain done"
} 2>&1 | tee "$LOG"
