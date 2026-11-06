#!/usr/bin/env bash
# Wait for the current in-flight AF3 eps=1.0 attack (PID $WAIT_PID) to exit,
# then launch the chain script. Must be started BEFORE the attack exits.
set -u

cd ${REPO_ROOT}

WAIT_PID="${1:-}"
if [[ -z "$WAIT_PID" ]]; then
    echo "usage: $0 <pid-to-wait-for>" >&2
    exit 2
fi

echo "[wait_then_chain] waiting for PID $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
done
echo "[wait_then_chain] PID $WAIT_PID exited. launching chain."

exec ${REPO_ROOT}/0_all_combined/scripts/chain_s1_s2_codec_robust.sh
