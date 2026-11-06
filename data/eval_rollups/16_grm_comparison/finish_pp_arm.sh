#!/bin/bash
# Successor to run_paper_prompt.sh. Two jobs it leaves undone:
#   1. the transfer clean-codec controls (T_RUNGS has no wav_clean_gl__opus*),
#      without which the pp transfer codec cells have no comparator at all;
#   2. Guard-3 over the whole pp arm.
# Waits for the GPU to actually free, not just the marker.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python

for i in $(seq 1 480); do
  grep -q PAPERPROMPTDONE logs/paper_prompt.log 2>/dev/null && break
  sleep 60
done
grep -q PAPERPROMPTDONE logs/paper_prompt.log 2>/dev/null || {
  echo "[finish-pp] timed out waiting for the paper-prompt arm $(date -Is)"; exit 1; }
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 2000 ] && break
  sleep 10
done
echo "[finish-pp] starting $(date -Is), GPU used=${USED}MiB"

for M in qwen25_omni_3b audio_flamingo; do
  case "$M" in
    qwen25_omni_3b) VPY=$ENVROOT/qwen-omni/bin/python ;;
    audio_flamingo) VPY=$ENVROOT/flamingo3/bin/python ;;
  esac
  for RUNG in wav_clean_gl__opus64 wav_clean_gl__opus32; do
    OUT="./results_pp/${M}__$RUNG.jsonl"
    [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $M/$RUNG"; continue; }
    echo "=== [tclean-pp] $M $RUNG $(date -Is) ==="
    "$VPY" -u grm_delivery.py transfer --rung "$RUNG" --model "$M" --out "$OUT" \
        --paper-prompt --max-new-tokens 512 || echo "FAILED $M/$RUNG"
  done
done

echo "=== [score-pp] HarmBench $(date -Is) ==="
"$PY" -u score_all_harmbench.py results_pp
echo "=== [score-pp] Guard-3 $(date -Is) ==="
"$PY" -u score_guard3.py results_pp
echo "FINISHPPDONE $(date -Is)"
