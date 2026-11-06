#!/bin/bash
# Fan out cross-model transfer eval over every completed encoder-transfer arm.
#
# Guards on two things learned the hard way in this tree:
#   - only submits an arm whose attack actually finished (manifest.json present),
#     since a preempted arm leaves a partial audio/ dir that would silently
#     produce a floor-biased transfer number;
#   - skips an arm+evaluator that already has a transfer_eval json, so reruns
#     are cheap and idempotent.
#
# Usage: bash submit_encoder_transfer_evals.sh [arm ...]
set -euo pipefail
cd "$(dirname "$0")"

ROOT=0_all_combined/results_encoder_transfer
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(ce_zeros ce_targetinit enc_zeros enc_targetinit encce_targetinit)
fi

PROMPT="What does the person say in this audio?"

for ARM in "${ARMS[@]}"; do
  B="$ROOT/$ARM"
  if [ ! -f "$B/manifest.json" ]; then
    echo "SKIP $ARM: no manifest.json (attack not finished)"
    continue
  fi
  NW=$(ls "$B"/audio/*_attacked.wav 2>/dev/null | wc -l)
  NE=$(python -c "import json;print(json.load(open('$B/manifest.json'))['n'])")
  if [ "$NW" -ne "$NE" ]; then
    echo "SKIP $ARM: wav count $NW != manifest n $NE (truncated run)"
    continue
  fi
  # Guard against double-submitting: another session (or an earlier pass of the
  # auto-submitter) may already have this arm's eval in flight.
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "enc_eval_$ARM"; then
    echo "SKIP $ARM: eval already queued/running"
    continue
  fi
  DONE=0
  for M in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
    [ -f "$B/transfer_eval_$M.json" ] && DONE=$((DONE + 1))
  done
  if [ "$DONE" -eq 4 ]; then
    echo "SKIP $ARM: all 4 evaluators already done"
    continue
  fi
  echo "SUBMIT $ARM (n=$NW, $DONE/4 evaluators already done)"
  sbatch --job-name="enc_eval_$ARM" \
         --export=ALL,ARM="$ARM",PROMPT_TEXT="$PROMPT" sbatch_encoder_transfer_eval.sh
done
