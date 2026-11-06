#!/bin/bash
# Local (no-sbatch) driver for the masked encoder-objective arms.
#
# Same parameters as sbatch_encoder_transfer_masked.sh -- see that file for why
# the mask exists. Runs on whatever GPU this shell already owns instead of
# queueing.
#
# Two things learned the hard way here:
#
#   - This node is shared with other sessions. A concurrent job held 38GB and
#     both arms OOM'd on model load. Each arm needs ~25GB, so the driver measures
#     free VRAM and only overlaps the arms when there is genuinely room for both.
#   - The first version ran the arms with `&` under `set -uo pipefail` and never
#     checked their exit codes, so it printed "BOTH ARMS DONE" 17 seconds after a
#     double OOM. Exit codes are now checked and a failed arm is fatal and loud.
#
# Per-clip results are cached, so re-running fills only the missing clips, and an
# interruption resumes. A partial bundle still scores as a floor-biased number,
# so count wavs against --n before scoring.
#
# Usage: bash run_encoder_transfer_masked_local.sh [steps] [n_pairs]
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codec-attack
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

STEPS=${1:-2500}
N_PAIRS=${2:-50}
NEED_MB=26000          # measured peak for one arm, with headroom

free_mb() {
  local used total
  read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
                         --format=csv,noheader,nounits | tr ',' ' ')
  echo $((total - used))
}

run_arm() {
  local loss="$1" name="$2"
  echo "=== $(date +%H:%M:%S) masked arm loss=$loss name=$name steps=$STEPS n=$N_PAIRS ==="
  python encoder_transfer_attack.py \
      --loss "$loss" \
      --init target \
      --init-scale 0.1 \
      --eps 1.0 \
      --steps "$STEPS" \
      --check-every 250 \
      --frame-mask target \
      --carrier-mode match \
      --n "$N_PAIRS" \
      --source-model qwen2_audio \
      --prompt-mode transcribe \
      --out-dir "0_all_combined/results_encoder_transfer/${name}"
  local rc=$?
  local got
  got=$(ls "0_all_combined/results_encoder_transfer/${name}"/audio/*_attacked.wav 2>/dev/null | wc -l)
  if [ $rc -ne 0 ]; then
    echo "!!! FAILED $name rc=$rc  wavs=$got/$N_PAIRS  (see this log above for the traceback)"
    return $rc
  fi
  if [ "$got" -ne "$N_PAIRS" ]; then
    echo "!!! INCOMPLETE $name: wavs=$got/$N_PAIRS despite rc=0 -- do not score this bundle"
    return 1
  fi
  echo "=== $(date +%H:%M:%S) done $name; wavs: $got/$N_PAIRS ==="
  return 0
}

FREE=$(free_mb)
echo "[driver] free VRAM: ${FREE} MB; one arm needs ~${NEED_MB} MB"
if [ "$FREE" -lt "$NEED_MB" ]; then
  echo "[driver] ABORT: not enough free VRAM for even one arm."
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
fi

# Hybrid first: it is the arm most likely to produce a working attack, so if the
# card gets taken out from under us the useful half is already done.
FAIL=0
if [ "$FREE" -ge $((NEED_MB * 2)) ]; then
  echo "[driver] room for both -- running concurrently"
  run_arm enc_ce encce_mask_match > logs/local_encce_mask_match.log 2>&1 &
  P1=$!
  run_arm enc    enc_mask_match   > logs/local_enc_mask_match.log   2>&1 &
  P2=$!
  wait $P1 || FAIL=1
  wait $P2 || FAIL=1
else
  echo "[driver] only room for one -- running sequentially (hybrid first)"
  run_arm enc_ce encce_mask_match > logs/local_encce_mask_match.log 2>&1 || FAIL=1
  run_arm enc    enc_mask_match   > logs/local_enc_mask_match.log   2>&1 || FAIL=1
fi

if [ $FAIL -ne 0 ]; then
  echo "=== FINISHED WITH FAILURES $(date +%H:%M:%S) -- check logs/local_*_mask_match.log ==="
  exit 1
fi
echo "=== BOTH ARMS DONE $(date +%H:%M:%S) ==="
