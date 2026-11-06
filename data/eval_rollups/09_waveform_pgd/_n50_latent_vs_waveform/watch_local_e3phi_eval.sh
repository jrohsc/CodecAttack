#!/bin/bash
# Local eval watcher for the alternate 3-model ensemble seed replicates.
#
# Why this exists: the 8 crafting shards saturate the account's 8-GPU cap, so an
# eval task whose shard finishes early still pends behind the cap. The local
# 40-GB card is a 9th GPU outside that cap, so it can evaluate the first shards
# to land while the stragglers are still crafting.
#
# It never scancels an eval task -- e3phi1agg depends on afterok:<evalarray>_*,
# so a cancelled element would permanently wedge the chain. Instead it holds the
# element, evaluates locally, then releases it; the released task finds every
# per-clip record cached and exits in seconds, satisfying afterok.
set -uo pipefail

REPO=${REPO_ROOT}
ENVROOT=${WORK_ROOT}/miniconda3/envs
STATE="$REPO/.e3phi_local_eval_watch"
LOG="$REPO/logs/e3phi_local_eval_watch.log"
CH="opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"
POLL=120

# seed:craft_array:eval_array
TRACKS=("7:64881331:64881332" "42:64881334:64881335")

mkdir -p "$STATE"
cd "$REPO"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONNOUSERSITE=1
export PATH="$ENVROOT/codec-attack/bin:$PATH"
export KIMI_INFER_ROOT=${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio
export PYTHONPATH="$KIMI_INFER_ROOT:${PYTHONPATH:-}"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

HELD=()
release_all() {
  for j in "${HELD[@]:-}"; do
    [[ -n "$j" ]] || continue
    scontrol release "$j" 2>/dev/null && log "released $j (cleanup)"
  done
  HELD=()
}
trap 'log "watcher exiting; releasing holds"; release_all; exit 0' INT TERM EXIT

py_for() { case "$1" in
  qwen2_audio) echo "$ENVROOT/codec-attack/bin/python" ;;
  qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
  audio_flamingo) echo "$ENVROOT/flamingo3/bin/python" ;;
  kimi_audio) echo "$ENVROOT/kimi-audio/bin/python" ;;
  phi4) echo "$ENVROOT/audio-llm/bin/python" ;; esac; }

eval_shard() {                      # $1=seed $2=shard $3=bundle
  local seed=$1 sh=$2 B=$3 rc=0
  for SPEC in "qwen25_omni omni7b_MEMBER" "audio_flamingo af3_MEMBER" \
              "phi4 phi4_MEMBER" "qwen2_audio qwen2_HELDOUT" \
              "kimi_audio kimi_HELDOUT_100tok" "qwen25_omni_3b omni3b_HELDOUT"; do
    set -- $SPEC; local M=$1 OUT=$2
    if [[ -f "$B/$OUT/summary.json" ]]; then
      local n
      n=$(python -c "import json;print(len(json.load(open('$B/$OUT/summary.json'))['results']))" 2>/dev/null || echo 0)
      if [[ "$n" == 56 ]]; then log "  seed$seed/sh$sh $OUT already complete (56)"; continue; fi
    fi
    local PY=("$(py_for "$M")")
    [[ "$M" == qwen25_omni || "$M" == qwen25_omni_3b ]] && PY=(env -u PYTHONNOUSERSITE "${PY[0]}")
    local EXTRA=(); [[ "$M" == kimi_audio ]] && EXTRA=(--max-new-tokens 100)
    log "  seed$seed/sh$sh -> $OUT starting"
    "${PY[@]}" -u 0_all_combined/scripts/ensemble_heldout_eval.py \
      --bundle-dir "$B" --target-model "$M" --out-name "$OUT" \
      --channels "$CH" "${EXTRA[@]}" \
      >> "$REPO/logs/e3phi_localeval_seed${seed}_sh${sh}_${OUT}.log" 2>&1
    if [[ $? -ne 0 ]]; then log "  !! seed$seed/sh$sh $OUT FAILED"; rc=1; break; fi
    log "  seed$seed/sh$sh -> $OUT done"
  done
  return $rc
}

log "=== watcher start (poll ${POLL}s, one shard at a time) ==="
while :; do
  remaining=0
  for T in "${TRACKS[@]}"; do
    IFS=: read -r SEED CRAFT EVAL <<< "$T"
    for SH in 0 1 2 3; do
      B="results/benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed${SEED}_shard${SH}of4"
      [[ -f "$STATE/done_${SEED}_${SH}" ]] && continue
      remaining=$((remaining+1))

      n=$(find "$B/audio" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
      cstate=$(squeue -j "${CRAFT}_${SH}" -h -o "%T" 2>/dev/null)

      # truncation audit: craft gone from the queue but bundle short
      if [[ -z "$cstate" && "$n" != 56 ]]; then
        if [[ ! -f "$STATE/warned_${SEED}_${SH}" ]]; then
          log "!! TRUNCATION: seed$SEED/sh$SH craft left queue with $n/56 wavs -- needs resubmit"
          touch "$STATE/warned_${SEED}_${SH}"
        fi
        continue
      fi
      [[ "$n" == 56 ]] || continue

      estate=$(squeue -j "${EVAL}_${SH}" -h -o "%T" 2>/dev/null)
      if [[ "$estate" != "PENDING" ]]; then
        # SLURM already owns it (RUNNING) or it is gone (done)
        [[ -z "$estate" ]] && { log "seed$SEED/sh$SH eval no longer queued; marking done"; touch "$STATE/done_${SEED}_${SH}"; }
        continue
      fi

      mkdir "$STATE/lock_${SEED}_${SH}" 2>/dev/null || continue
      log "claiming seed$SEED/sh$SH ($n/56 wavs, eval ${EVAL}_${SH} PENDING)"
      if scontrol hold "${EVAL}_${SH}" 2>/dev/null; then
        HELD+=("${EVAL}_${SH}"); log "  held ${EVAL}_${SH}"
      else
        log "  could not hold ${EVAL}_${SH}; skipping to avoid a concurrent write"
        rmdir "$STATE/lock_${SEED}_${SH}"; continue
      fi

      if eval_shard "$SEED" "$SH" "$B"; then
        touch "$STATE/done_${SEED}_${SH}"
        log "seed$SEED/sh$SH LOCAL EVAL COMPLETE"
      else
        log "seed$SEED/sh$SH local eval failed; leaving it to SLURM"
      fi
      scontrol release "${EVAL}_${SH}" 2>/dev/null && log "  released ${EVAL}_${SH}"
      HELD=($(printf '%s\n' "${HELD[@]:-}" | grep -vx "${EVAL}_${SH}" || true))
      rmdir "$STATE/lock_${SEED}_${SH}" 2>/dev/null
      break 2          # re-poll from the top; only one shard at a time on one GPU
    done
  done
  (( remaining == 0 )) && { log "=== all 8 shards accounted for; watcher done ==="; break; }
  sleep "$POLL"
done
