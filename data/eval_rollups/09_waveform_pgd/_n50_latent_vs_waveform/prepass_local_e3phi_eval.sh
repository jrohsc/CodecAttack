#!/bin/bash
# One-shot PRE-PASS: score the wavs that already exist in the 8 e3phi_rep bundles
# while crafting is still running. ensemble_heldout_eval.py caches per-clip
# records and skips them on rerun, so the watcher / SLURM eval later only has
# to score the clips crafted after this pass.
#
# Coordination with watch_local_e3phi_eval.sh: each bundle is evaluated under
# the SAME lock dir the watcher uses ($STATE/lock_<seed>_<shard>); the watcher
# `continue`s when the mkdir fails, so the two never write one bundle at once.
#
# Truncated-wav guard: wavs are sf.write'n in place (no tmp+rename). The eval
# globs at start and never sees later files, so the only race is a wav being
# written at glob time -> wait until the newest wav is >=30s old.
#
# Kimi is deliberately excluded (codec-attack project rule: no Kimi).
set -uo pipefail

REPO=${REPO_ROOT}
ENVROOT=${WORK_ROOT}/miniconda3/envs
STATE="$REPO/.e3phi_local_eval_watch"
LOG="$REPO/logs/e3phi_prepass.log"
CH="opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"

cd "$REPO"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONNOUSERSITE=1
export PATH="$ENVROOT/codec-attack/bin:$PATH"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

py_for() { case "$1" in
  qwen2_audio) echo "$ENVROOT/codec-attack/bin/python" ;;
  qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
  audio_flamingo) echo "$ENVROOT/flamingo3/bin/python" ;;
  phi4) echo "$ENVROOT/audio-llm/bin/python" ;; esac; }

wait_quiet() {                      # newest wav in $1/audio >= 30s old
  while :; do
    local newest now
    newest=$(find "$1/audio" -maxdepth 1 -name '*.wav' -printf '%T@\n' | sort -n | tail -1)
    now=$(date +%s)
    [[ -z "$newest" ]] && return 0
    (( now - ${newest%.*} >= 30 )) && return 0
    sleep 10
  done
}

log "=== prepass start ==="
for SPEC in "qwen25_omni omni7b_MEMBER" "audio_flamingo af3_MEMBER" "phi4 phi4_MEMBER" \
            "qwen2_audio qwen2_HELDOUT" "qwen25_omni_3b omni3b_HELDOUT"; do
  set -- $SPEC; M=$1; OUT=$2
  PY=("$(py_for "$M")")
  [[ "$M" == qwen25_omni || "$M" == qwen25_omni_3b ]] && PY=(env -u PYTHONNOUSERSITE "${PY[0]}")
  for SEED in 7 42; do for SH in 0 1 2 3; do
    B="results/benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed${SEED}_shard${SH}of4"
    [[ -f "$STATE/done_${SEED}_${SH}" ]] && { log "seed$SEED/sh$SH already done by watcher"; continue; }
    if ! mkdir "$STATE/lock_${SEED}_${SH}" 2>/dev/null; then
      log "seed$SEED/sh$SH locked by watcher; skipping $OUT"; continue
    fi
    wait_quiet "$B"
    n=$(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l)
    log "seed$SEED/sh$SH -> $OUT ($n wavs) starting"
    "${PY[@]}" -u 0_all_combined/scripts/ensemble_heldout_eval.py \
      --bundle-dir "$B" --target-model "$M" --out-name "$OUT" --channels "$CH" \
      >> "$REPO/logs/e3phi_prepass_seed${SEED}_sh${SH}_${OUT}.log" 2>&1
    log "seed$SEED/sh$SH -> $OUT rc=$?"
    rmdir "$STATE/lock_${SEED}_${SH}"
  done; done
done
log "=== prepass done ==="
