#!/bin/bash
# Re-eval S3a + S3b adding AAC (128/192/256) and MP3 (192/320) channels.
# Relies on runner resume logic (channel_tag, adv_wav) — only new channel cells fire.
# Usage: ./reeval_s3_aac_mp3.sh <model> [eps ...]
#   model: qwen2 | af3 | omni
#   eps:   one or more of 0.5 1.0 (default: both)
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-qwen2}"; shift || true
EPS_LIST=("$@"); [[ ${#EPS_LIST[@]} -eq 0 ]] && EPS_LIST=(0.5 1.0)

case "$MODEL" in
  qwen2) PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python; TAG=qwen2_audio ;;
  af3)   PY=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python;     TAG=audio_flamingo ;;
  omni)  PY=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python;     TAG=qwen25_omni ;;
  *) echo "bad model: $MODEL (qwen2|af3|omni)"; exit 1 ;;
esac

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
ENV_BIN="$(dirname "$PY")"
export PATH="$ENV_BIN:$PATH"   # ffmpeg for aac/mp3
export PYTHONPATH="$PWD/src:$PWD/..:${PYTHONPATH:-}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

for scen in 03a_ai_detection_bypass 03b_copyright_bypass; do
  case "$scen" in
    03a*) CMAP="data/music/03a_ai_detection_bypass/carrier_to_target.json" ;;
    03b*) CMAP="data/music/03b_copyright_bypass/carrier_to_target.json" ;;
  esac
  export AUDIO_AGENT_BENCH_CARRIER_MAP="$CMAP"
  for eps in "${EPS_LIST[@]}"; do
    BUNDLE="results/$scen/$TAG/eps_$eps"
    if [[ ! -d "$BUNDLE/audio" ]]; then
      echo "[$(ts)] skip $scen $TAG eps=$eps — no adv wavs"
      continue
    fi
    LOG="scripts/reeval_s3_aac_mp3_${TAG}_${scen}_eps${eps}.log"
    echo "[$(ts)] START $scen $TAG eps=$eps"
    "$PY" -u -m audio_agent_bench run "scenarios/$scen" \
        --model "$TAG" --out "$BUNDLE" > "$LOG" 2>&1
    rc=$?
    echo "[$(ts)] DONE  $scen $TAG eps=$eps rc=$rc -> $LOG"
  done
done

echo "[$(ts)] refreshing rollups"
"$CA" scripts/rollup_results.py || true
echo "[$(ts)] all done"
