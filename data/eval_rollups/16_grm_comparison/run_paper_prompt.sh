#!/bin/bash
# Paper-protocol arm: identical rungs, identical rendered audio, but evaluated
# under generate.py's prompt (the one Table 1 is measured with) at its
# max_new_tokens=512. Results land in results_pp/ so the matched-prompt ladder
# in results/ is untouched.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python
TILE=./logs/run_qwen2audio/best.pt
R=./results_pp
mkdir -p "$R"

MEL_RUNGS="mel_clean mel_grm"
WAV_RUNGS="wav_raw wav_clean_gl wav_grm_gl \
           wav_grm_gl__opus64 wav_grm_gl__opus32 wav_grm_gl__mp3_64 wav_grm_gl__aac_64 \
           wav_clean_gl__opus64 wav_clean_gl__opus32 wav_clean_gl__mp3_64 wav_clean_gl__aac_64"
for RUNG in $MEL_RUNGS $WAV_RUNGS; do
  OUT="$R/qwen2_audio__$RUNG.jsonl"
  [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $RUNG (scored)"; continue; }
  echo "=== [gen-pp] qwen2_audio $RUNG $(date -Is) ==="
  "$PY" -u grm_delivery.py generate --rung "$RUNG" --model qwen2_audio \
      --tile "$TILE" --out "$OUT" --batch-size 8 \
      --paper-prompt --max-new-tokens 512 || echo "FAILED gen $RUNG"
done

T_RUNGS="wav_raw wav_clean_gl wav_grm_gl wav_grm_gl__opus64 wav_grm_gl__opus32"
for M in qwen25_omni_3b audio_flamingo; do
  case "$M" in
    qwen25_omni_3b) VPY=$ENVROOT/qwen-omni/bin/python ;;
    audio_flamingo) VPY=$ENVROOT/flamingo3/bin/python ;;
  esac
  for RUNG in $T_RUNGS; do
    OUT="$R/${M}__$RUNG.jsonl"
    [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $M/$RUNG (scored)"; continue; }
    echo "=== [transfer-pp] $M $RUNG $(date -Is) ==="
    "$VPY" -u grm_delivery.py transfer --rung "$RUNG" --model "$M" --out "$OUT" \
        --paper-prompt --max-new-tokens 512 || echo "FAILED transfer $M/$RUNG"
  done
done

echo "=== [score-pp] HarmBench over $R $(date -Is) ==="
"$PY" -u score_all_harmbench.py "$R"
echo "PAPERPROMPTDONE $(date -Is)"
