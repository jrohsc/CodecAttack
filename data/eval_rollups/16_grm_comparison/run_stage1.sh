#!/bin/bash
# Stage 1: GRM delivery ladder on Qwen2-Audio (white-box) + 2 held-out victims.
# Waits for training to finish, then renders, generates, and scores every cell.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python
TILE=./logs/run_qwen2audio/best.pt
R=./results
mkdir -p "$R" logs

until grep -q TRAINDONE logs/train_qwen2audio.log; do sleep 60; done
echo "=== training done, tile=$TILE $(date -Is) ==="

# ---- 1. render every waveform rung once ----
if [ ! -f delivery/.rendered ]; then
  "$PY" -u grm_delivery.py render --tile "$TILE" && touch delivery/.rendered
fi

# ---- 2. white-box generation on Qwen2-Audio ----
MEL_RUNGS="mel_clean mel_grm"
WAV_RUNGS="wav_raw wav_clean_gl wav_grm_gl \
           wav_grm_gl__opus64 wav_grm_gl__opus32 wav_grm_gl__mp3_64 wav_grm_gl__aac_64 \
           wav_clean_gl__opus64 wav_clean_gl__opus32 wav_clean_gl__mp3_64 wav_clean_gl__aac_64"
for RUNG in $MEL_RUNGS $WAV_RUNGS; do
  OUT="$R/qwen2_audio__$RUNG.jsonl"
  [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $RUNG (scored)"; continue; }
  echo "=== [gen] qwen2_audio $RUNG $(date -Is) ==="
  "$PY" -u grm_delivery.py generate --rung "$RUNG" --model qwen2_audio \
      --tile "$TILE" --out "$OUT" --batch-size 8
done

# ---- 3. transfer generation on held-out victims (waveform rungs only) ----
T_RUNGS="wav_raw wav_clean_gl wav_grm_gl wav_grm_gl__opus64 wav_grm_gl__opus32"
for M in qwen25_omni_3b audio_flamingo; do
  case "$M" in
    qwen25_omni_3b) VPY=$ENVROOT/qwen-omni/bin/python ;;
    audio_flamingo) VPY=$ENVROOT/flamingo3/bin/python ;;
  esac
  for RUNG in $T_RUNGS; do
    OUT="$R/${M}__$RUNG.jsonl"
    [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $M/$RUNG (scored)"; continue; }
    echo "=== [transfer] $M $RUNG $(date -Is) ==="
    "$VPY" -u grm_delivery.py transfer --rung "$RUNG" --model "$M" --out "$OUT"
  done
done

# ---- 4. score everything with HarmBench ----
for F in "$R"/*.jsonl; do
  case "$F" in *_scored.jsonl) continue ;; esac
  [ -f "${F%.jsonl}_scored.jsonl" ] && continue
  echo "=== [score] $F $(date -Is) ==="
  "$PY" -u grm_delivery.py score --responses "$F" --batch-size 8
done

echo "STAGE1DONE $(date -Is)"
