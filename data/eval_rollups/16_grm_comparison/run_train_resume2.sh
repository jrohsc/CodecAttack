#!/bin/bash
# Resume GRM tile training on ece-gpu001 with OOM fallback + auto-restart.
# Total target: 100 epochs. Counts every completed-epoch line in the shared log,
# so it is safe to run repeatedly.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
LOG=logs/train_qwen2audio.log
SAVE=./logs/run_qwen2audio
MODEL=${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a
TARGET=100

# (batch_size, update_every) ladder -- effective batch stays 8
LADDER=("8 1" "4 2" "2 4")
STEP=0

for ATTEMPT in 1 2 3 4 5 6; do
  DONE=$(grep -c '^\[E' "$LOG")
  REMAIN=$(( TARGET - DONE ))
  if [ "$REMAIN" -le 0 ]; then
    echo "[resume2] $DONE/$TARGET epochs done" ; echo "TRAINDONE $(date -Is)" >> "$LOG"; exit 0
  fi
  set -- ${LADDER[$STEP]}; BS=$1; UE=$2
  echo "[resume2] attempt $ATTEMPT: $DONE done, $REMAIN to go, batch=$BS update_every=$UE $(date -Is)"
  cp -f "$SAVE/best.pt" "$SAVE/best_e${DONE}_preresume.pt" 2>/dev/null

  {
    echo "=== RESUME2 attempt $ATTEMPT on $(hostname): from epoch $DONE, $REMAIN to go, bs=$BS ue=$UE $(date -Is) ==="
    "$PY" -u main.py \
      --model_path "$MODEL" \
      --mask_path ./logs/top48_weighted.npz \
      --train_file ./train.jsonl \
      --resume_tile "$SAVE/best.pt" \
      --num_epochs "$REMAIN" --batch_size "$BS" --update_every "$UE" \
      --tau 0.5 --lr 0.01 --semantic_loss_weight 5 \
      --save_dir "$SAVE"
    echo "[resume2] main.py exit=$? $(date -Is)"
  } >> "$LOG" 2>&1

  NEWDONE=$(grep -c '^\[E' "$LOG")
  if [ "$NEWDONE" -ge "$TARGET" ]; then
    echo "[resume2] reached $NEWDONE/$TARGET $(date -Is)"; echo "TRAINDONE $(date -Is)" >> "$LOG"; exit 0
  fi
  if [ "$NEWDONE" -le "$DONE" ]; then
    # made zero progress -> almost certainly OOM, step down the ladder
    STEP=$(( STEP + 1 ))
    if [ "$STEP" -ge "${#LADDER[@]}" ]; then
      echo "[resume2] out of batch-size fallbacks, giving up $(date -Is)"; exit 1
    fi
    echo "[resume2] no progress, stepping down to batch ${LADDER[$STEP]}"
  else
    echo "[resume2] progressed to $NEWDONE epochs, restarting $(date -Is)"
  fi
  sleep 10
done
echo "[resume2] attempts exhausted $(date -Is)"; exit 1
