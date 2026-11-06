#!/bin/bash
# Resume GRM tile training on THIS node after the umd-cscdr-gpu001 vscode job
# (64146462, EndTime 17:56:33) is killed at its walltime.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
LOG=logs/train_qwen2audio.log
SAVE=./logs/run_qwen2audio
OLDJOB=64146462

echo "[resume] waiting for old job $OLDJOB to end ... $(date -Is)"
while squeue -h -j "$OLDJOB" -o %T 2>/dev/null | grep -q RUNNING; do sleep 20; done
echo "[resume] job $OLDJOB gone $(date -Is)"

# belt-and-braces: make sure nothing is still appending to the log
while :; do
  A=$(grep -c '^\[E' "$LOG"); sleep 90; B=$(grep -c '^\[E' "$LOG")
  [ "$A" = "$B" ] && break
  echo "[resume] log still growing ($A -> $B), waiting"
done

DONE=$(grep -c '^\[E' "$LOG")
REMAIN=$(( 100 - DONE ))
echo "[resume] $DONE epochs already done, running $REMAIN more $(date -Is)"
if [ "$REMAIN" -le 0 ]; then echo "TRAINDONE $(date -Is)" >> "$LOG"; exit 0; fi

cp -f "$SAVE/best.pt" "$SAVE/best_e${DONE}_preresume.pt"
echo "[resume] backed up tile at epoch $DONE -> best_e${DONE}_preresume.pt"

{
  echo "=== RESUMED on $(hostname) from $SAVE/best.pt at epoch $DONE, $REMAIN epochs to go $(date -Is) ==="
  "$PY" -u main.py \
    --model_path ${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a \
    --mask_path ./logs/top48_weighted.npz \
    --train_file ./train.jsonl \
    --resume_tile "$SAVE/best.pt" \
    --num_epochs "$REMAIN" --batch_size 8 --tau 0.5 --lr 0.01 \
    --semantic_loss_weight 5 \
    --save_dir "$SAVE"
  RC=$?
  echo "[resume] main.py exit=$RC $(date -Is)"
  [ "$RC" = "0" ] && echo "TRAINDONE $(date -Is)"
} >> "$LOG" 2>&1
