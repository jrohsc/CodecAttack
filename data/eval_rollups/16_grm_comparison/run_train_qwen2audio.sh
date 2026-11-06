#!/bin/bash
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
# Paper setting: K=48 released mask, tau=0.5, AdamW lr 0.01, 100 epochs,
# semantic_loss_weight 5 (README), batch 8, bf16, tile 128x3000.
"$PY" -u main.py \
  --model_path ${MODEL_ROOT}/qwen2/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a \
  --mask_path ./logs/top48_weighted.npz \
  --train_file ./train.jsonl \
  --num_epochs 100 --batch_size 8 --tau 0.5 --lr 0.01 \
  --semantic_loss_weight 5 \
  --save_dir ./logs/run_qwen2audio
echo "TRAINDONE $(date -Is)"
