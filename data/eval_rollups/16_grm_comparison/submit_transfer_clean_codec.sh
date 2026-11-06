#!/bin/bash
#SBATCH --job-name=grm-tclean
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s|h100|a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm_tclean_%j.out
#SBATCH --error=logs/slurm_tclean_%j.out
# run_stage1.sh ran the held-out victims on wav_grm_gl__opus{64,32} but NOT on
# the matching wav_clean_gl__opus{64,32}. Without those, AF3's 16.35 -> 9.62 at
# opus64 has no comparator and cannot be attributed to the attack at all.
set -uo pipefail
cd ${PROJECT_ROOT}/grm_repro
export HF_HUB_OFFLINE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
for M in qwen25_omni_3b audio_flamingo; do
  case "$M" in
    qwen25_omni_3b) VPY=$ENVROOT/qwen-omni/bin/python ;;
    audio_flamingo) VPY=$ENVROOT/flamingo3/bin/python ;;
  esac
  for RUNG in wav_clean_gl__opus64 wav_clean_gl__opus32; do
    OUT="./results/${M}__$RUNG.jsonl"
    [ -f "${OUT%.jsonl}_scored.jsonl" ] && { echo "skip $M/$RUNG"; continue; }
    echo "=== [tclean] $M $RUNG $(date -Is) ==="
    "$VPY" -u grm_delivery.py transfer --rung "$RUNG" --model "$M" --out "$OUT" \
      || echo "FAILED $M/$RUNG"
  done
done
echo "=== [score] $(date -Is) ==="
$ENVROOT/codec-attack/bin/python -u score_all_harmbench.py results
echo "TCLEANDONE $(date -Is)"
