#!/bin/bash
#SBATCH --job-name=e3o6_eval
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Scores one of four shards for a seed replicate: six models, seven channels.
# Prompt comes from each bundle's config.json (these jazz bundles have one), which
# is the transcribe prompt the attack was optimized under.
set -euo pipefail
SEED="${SEED:?set SEED}"; CODEC="${CODEC:-opus}"; STEPS="${STEPS:-600}"
ZERO_EOT="${ZERO_EOT:-0}"
REPO=${REPO_ROOT}
cd "$REPO"
export HF_HUB_OFFLINE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
export PATH="$ENVROOT/codec-attack/bin:$PATH" PYTHONUTF8=1 PYTHONNOUSERSITE=1
export KIMI_INFER_ROOT=${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio
export PYTHONPATH="$KIMI_INFER_ROOT:${PYTHONPATH:-}"
CH="opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"
py_for() { case "$1" in
  qwen2_audio) echo "$ENVROOT/codec-attack/bin/python" ;;
  qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
  audio_flamingo) echo "$ENVROOT/flamingo3/bin/python" ;;
  kimi_audio) echo "$ENVROOT/kimi-audio/bin/python" ;;
  phi4) echo "$ENVROOT/audio-llm/bin/python" ;; esac; }

for SH in "${SLURM_ARRAY_TASK_ID:?submit array 0-3}"; do
  [[ "$SH" =~ ^[0-3]$ ]] || exit 2
  if [[ "$ZERO_EOT" == 1 ]]; then
    B="results/benchmark_ensemble3_${CODEC}eot_transcribe_eps10.0_jazz_1_bestpostcodec_steps${STEPS}_zeroeot_seed${SEED}_shard${SH}of4"
  else
    B="results/benchmark_ensemble3_${CODEC}eot_transcribe_eps10.0_jazz_1_bestpostcodec_steps${STEPS}_seed${SEED}_shard${SH}of4"
  fi
  n=$(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l)
  echo "=== [$(date -Is)] seed=$SEED shard=$SH wavs=$n/56"
  [ "$n" -eq 56 ] || { echo "ERROR: incomplete shard $B ($n/56)" >&2; exit 1; }
  for SPEC in "qwen2_audio qwen2_MEMBER" "qwen25_omni omni7b_MEMBER" \
              "audio_flamingo af3_MEMBER" "qwen25_omni_3b omni3b_HELDOUT" \
              "kimi_audio kimi_HELDOUT_100tok" "phi4 phi4_HELDOUT"; do
    set -- $SPEC; M=$1; OUT=$2
    echo "--- [$(date -Is)] seed=$SEED shard=$SH eval $M -> $OUT"
    PY=("$(py_for "$M")")
    if [[ "$M" == qwen25_omni || "$M" == qwen25_omni_3b ]]; then
      PY=(env -u PYTHONNOUSERSITE "${PY[0]}")
    fi
    EXTRA=()
    [[ "$M" == kimi_audio ]] && EXTRA=(--max-new-tokens 100)
    "${PY[@]}" 0_all_combined/scripts/ensemble_heldout_eval.py \
        --bundle-dir "$B" --target-model "$M" --out-name "$OUT" --channels "$CH" \
        "${EXTRA[@]}"
  done
done
echo "=== [$(date -Is)] SEED $SEED EVAL DONE"
