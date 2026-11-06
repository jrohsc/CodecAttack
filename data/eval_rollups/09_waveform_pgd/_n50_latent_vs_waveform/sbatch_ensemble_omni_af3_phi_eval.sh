#!/bin/bash
#SBATCH --job-name=e3phi1eval
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
set -euo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
export PATH="$ENVROOT/codec-attack/bin:$PATH" PYTHONUTF8=1 PYTHONNOUSERSITE=1
export KIMI_INFER_ROOT=${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio
export PYTHONPATH="$KIMI_INFER_ROOT:${PYTHONPATH:-}"
SEED="${SEED:-1}"
[[ "$SEED" =~ ^(1|7|42)$ ]] || exit 2
PREFIX="benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed${SEED}"
CH="opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"
py_for() { case "$1" in
  qwen2_audio) echo "$ENVROOT/codec-attack/bin/python" ;;
  qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
  audio_flamingo) echo "$ENVROOT/flamingo3/bin/python" ;;
  kimi_audio) echo "$ENVROOT/kimi-audio/bin/python" ;;
  phi4) echo "$ENVROOT/audio-llm/bin/python" ;; esac; }
for SH in "${SLURM_ARRAY_TASK_ID:?submit array 0-3}"; do
  [[ "$SH" =~ ^[0-3]$ ]] || exit 2
  B="results/${PREFIX}_shard${SH}of4"
  n=$(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l)
  [[ "$n" == 56 ]] || { echo "Incomplete shard $B: $n/56" >&2; exit 1; }
  for SPEC in "qwen25_omni omni7b_MEMBER" "audio_flamingo af3_MEMBER" \
              "phi4 phi4_MEMBER" "qwen2_audio qwen2_HELDOUT" \
              "kimi_audio kimi_HELDOUT_100tok" "qwen25_omni_3b omni3b_HELDOUT"; do
    set -- $SPEC; M=$1; OUT=$2
    PY=("$(py_for "$M")")
    if [[ "$M" == qwen25_omni || "$M" == qwen25_omni_3b ]]; then
      PY=(env -u PYTHONNOUSERSITE "${PY[0]}")
    fi
    EXTRA=()
    [[ "$M" == kimi_audio ]] && EXTRA=(--max-new-tokens 100)
    "${PY[@]}" 0_all_combined/scripts/ensemble_heldout_eval.py \
      --bundle-dir "$B" --target-model "$M" --out-name "$OUT" \
      --channels "$CH" "${EXTRA[@]}"
  done
done
