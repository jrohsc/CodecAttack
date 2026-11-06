#!/bin/bash
# Run one disjoint MP3-EoT ensemble shard on the dashboard's local A100.
# AF3 shares the 80GB device with the two in-process members; use only where
# CUDA_VISIBLE_DEVICES exposes an A100-80GB.
set -euo pipefail

SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
[[ "$SHARD_ID" =~ ^[0-9]+$ && "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( SHARD_ID < NUM_SHARDS )) || exit 2

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

mapfile -t ALL_COMMANDS < <(python - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
COMMANDS=()
for ((i=SHARD_ID; i<${#ALL_COMMANDS[@]}; i+=NUM_SHARDS)); do
    COMMANDS+=("${ALL_COMMANDS[$i]}")
done
(( ${#ALL_COMMANDS[@]} == 224 && ${#COMMANDS[@]} == 56 )) || exit 2

export CUDA_VISIBLE_DEVICES=0
export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="results/benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1_shard${SHARD_ID}of${NUM_SHARDS}"
echo "LOCAL MP3-EoT ensemble shard ${SHARD_ID}/${NUM_SHARDS}; commands=${#COMMANDS[@]}"
python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --ensemble-channel-mode mp3_multibitrate \
    --music jazz_1 --output-dir "$OUT" \
    --commands "${COMMANDS[@]}"
