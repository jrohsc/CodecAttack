#!/bin/bash
# Serial local fallback for the seed 7 and 42 alternate-ensemble replications.
# Uses distinct output directories so it cannot collide with queued SLURM jobs.
set -euo pipefail

REPO=${REPO_ROOT}
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0 ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mapfile -t ALL_COMMANDS < <("$PY" - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
(( ${#ALL_COMMANDS[@]} == 224 )) || exit 2

for SEED in 7 42; do
  for SHARD_ID in 0 1 2 3; do
    PREFIX="benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_LOCAL_seed${SEED}"
    OUT="results/${PREFIX}_shard${SHARD_ID}of4"
    COMMANDS=()
    for ((i=SHARD_ID; i<224; i+=4)); do COMMANDS+=("${ALL_COMMANDS[$i]}"); done
    (( ${#COMMANDS[@]} == 56 )) || exit 2
    if [[ -d "$OUT/audio" ]]; then
      N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
      if (( N == 56 )); then
        echo "$(date -u) seed=$SEED shard=$SHARD_ID already complete"
        continue
      fi
    fi
    echo "$(date -u) starting seed=$SEED shard=$SHARD_ID"
    "$PY" -u run_benchmark.py --ensemble \
      --ensemble-models qwen25_omni,audio_flamingo,phi4 \
      --target-model qwen25_omni \
      --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
      --ensemble-strict-workers \
      --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
      --prompt-mode transcribe --seed "$SEED" --seed-codec-per-clip \
      --music jazz_1 --output-dir "$OUT" --commands "${COMMANDS[@]}"
    N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
    echo "$(date -u) seed=$SEED shard=$SHARD_ID crafted=$N/56"
    (( N == 56 )) || exit 1
  done
done
