#!/usr/bin/env bash
# Use the current interactive A100 for a few shard-0 clips before Slurm starts.
set -euo pipefail
cd ${REPO_ROOT}
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
mapfile -t ALL_COMMANDS < <(python - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
(( ${#ALL_COMMANDS[@]} == 224 )) || exit 2
COMMANDS=()
for ((i=0; i<32; i+=4)); do COMMANDS+=("${ALL_COMMANDS[$i]}"); done
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed1_shard0of4
python -u run_benchmark.py --ensemble \
  --ensemble-models qwen25_omni,audio_flamingo,phi4 \
  --target-model qwen25_omni \
  --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
  --ensemble-strict-workers \
  --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
  --prompt-mode transcribe --seed 1 \
  --music jazz_1 --output-dir "$OUT" --commands "${COMMANDS[@]}"
