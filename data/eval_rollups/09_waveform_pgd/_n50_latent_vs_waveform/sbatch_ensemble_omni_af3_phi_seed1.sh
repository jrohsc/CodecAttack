#!/bin/bash
# 224 robot-action targets; one zero-initialized alternate-family ensemble.
#SBATCH --job-name=e3phi1
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
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
SHARD_ID="${SLURM_ARRAY_TASK_ID:?submit array 0-3}"
[[ "$SHARD_ID" =~ ^[0-3]$ ]] || exit 2
REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack
mapfile -t ALL_COMMANDS < <(python - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
(( ${#ALL_COMMANDS[@]} == 224 )) || exit 2
COMMANDS=()
for ((i=SHARD_ID; i<224; i+=4)); do COMMANDS+=("${ALL_COMMANDS[$i]}"); done
(( ${#COMMANDS[@]} == 56 )) || exit 2
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="results/benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed1_shard${SHARD_ID}of4"
if [[ -f "$OUT/config.json" ]]; then
  python - "$OUT/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
expected={'attack_steps':600,'eps':10.0,'alpha':0.2,'perceptual_weight':0.1,
          'target_model':'qwen25_omni','prompt_mode':'transcribe',
          'ensemble_models':['qwen25_omni','audio_flamingo','phi4'],
          'ensemble_warmup':50,'ensemble_channel_mode':'multi_bitrate',
          'ensemble_best_after_warmup':True,'rand_init':False,'seed':1,
          'seed_codec_per_clip':False,'ensemble_strict_workers':True}
for key,value in expected.items():
    if c.get(key)!=value: raise SystemExit(f'Config mismatch: {key}: {c.get(key)} != {value}')
PY
fi
python -u run_benchmark.py --ensemble \
  --ensemble-models qwen25_omni,audio_flamingo,phi4 \
  --target-model qwen25_omni \
  --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
  --ensemble-strict-workers \
  --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
  --prompt-mode transcribe --seed 1 \
  --music jazz_1 --output-dir "$OUT" --commands "${COMMANDS[@]}"
N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/56"
(( N == 56 )) || exit 1
