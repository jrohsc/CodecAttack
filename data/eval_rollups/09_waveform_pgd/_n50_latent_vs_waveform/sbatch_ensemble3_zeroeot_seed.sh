#!/bin/bash
# Zero-initialized 224-target replication of the successful 600-step Opus-EoT cell.
# SEED controls per-clip Opus EoT draws; the published seed-1 bundle did not
# explicitly seed those draws and is kept separate from this controlled sweep.
#SBATCH --partition=gpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH --exclude=gypsum-gpu115,gypsum-gpu162
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
SEED="${SEED:?set SEED=1, 7, or 42}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?submit array 0-3}"
[[ "$SEED" =~ ^(1|7|42)$ && "$SHARD_ID" =~ ^[0-3]$ ]] || exit 2
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
OUT="results/benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600_zeroeot_seed${SEED}_shard${SHARD_ID}of4"
if [[ -f "$OUT/config.json" ]]; then
  python - "$OUT/config.json" "$SEED" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
expected={'attack_steps':600,'eps':10.0,'alpha':0.2,'perceptual_weight':0.1,
          'prompt_mode':'transcribe','ensemble_models':['qwen2_audio','qwen25_omni','audio_flamingo'],
          'ensemble_warmup':50,'ensemble_channel_mode':'multi_bitrate',
          'ensemble_best_after_warmup':True,'rand_init':False,
          'seed':int(sys.argv[2]),'seed_codec_per_clip':True}
for key,value in expected.items():
    if c.get(key)!=value: raise SystemExit(f'Config mismatch: {key}: {c.get(key)} != {value}')
PY
fi
python -u run_benchmark.py --ensemble \
  --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
  --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
  --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
  --prompt-mode transcribe --seed "$SEED" --seed-codec-per-clip \
  --music jazz_1 --output-dir "$OUT" --commands "${COMMANDS[@]}"
N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/56"
(( N == 56 )) || exit 1
