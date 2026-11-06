#!/bin/bash
# Seed replicates of the corrected three-member codec EoT cell (eps=10, 600 steps).
# Submit as an array 0-3 with SEED=1, 7 or 42 and CODEC=opus|mp3. Writes to a
# seed-scoped output dir; the published seed-1 bundles are never touched.
# --rand-init is MANDATORY here: with the default zeros-init the latent attack is
# fully deterministic, so a seed sweep without it yields byte-identical bundles
# (see run_benchmark.py --rand-init help and ensemble_attack.py init block).
# SEED=1 is therefore a NEW rand-init cell, not the existing unseeded bundle --
# it is what makes the 1/7/42 error bar homogeneous, matching the S1 convention
# in run_s1_ensemble_bundleA.py (--seed 1 --rand-init).
# Results are separate from the original 300-step, unrestricted-best bundles.
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

CODEC="${CODEC:?set CODEC=opus or mp3}"
SHARD_ID="${SLURM_ARRAY_TASK_ID:?submit as an array 0-3}"
SEED="${SEED:?set SEED=1, 7 or 42}"
NUM_SHARDS="${NUM_SHARDS:-4}"
STEPS="${STEPS:-600}"
PILOT="${PILOT:-0}"
case "$CODEC" in
    opus) CHANNEL_MODE=multi_bitrate ;;
    mp3)  CHANNEL_MODE=mp3_multibitrate ;;
    *) echo "ERROR: CODEC must be opus or mp3" >&2; exit 2 ;;
esac
[[ "$SHARD_ID" =~ ^[0-9]+$ && "$NUM_SHARDS" =~ ^[1-9][0-9]*$ \
   && "$STEPS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || exit 2
(( SHARD_ID < NUM_SHARDS )) || exit 2
[[ "$PILOT" == 0 || "$PILOT" == 1 ]] || exit 2

REPO=${REPO_ROOT}
cd "$REPO"
source ${WORK_ROOT}/miniconda3/etc/profile.d/conda.sh
conda activate codec-attack

mapfile -t ALL_COMMANDS < <(python - <<'PY'
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
(( ${#ALL_COMMANDS[@]} == 224 && NUM_SHARDS == 4 )) || exit 2
COMMANDS=()
for ((i=SHARD_ID; i<${#ALL_COMMANDS[@]}; i+=NUM_SHARDS)); do
    COMMANDS+=("${ALL_COMMANDS[$i]}")
done
(( ${#COMMANDS[@]} == 56 )) || exit 2
TAG=""
if (( PILOT )); then
    COMMANDS=("${COMMANDS[@]:0:2}")
    TAG="_pilot2"
fi

export ENSEMBLE_PERSISTENT_WORKER=1
export ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="results/benchmark_ensemble3_${CODEC}eot_transcribe_eps10.0_jazz_1_bestpostcodec_steps${STEPS}${TAG}_seed${SEED}_shard${SHARD_ID}of${NUM_SHARDS}"
echo "Corrected ${CODEC} EoT; steps=${STEPS}; best only after codec warmup; ${#COMMANDS[@]} commands; output=$OUT"

# run_benchmark resumes existing records but does not compare old config with
# current arguments. Reject a mismatched directory before loading any model.
if [[ -f "$OUT/config.json" ]]; then
    python - "$OUT/config.json" "$STEPS" "$CHANNEL_MODE" "$SEED" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
expected = (int(sys.argv[2]), 10.0, 'transcribe',
            ['qwen2_audio', 'qwen25_omni', 'audio_flamingo'],
            sys.argv[3], True, True, int(sys.argv[4]))
found = (config.get('attack_steps'), config.get('eps'), config.get('prompt_mode'),
         config.get('ensemble_models'), config.get('ensemble_channel_mode'),
         config.get('ensemble_best_after_warmup'), config.get('rand_init'),
         config.get('seed'))
if found != expected:
    raise SystemExit(f'ERROR: existing config {found} != requested {expected}')
PY
fi

python -u run_benchmark.py --ensemble \
    --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
    --ensemble-channel-mode "$CHANNEL_MODE" \
    --ensemble-best-after-warmup \
    --steps "$STEPS" --eps 10.0 --prompt-mode transcribe \
    --seed "$SEED" --rand-init \
    --music jazz_1 --output-dir "$OUT" \
    --commands "${COMMANDS[@]}"

N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/${#COMMANDS[@]}"
(( N == ${#COMMANDS[@]} )) || exit 1
if grep -q "subprocess gradient requests FAILED" \
    "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"; then
    echo "ERROR: AF3 gradient failures detected" >&2
    exit 1
fi
