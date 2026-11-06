#!/bin/bash
# Qwen2 + Omni-7B targeted transcription attack on the 16 kHz waveform.
# Four shards cover the same 224 jazz_1 commands as the latent ensemble arms.
#SBATCH --partition=gpu-preempt
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
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
SHARD_ID="${SHARD_ID:?set SHARD_ID=0..3}"
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
for ((i=SHARD_ID; i<224; i+=4)); do
    COMMANDS+=("${ALL_COMMANDS[$i]}")
done
(( ${#COMMANDS[@]} == 56 )) || exit 2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="results/benchmark_ens2_waveform_transcribe_eps0.075_jazz_1_shard${SHARD_ID}of4"
echo "Waveform Qwen2+Omni shard $SHARD_ID/4; 56 commands; output=$OUT"

if [[ -f "$OUT/config.json" ]]; then
    python - "$OUT/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
expected={'ensemble_domain':'waveform','ensemble_models':['qwen2_audio','qwen25_omni'],
          'attack_steps':300,'eps':0.075,'alpha':0.015,'ensemble_warmup':50,
          'ensemble_channel_mode':'none','ensemble_best_after_warmup':True}
for key,value in expected.items():
    if c.get(key)!=value:raise SystemExit(f'config mismatch: {key}: {c.get(key)!r} != {value!r}')
PY
fi

python -u run_benchmark.py --ensemble --ensemble-domain waveform \
    --ensemble-models qwen2_audio,qwen25_omni \
    --ensemble-best-after-warmup --ensemble-warmup 50 \
    --ensemble-channel-mode none \
    --music jazz_1 --commands "${COMMANDS[@]}" \
    --eps 0.075 --alpha 0.015 --steps 300 \
    --output-dir "$OUT"

N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "crafted wavs: $N/56"
(( N == 56 )) || exit 1
