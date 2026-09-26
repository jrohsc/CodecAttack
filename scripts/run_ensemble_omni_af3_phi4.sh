#!/bin/bash
# Three-model Opus-EoT ensemble (Qwen2.5-Omni-7B + Audio Flamingo 3 + Phi-4-MM),
# held-out evaluation, and per-seed summary. See docs/ENSEMBLE.md.
#
#   SEED=1  bash scripts/run_ensemble_omni_af3_phi4.sh          # all 4 shards
#   SEED=7  SHARDS="0 1" bash scripts/run_ensemble_omni_af3_phi4.sh
#   STAGE=eval SEED=7 bash scripts/run_ensemble_omni_af3_phi4.sh
#
# STAGE: craft | eval | summarize | all (default). Shards are independent, so
# they can run on separate GPUs in parallel (one SHARDS value per GPU).
# Needs the conda envs codec-attack, qwen-omni, flamingo3, audio-llm and
# kimi-audio (named exactly so; ensemble members are routed by env name).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-1}"
SHARDS="${SHARDS:-0 1 2 3}"
STAGE="${STAGE:-all}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
ENVS="$CONDA_BASE/envs"
PREFIX="benchmark_ensemble_omni7_af3_phi4_opuseot_transcribe_eps10.0_jazz_1_steps600_seed${SEED}"
CHANNELS="opus64,opus128,mp3_64,mp3_128,aac_64,aac_128"

# Seed 1 is the original run; later seeds also seed each clip's Opus EoT draws.
SEED_ARGS=(--seed "$SEED")
[[ "$SEED" == 1 ]] || SEED_ARGS+=(--seed-codec-per-clip)

py_for() {
  case "$1" in
    qwen2_audio) echo "$ENVS/codec-attack/bin/python" ;;
    qwen25_omni|qwen25_omni_3b) echo "$ENVS/qwen-omni/bin/python" ;;
    audio_flamingo) echo "$ENVS/flamingo3/bin/python" ;;
    phi4) echo "$ENVS/audio-llm/bin/python" ;;
    kimi_audio) echo "$ENVS/kimi-audio/bin/python" ;;
  esac
}

# (model, output name) pairs: the three members, then the held-out victims.
VICTIMS=(
  "qwen25_omni omni7b_MEMBER"
  "audio_flamingo af3_MEMBER"
  "phi4 phi4_MEMBER"
  "qwen2_audio qwen2_HELDOUT"
  "kimi_audio kimi_HELDOUT_100tok"
  "qwen25_omni_3b omni3b_HELDOUT"
)

craft() {
  local sh=$1
  mapfile -t ALL < <("$ENVS/codec-attack/bin/python" - <<'PY'
import sys; sys.path.insert(0, "scripts")
from run_benchmark import filter_commands
print("\n".join(filter_commands(None, mode="transcribe", dataset="ours")))
PY
)
  (( ${#ALL[@]} == 224 )) || { echo "expected 224 targets, got ${#ALL[@]}" >&2; exit 2; }
  local cmds=()
  for ((i=sh; i<224; i+=4)); do cmds+=("${ALL[$i]}"); done
  ENSEMBLE_PERSISTENT_WORKER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ENVS/codec-attack/bin/python" -u scripts/run_benchmark.py --ensemble \
    --ensemble-models qwen25_omni,audio_flamingo,phi4 \
    --target-model qwen25_omni \
    --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
    --ensemble-strict-workers \
    --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
    --prompt-mode transcribe "${SEED_ARGS[@]}" \
    --music jazz_1 --output-dir "results/${PREFIX}_shard${sh}of4" \
    --commands "${cmds[@]}"
}

evaluate() {
  local sh=$1 bundle="results/${PREFIX}_shard${sh}of4"
  local n; n=$(find "$bundle/audio" -maxdepth 1 -name '*.wav' | wc -l)
  [[ "$n" == 56 ]] || { echo "incomplete shard $bundle: $n/56 wavs" >&2; exit 1; }
  for spec in "${VICTIMS[@]}"; do
    set -- $spec
    local extra=()
    [[ "$1" == kimi_audio ]] && extra=(--max-new-tokens 100)
    "$(py_for "$1")" scripts/ensemble_heldout_eval.py \
      --bundle-dir "$bundle" --target-model "$1" --out-name "$2" \
      --channels "$CHANNELS" "${extra[@]}"
  done
}

summarize() {
  local extra=()
  [[ "$SEED" == 1 ]] || extra=(--require-seeded-codec)
  for spec in "${VICTIMS[@]}"; do
    set -- $spec
    python scripts/aggregate_extra_heldout.py --prefix "$PREFIX" \
      --out-name "$2" --model "$1" --expected-seed "$SEED" \
      --expected-init zero "${extra[@]}"
  done
  python scripts/summarize_ensemble_omni_af3_phi.py --seed "$SEED"
}

if [[ "$STAGE" == craft || "$STAGE" == all ]]; then
  for sh in $SHARDS; do craft "$sh"; done
fi
if [[ "$STAGE" == eval || "$STAGE" == all ]]; then
  for sh in $SHARDS; do evaluate "$sh"; done
fi
if [[ "$STAGE" == summarize || "$STAGE" == all ]]; then
  summarize
fi
