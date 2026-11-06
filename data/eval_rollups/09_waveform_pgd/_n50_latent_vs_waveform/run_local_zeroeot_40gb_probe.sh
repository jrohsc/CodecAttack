#!/bin/bash
# One-clip feasibility test of the exact zero-init, 600-step, 3-member recipe.
# Uses the local 40-GB GPU only after the full-codec evaluation exits.
set -euo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
ENVROOT=${WORK_ROOT}/miniconda3/envs
WAIT_PID="${WAIT_PID:?set WAIT_PID to the full-grid driver PID}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
if [[ ! -f results/benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600_FULL_CODEC_GRID_AGGREGATED.json ]]; then
  echo 'Full-grid evaluation did not finish cleanly; local training probe deferred.' >&2
  exit 3
fi
SCRATCH="$REPO/.zeroeot_local_probe_tmp"
mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH" TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor"
export PATH="$ENVROOT/codec-attack/bin:$PATH" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1 ENSEMBLE_WORKER_GPU=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600_zeroeot_LOCAL40GB_PROBE_seed7
"$ENVROOT/codec-attack/bin/python" -u run_benchmark.py --ensemble \
  --ensemble-models qwen2_audio,qwen25_omni,audio_flamingo \
  --ensemble-channel-mode multi_bitrate --ensemble-best-after-warmup \
  --steps 600 --eps 10.0 --alpha 0.2 --perceptual-weight 0.1 \
  --prompt-mode transcribe --seed 7 --seed-codec-per-clip \
  --music jazz_1 --output-dir "$OUT" --commands nav_turn_left
N=$(find "$OUT/audio" -maxdepth 1 -name '*.wav' | wc -l)
echo "local probe wavs: $N/1"
(( N == 1 ))
