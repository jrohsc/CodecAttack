#!/usr/bin/env bash
# Run a multi-seed S1 bundle locally (no SLURM). Same methodology as the sbatch
# job: --seed <S> --rand-init, 1000 steps, n=50, + MP3/AAC sidecar. Resumable.
#
# Usage:
#   bash run_local_seed.sh <SEED> [latent|waveform|both]   # default: both (chained)
#
# Two GPUs? Run the halves in separate terminals:
#   CUDA_VISIBLE_DEVICES=0 bash run_local_seed.sh 7 latent
#   CUDA_VISIBLE_DEVICES=1 bash run_local_seed.sh 7 waveform
set -uo pipefail
SEED="${1:?usage: run_local_seed.sh <SEED> [latent|waveform|both]}"
WHICH="${2:-both}"
REPO=${REPO_ROOT}
cd "$REPO"
PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
export CODEC_ATTACK_PY="$PY"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCEN=01_finance_voice_agent
LAT=0_all_combined/results_codec_robust/${SCEN}/qwen2_audio/eps_1.0_seed${SEED}
WAV=0_all_combined/results_codec_robust_waveform/${SCEN}/qwen2_audio/eps_0.075_seed${SEED}
ts(){ date "+%F %H:%M:%S"; }
echo "=== [$(ts)] LOCAL seed=$SEED which=$WHICH GPU=$CUDA_VISIBLE_DEVICES ==="

if [ "$WHICH" = "latent" ] || [ "$WHICH" = "both" ]; then
  echo "=== [$(ts)] LATENT attack ==="
  $PY -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
      --target-model qwen2_audio --eps 1.0 --seed "$SEED" --rand-init
  echo "=== [$(ts)] LATENT sidecar ==="
  $PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
      --bundle-dir "$LAT" --target-model qwen2_audio
fi
if [ "$WHICH" = "waveform" ] || [ "$WHICH" = "both" ]; then
  echo "=== [$(ts)] WAVEFORM attack ==="
  $PY -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
      --eps 0.075 --seed "$SEED" --rand-init
  echo "=== [$(ts)] WAVEFORM sidecar ==="
  $PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
      --bundle-dir "$WAV" --target-model qwen2_audio
fi
echo "=== [$(ts)] seed=$SEED ($WHICH) DONE ==="
