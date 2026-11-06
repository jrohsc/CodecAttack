#!/bin/bash
# Preflight the pool-ladder / white-box-ceiling attack path before re-spending
# ~100 GPU-h on it. Four steps, one clip, per cell.
#
# The ladder has now produced two generations of bundles that look complete but
# were written from unattacked carriers -- see the header of
# 0_all_combined/scripts/preflight_ensemble_workers.py for the two silent
# degradation paths. Every one of them would have been caught by a 4-step run
# that simply checked "did each pool member contribute a gradient". That check is
# this script. Read the VERDICT line in each log; do not submit the real cells
# until they all say PASS.
#
# Usage: ./submit_preflight_workers.sh [--dry-run]

set -euo pipefail
cd ${REPO_ROOT}
mkdir -p logs

DRY=0
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; esac; done

PF=0_all_combined/scripts/preflight_ensemble_workers.py
CA=${WORK_ROOT}/miniconda3/envs

submit () {  # $1=name $2=env $3=gres $4=extra-env $5...=py args
  local name="$1" env="$2" gres="$3" xenv="$4"; shift 4
  if squeue -u "$USER" -h -o "%j" | grep -qx "$name"; then
    echo "SKIP (already queued/running): $name"; return
  fi
  if [ "$DRY" = "1" ]; then echo "DRY: $name [$env $gres] :: $*"; return; fi
  sbatch --job-name="$name" \
         --partition=gpu,gpu-preempt --gres="$gres" --constraint=vram80 \
         --nodes=1 --requeue --mem=128G --cpus-per-task=8 --time=01:00:00 \
         --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
         --wrap="\
export PATH=$CA/codec-attack/bin:\$PATH && \
export HF_HUB_OFFLINE=1 && \
export PYTORCH_ALLOC_CONF=expandable_segments:True && \
${xenv}$CA/$env/bin/python $PF $*" \
    | sed "s/^/[$name] /"
}

# --- white-box ceilings: single victim, in-process, no worker involved. These
#     gate the four ceiling re-runs. The failure they check for is the
#     primary_model default, so a PASS here means the ceiling bundle will be
#     written from a carrier that was actually optimized. ---
submit pf_ceil_qwen2_audio    codec-attack gpu:1 ""  --pool qwen2_audio
submit pf_ceil_qwen25_omni    codec-attack gpu:1 ""  --pool qwen25_omni
submit pf_ceil_audio_flamingo flamingo3    gpu:1 ""  --pool audio_flamingo
submit pf_ceil_phi4           audio-llm    gpu:1 ""  --pool phi4

# --- out-of-env persistent workers: the path that lost ~50% of every N>=2
#     ladder cell's gradient steps. Worker gets its own card, as the real cells
#     do, so a PASS here is a PASS under the same memory conditions. ---
WORKER_ENV="export ENSEMBLE_PERSISTENT_WORKER=1 && export ENSEMBLE_WORKER_GPU=1 && "
submit pf_worker_audio_flamingo codec-attack gpu:2 "$WORKER_ENV" --pool qwen2_audio,audio_flamingo
submit pf_worker_phi4           codec-attack gpu:2 "$WORKER_ENV" --pool qwen2_audio,phi4
submit pf_worker_gemma3n        codec-attack gpu:1 ""            --pool qwen2_audio,gemma3n

echo
echo "Verdicts:  grep -h 'VERDICT' logs/pf_*_*.out"
