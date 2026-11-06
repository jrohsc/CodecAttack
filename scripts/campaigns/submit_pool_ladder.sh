#!/bin/bash
# Submit the surrogate-pool scaling ladder (reviewer W5).
#
# 16 ladder cells (4 held-out victims x rungs N=1..4). White-box ceilings are NOT
# submitted here any more -- they already exist from the paper's single-model
# codec-robust attack; see the block near the bottom of this file.
# Every cell is a separate job so they fan out across the partition; each is
# resumable (skips wavs already on disk), so a preempted job just needs
# resubmitting. Cells whose pool contains an out-of-env surrogate get gpu:2 (the
# worker takes card 1); worker-free cells get gpu:1 and queue much faster.
#
# Submit the worker-free rungs first with --worker-free-only: after the 2026-08-24
# reorder those are N=1..3 for every victim, they carry none of the cross-env
# fragility that voided the previous ladder, and they alone answer whether
# transfer emerges with encoder diversity (gemma3n/USM enters at N=3).
#
# Guarded against double submission by job name: two sessions submitting into
# the same tree would otherwise interleave writes into one bundle.
#
# Usage:
#   ./submit_pool_ladder.sh [--model-eot] [--dry-run] [--seed N] [--pairs N]

set -euo pipefail
cd ${REPO_ROOT}
mkdir -p logs

SEED=1
PAIRS=50
EOT=""
DRY=0
# Relative to 0_all_combined/ (run_pool_ladder.py resolves it against its own
# ROOT), NOT to the repo root -- a leading "0_all_combined/" here doubles the
# prefix and writes a parallel tree. The driver now refuses that outright.
OUT_BASE="results_pool_ladder_v2/01_finance_voice_agent"
for a in "$@"; do
  case "$a" in
    --model-eot) EOT="--model-eot" ;;
    --dry-run)   DRY=1 ;;
    --seed=*)    SEED="${a#*=}" ;;
    --pairs=*)   PAIRS="${a#*=}" ;;
    --out-base=*) OUT_BASE="${a#*=}" ;;
    --victims=*) IFS=',' read -r -a VICTIMS_OVERRIDE <<< "${a#*=}" ;;
    --ceilings-only) CEIL_ONLY=1 ;;
    --worker-free-only) WF_ONLY=1 ;;
    # Resubmit one dead rung without also re-launching its finished siblings.
    # A finished cell is resumable and would just reload every surrogate (~10 min
    # of GPU) before exiting on a full bundle -- harmless, but it burns a vram80
    # slot the live cells are queued for.
    --rungs=*)   IFS=',' read -r -a RUNGS_OVERRIDE <<< "${a#*=}" ;;
  esac
done

# Held-out victims. phi4 is included only once its white-box ceiling shows the
# attack can land on it at all -- an unattackable victim yields a 0 that cannot
# be distinguished from "no transfer". Override with --victims=a,b,c
VICTIMS=(qwen2_audio qwen25_omni audio_flamingo phi4)
if [ -n "${VICTIMS_OVERRIDE+x}" ]; then VICTIMS=("${VICTIMS_OVERRIDE[@]}"); fi
CEIL_ONLY=${CEIL_ONLY:-0}
# REORDERED 2026-08-24 to match run_pool_ladder.py: gemma3n (USM, in-process)
# moves to position 3 so rungs N=1..3 are worker-free for every victim. Keep the
# two lists identical -- the driver is the authority, this is only used to decide
# which cells need a second GPU for an out-of-env worker.
ORDER=(qwen2_audio qwen25_omni gemma3n audio_flamingo phi4)
# Surrogates that need a cross-env persistent gradient worker.
OUT_OF_ENV="audio_flamingo phi4"
RUNGS=(1 2 3 4)
WF_ONLY=${WF_ONLY:-0}

# Pool for (victim, rung) under ORDER, and whether it touches the worker path.
pool_for () {  # $1=victim $2=rung -> echoes space-separated pool
  local v="$1" n="$2" out=() m
  for m in "${ORDER[@]}"; do [ "$m" != "$v" ] && out+=("$m"); done
  echo "${out[@]:0:$n}"
}
needs_worker () {  # $1=victim $2=rung -> 0 if an out-of-env surrogate is in the pool
  local p m w
  p=$(pool_for "$1" "$2")
  for m in $p; do for w in $OUT_OF_ENV; do [ "$m" = "$w" ] && return 0; done; done
  return 1
}
if [ -n "${RUNGS_OVERRIDE+x}" ]; then RUNGS=("${RUNGS_OVERRIDE[@]}"); fi

# Which interpreter a cell runs under. EnsembleLatentAttacker requires at least
# one IN-PROCESS member (it needs a live model for its progress-check generate),
# so a phi4-only white-box cell cannot run from codec-attack -- phi4 would be
# subprocess-only and the run dies with "At least one in-process model is
# required". Those cells run from audio-llm instead, with codec-attack's bin on
# PATH for ffmpeg (DifferentiableOpusProxy shells out to it).
py_for () {
  case "$1" in
    wbceil_phi4_*)           echo "${WORK_ROOT}/miniconda3/envs/audio-llm/bin/python" ;;
    wbceil_audio_flamingo_*) echo "${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python" ;;
    *)                       echo "${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python" ;;
  esac
}

# Per-cell env overrides. audio_flamingo's N=4 pool (qwen2_audio + qwen25_omni +
# phi4 + gemma3n) is the heaviest of the four N=4 pools and OOMed at 79.23/79.25 GB
# on an 80 GB A100 77 s in -- expandable_segments was already set, so the only
# lever left is trading compute for activation memory.
# audio_flamingo's N=4 pool is the ONLY cell with three in-process models:
# holding out audio_flamingo leaves {qwen2_audio, qwen25_omni, gemma3n} all
# resident plus a phi4 subprocess worker on the same card, where every other N=4
# pool offloads audio_flamingo to a second subprocess. It OOMed at 79.2/79.2 GB
# twice -- once bare, once with ENSEMBLE_GRAD_CKPT=1, both in backward. So give
# the phi4 worker its own card (ensemble_attack.py:220 names exactly this fix)
# and keep checkpointing on for the three that still share card 0.
# Why not grad-checkpointing: under model_eot only ONE model is active per step,
# so activations never stack -- the pressure is RESIDENT WEIGHTS, and
# checkpointing does nothing about those (it was tried, and OOMed identically).
# Card-0 residency, measured from weights on disk:
#   af3 N4  = 46.5 GB in-process (q2a 15.6 + omni 20.8 + gemma 10.1) + 10.4 phi4
#             worker on the SAME card = 56.9 GB, leaving ~22 GB for the backward.
# ENSEMBLE_WORKER_GPU=1 moves the worker to card 1, taking card 0 down to 46.5 GB
# (~33 GB headroom) and card 1 to 10.4 of 80. This changes only WHERE a gradient
# is computed -- not the pool, its order, or the EoT sample sequence -- so cells
# stay comparable to the already-finished N1/N2 bundles.
#
# ENSEMBLE_PERSISTENT_WORKER=1 is the other half. Without it every subprocess step
# respawns and RELOADS the model (~46 s, ensemble_attack.py:286). It shows up
# directly in the cell timings: N1 (no subprocess model) ran 6.1 min/pair, N2
# (+audio_flamingo) 44, N3/N4 (+phi4) 62-75. Loading once should pull that back
# toward ~10 min/pair. A worker that fails to start RAISES rather than silently
# running a degraded ensemble, so a bad persistent path fails loud.
# Routed by whether the cell's POOL actually contains an out-of-env surrogate,
# not by rung number. Under the old order N3/N4 was a good proxy for that; after
# the reorder it is wrong (audio_flamingo N3 is now worker-free, and asking for
# gpu:2 would just queue it behind two-card availability for nothing).
env_for () {   # $1=victim $2=rung
  if needs_worker "$1" "$2"; then
    echo "export ENSEMBLE_WORKER_GPU=1 && export ENSEMBLE_PERSISTENT_WORKER=1 && "
  else
    echo ""
  fi
}

gres_for () {  # $1=victim $2=rung
  if needs_worker "$1" "$2"; then echo "gpu:2"; else echo "gpu:1"; fi
}

submit () {  # $1=jobname $2=victim $3=rung  $4...=python args
  local name="$1" _v="$2" _n="$3"; shift 3
  if squeue -u "$USER" -h -o "%j" | grep -qx "$name"; then
    echo "SKIP (already queued/running): $name"; return
  fi
  if [ "$DRY" = "1" ]; then
    echo "DRY: $name :: $*"; return
  fi
  # --constraint=vram80 is REQUIRED, not a preference. Every surrogate is
  # resident on the same GPU (subprocess workers inherit CUDA_VISIBLE_DEVICES),
  # so an N=3 cell measured ~65 GB and N=4 is tighter still. The gpu partition is
  # mostly 32-48 GB cards (v100/a40/l40s/2080ti); landing on one is a guaranteed
  # OOM part-way through a cell.
  # gpu,gpu-preempt: the `gpu` partition holds only 28 vram80 GPUs and they sit
  # saturated (all 16 cells stuck PENDING/Priority with no start estimate), while
  # gpu-preempt exposes ~100+ more of the same 80 GB a100/h100 nodes. Preemption
  # is safe here BECAUSE the ladder driver is resumable: it skips wavs already on
  # disk, so a requeued cell continues rather than restarting. --requeue makes
  # that automatic. Still count wavs per cell before scoring -- a preempted job
  # can log COMPLETED on a partial bundle.
  sbatch --job-name="$name" \
         --partition=gpu,gpu-preempt --gres=$(gres_for "$_v" "$_n") --constraint=vram80 \
         --nodes=1 \
         --requeue --mem=128G --cpus-per-task=8 \
         --time=2-00:00:00 \
         --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
         --wrap="cd ${REPO_ROOT} && \
export PATH=${WORK_ROOT}/miniconda3/envs/codec-attack/bin:\$PATH && \
export HF_HUB_OFFLINE=1 && \
export PYTORCH_ALLOC_CONF=expandable_segments:True && \
$(env_for "$_v" "$_n")$(py_for "$name") $*" \
    | sed "s/^/[$name] /"
}

# --- ladder: victim v held out, first N surrogates of ORDER minus v ---
[ "$CEIL_ONLY" = "1" ] || for v in "${VICTIMS[@]}"; do
  pool=(); for m in "${ORDER[@]}"; do [ "$m" != "$v" ] && pool+=("$m"); done
  for N in "${RUNGS[@]}"; do
    if [ "$WF_ONLY" = "1" ] && needs_worker "$v" "$N"; then
      echo "SKIP (needs out-of-env worker, --worker-free-only): ladder_${v}_N${N}"
      continue
    fi
    submit "ladder_${v}_N${N}_s${SEED}" "$v" "$N" \
      "0_all_combined/scripts/run_pool_ladder.py --victim $v --rung $N \
       --seed $SEED --max-pairs $PAIRS --out-base '$OUT_BASE' $EOT"
  done
done

# --- white-box ceilings: NOT submitted from here any more. ---
# They already exist, produced by the paper's own single-model codec-robust
# attack on the SAME carriers, targets, eps, step count, Opus EoT grid and a
# byte-identical BANK_PROMPT:
#
#   results_codec_robust/01_finance_voice_agent/<model>/eps_1.0/
#   qwen2_audio 88.0%  qwen25_omni 82.0%  audio_flamingo 44.0%  (clean, n=50)
#
# Running them through run_s1_ensemble_bundleA.py with a ONE-model pool bought
# nothing and is what introduced the primary_model latch that voided three of the
# four ceiling bundles. phi4 is the only ceiling genuinely missing; produce it on
# the same validated single-model path (latent_attack.py now loads models via
# model_registry, so it accepts phi4):
#
#   CODEC_ATTACK_PY=$CONDA_PREFIX_ROOT/envs/audio-llm/bin/python \
#     python 0_all_combined/scripts/run_s1_attack_bundleA.py \
#       --target-model phi4 --eps 1.0 --channel-mode multi_bitrate
if [ "${CEIL_ONLY:-0}" = "1" ]; then
  echo "ERROR: --ceilings-only is retired; see the comment above this line." >&2
  exit 2
fi

echo
echo "Submitted. Track with: squeue -u \$USER -o '%.18i %.30j %.8T %.10M'"
echo "NOTE: phi4 cells need the audio-llm env in-process; they auto-route to a"
echo "      subprocess worker when run from codec-attack, which is the intent."
