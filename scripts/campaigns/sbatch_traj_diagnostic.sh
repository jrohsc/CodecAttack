#!/bin/bash
# TRAJECTORY DIAGNOSTIC -- does transfer exist anywhere along the PGD path?
#
# Every transfer cell we have is measured at the FINAL step. This job saves the
# perturbation every 100 steps for a handful of pairs and scores each checkpoint
# on the models that were NOT used for gradients.
#
#   all-zero along the trajectory -> nothing to select; OSLO/GA-style
#                                    trajectory-selection is ruled out cheaply
#   non-zero then decaying to 0   -> the perturbation overfits the source, and
#                                    trajectory selection is aimed at exactly that
#
# This costs ~1 GPU-hour and decides whether a ~31 GPU-hour ensemble ladder and a
# rung-5 implementation are worth running. Run it BEFORE committing to those.
#
# Source model = Qwen2-Audio (cheapest: no codec EoT on this path, ~0.04 s/step).
# Held-out     = Omni-7B, Omni-3B, AF3.
# Prompt       = transcribe, identical in optimization and evaluation (no swap).
#
# 5 tasks: task 0 = attack + checkpoints, tasks 1-4 = evaluate on each model.
# Submit the eval array with --dependency=afterok on the attack job.
#
#SBATCH --job-name=traj_diag
#SBATCH -p gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80g
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -o ${REPO_ROOT}/logs/traj_diag_%A_%a.log

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

PROMPT="What does the person say in this audio?"
CKPT=${REPO_ROOT}/0_all_combined/results_traj_diagnostic
MANIFEST=$CKPT/manifest.json
NPAIRS=${NPAIRS:-3}

PHASE=${PHASE:-attack}

if [[ "$PHASE" == "attack" ]]; then
    mkdir -p "$CKPT/wavs"
    conda activate codec-attack
    export ATTACK_CKPT_EVERY=100
    export ATTACK_CKPT_DIR="$CKPT/wavs"

    # Reuse the S1 driver so the attack is byte-for-byte the one behind the
    # published bundles -- only the checkpoint env vars differ.
    CARRIERS=$(python - "$NPAIRS" <<'PY'
import json, sys
n = int(sys.argv[1])
p = "0_all_combined/data/speech/01_finance/carrier_to_target.json"
pairs = sorted(json.load(open(p))["pairs"], key=lambda x: x["carrier"])[:n]
print(" ".join(x["carrier"] for x in pairs))
PY
)
    echo "=== trajectory diagnostic: carriers = $CARRIERS ==="

    python -u 0_all_combined/scripts/run_s1_attack.py \
        --target-model qwen2_audio \
        --eps 1.0 \
        --prompt-text "$PROMPT" \
        --results-root results_traj_diagnostic/bundle \
        --carriers $CARRIERS

    # manifest: checkpoint stem -> target text
    python - "$NPAIRS" "$MANIFEST" <<'PY'
import json, sys
n, out = int(sys.argv[1]), sys.argv[2]
p = "0_all_combined/data/speech/01_finance/carrier_to_target.json"
pairs = sorted(json.load(open(p))["pairs"], key=lambda x: x["carrier"])[:n]
# robust_latent_attack names checkpoints by music_name, which run_attack.py sets
# to the carrier stem.
json.dump({x["carrier"].replace(".wav", ""): x["target"] for x in pairs},
          open(out, "w"), indent=1)
print(f"wrote {out}")
PY
    echo "=== checkpoints: $(ls "$CKPT/wavs" | wc -l) wavs ==="
    exit 0
fi

# ---- eval phase ----
MODELS=(qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo)
ENVS=(  codec-attack qwen-omni  qwen-omni      flamingo3)
i=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$i]}
ENV=${ENVS[$i]}
ROLE="HELD-OUT (transfer)"
[[ "$MODEL" == "qwen2_audio" ]] && ROLE="SOURCE (white-box reference)"

echo "=== trajectory eval: $MODEL [$ROLE] ==="
conda activate "$ENV"
python -u diagnose_trajectory_eval.py \
    --ckpt-dir "$CKPT/wavs" \
    --manifest "$MANIFEST" \
    --target-model "$MODEL" \
    --prompt-text "$PROMPT" \
    --out "$CKPT/traj_${MODEL}.json"
