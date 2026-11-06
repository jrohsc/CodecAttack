#!/bin/bash
# FOUR-way rung of the ensemble ladder: all three trainable victims co-optimized
# into ONE perturbation, under the MATCHED transcribe prompt.
#
#   ensemble: Qwen2-Audio + Qwen2.5-Omni-7B + Audio Flamingo 3
#   held-out: Qwen2.5-Omni-3B   (never trained on in any combo)
#
# Read against the pairwise runs (run_s1_ensemble3_transcribe_local.sh): if
# surrogate diversity is what buys transfer, Omni-3B should climb monotonically
# 1 victim -> 2 victims -> 3 victims. If it stays 0 at all three rungs, ensembling
# is not the missing ingredient and the encoder gap is the binding constraint.
#
# HARD RULE: optimization and evaluation use the identical prompt. Never swap.
#
# MEMORY: Q2A + Omni alone sit at ~57 GB on the 80 GB card, and AF3 joins as a
# persistent gradient worker in the flamingo3 env sharing the same GPU. If this
# OOMs, re-run with ENSEMBLE_GRAD_CKPT=1 (gradient checkpointing) -- do NOT drop a
# victim, that would silently turn this into a different rung.
#
# Usage: bash run_s1_ensemble_all3_transcribe_local.sh

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python
PROMPT="What does the person say in this audio?"
OUTBASE=results_transcribe_prompt_ensemble/01_finance_voice_agent
NAME=ens_all3
BUNDLE=0_all_combined/$OUTBASE/$NAME/eps_1_seed1
LOG=logs/s1_ens3_transcribe.log        # shared log: same experiment family

py_for() {
    case "$1" in
        qwen2_audio)                echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)             echo "$ENVROOT/flamingo3/bin/python" ;;
    esac
}

echo "=== [E-A] $NAME  victims=q2a+omni+af3  $(date -Is) ===" | tee -a "$LOG"
"$PY" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --ensemble-models "qwen2_audio,qwen25_omni,audio_flamingo" \
    --prompt-text "$PROMPT" \
    --out-base "$OUTBASE" \
    --bundle-name "$NAME" \
    --eps 1.0 --steps 1000 --seed 1 --rand-init >> "$LOG" 2>&1
N=$(ls "$BUNDLE"/audio/*attacked*.wav 2>/dev/null | wc -l)
echo "=== [E-A] $NAME done: $N/50 wavs  $(date -Is) ===" | tee -a "$LOG"

if [[ "$N" -lt 50 ]]; then
    echo "!! [E-B] SKIP $NAME -- partial bundle ($N/50); re-run phase A" | tee -a "$LOG"
    exit 1
fi

for EV in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
    ROLE="in-ensemble (white-box)"
    [[ "$EV" == "qwen25_omni_3b" ]] && ROLE="HELD-OUT (transfer)"
    echo "=== [E-B] $NAME -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
    "$(py_for "$EV")" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
        --bundle-dir "$BUNDLE" \
        --target-model "$EV" \
        --prompt-text "$PROMPT" \
        --out-tag transcribe \
        --opus-bitrates 32,64 >> "$LOG" 2>&1
done

echo "================ ENSEMBLE all3 DONE $(date -Is) ================" | tee -a "$LOG"
