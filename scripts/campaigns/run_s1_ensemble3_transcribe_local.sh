#!/bin/bash
# Three pairwise ENSEMBLE surrogates, all under the MATCHED transcribe prompt.
#
# Design -- every model is in-ensemble in two combos and held-out in one, so the
# same victim can be compared against itself across conditions. Omni-3B is never
# trained on, giving a clean same-family probe in all three rows:
#
#   combo               in-ensemble (white-box)     held-out (TRANSFER)
#   Q2A + Omni-7B       Q2A, Omni-7B                Omni-3B, AF3
#   Q2A + AF3           Q2A, AF3                    Omni-7B, Omni-3B
#   AF3 + Omni-7B       AF3, Omni-7B                Q2A, Omni-3B
#
# The published ensemble arm (0/3900 held-out) ran entirely under the bank prompt,
# where clean speech of the target itself scores only ~7% -- so a partially
# transferring perturbation would have been floored to 0 regardless. That run never
# got a fair test. Here optimization AND evaluation both use the transcribe prompt,
# whose ceiling is ~98.5%, so a held-out zero would be a real null.
#
# HARD RULE: never optimize under one prompt and score under another. Both phases
# below pass the identical PROMPT string.
#
# All three combos run from the codec-attack env: Q2A and Omni-7B are in-process,
# AF3 is dispatched to the flamingo3 env via gradient_worker (ensemble_attack.py
# filters ENSEMBLE_IN_PROCESS / ENSEMBLE_SUBPROCESS by membership automatically).
#
# Resumable: a pair whose attacked wav exists is skipped.
#
# Usage: bash run_s1_ensemble3_transcribe_local.sh

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python
PROMPT="What does the person say in this audio?"
OUTBASE=results_transcribe_prompt_ensemble/01_finance_voice_agent
BUNDLES=0_all_combined/$OUTBASE
LOG=logs/s1_ens3_transcribe.log

COMBOS=("qwen2_audio,qwen25_omni"  "qwen2_audio,audio_flamingo"  "audio_flamingo,qwen25_omni")
NAMES=( "ens_q2a_omni"             "ens_q2a_af3"                 "ens_af3_omni")

mkdir -p logs
echo "================ ENSEMBLE x3 START $(date -Is) ================" | tee -a "$LOG"

py_for() {
    case "$1" in
        qwen2_audio)                echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)             echo "$ENVROOT/flamingo3/bin/python" ;;
    esac
}

# ---------------- Phase A: co-optimize each pair ----------------
for i in 0 1 2; do
    MODELS=${COMBOS[$i]}
    NAME=${NAMES[$i]}
    echo "=== [E-A] $NAME  victims=$MODELS  $(date -Is) ===" | tee -a "$LOG"
    "$PY" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
        --ensemble-models "$MODELS" \
        --prompt-text "$PROMPT" \
        --out-base "$OUTBASE" \
        --bundle-name "$NAME" \
        --eps 1.0 --steps 1000 --seed 1 --rand-init >> "$LOG" 2>&1
    N=$(ls "$BUNDLES/$NAME/eps_1_seed1"/audio/*attacked*.wav 2>/dev/null | wc -l)
    echo "=== [E-A] $NAME done: $N/50 wavs  $(date -Is) ===" | tee -a "$LOG"
done

# ---------------- Phase B: evaluate every combo on all four models ----------------
for i in 0 1 2; do
    NAME=${NAMES[$i]}
    MODELS=${COMBOS[$i]}
    B="$BUNDLES/$NAME/eps_1_seed1"
    N=$(ls "$B"/audio/*attacked*.wav 2>/dev/null | wc -l)
    if [[ "$N" -lt 50 ]]; then
        echo "!! [E-B] SKIP $NAME -- partial bundle ($N/50); re-run phase A" | tee -a "$LOG"
        continue
    fi
    for EV in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
        ROLE="HELD-OUT (transfer)"
        [[ ",$MODELS," == *",$EV,"* ]] && ROLE="in-ensemble (white-box)"
        echo "=== [E-B] $NAME -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
        "$(py_for "$EV")" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
            --bundle-dir "$B" \
            --target-model "$EV" \
            --prompt-text "$PROMPT" \
            --out-tag transcribe \
            --opus-bitrates 32,64 >> "$LOG" 2>&1
    done
done

echo "================ ENSEMBLE x3 DONE $(date -Is) ================" | tee -a "$LOG"
