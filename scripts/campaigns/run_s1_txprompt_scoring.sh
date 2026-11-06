#!/bin/bash
# Phase B ONLY for the transcribe-prompt S1 run: the attack phase finished 50/50
# for qwen2_audio and qwen25_omni on 2026-08-14, but the driver
# (run_s1_transcribe_prompt_local.sh) died in the audio_flamingo attack before it
# ever reached its own eval loop, so nothing was scored.
#
# Scores ONLY the two complete bundles. audio_flamingo (0/50, env bug) and the
# ensemble bundles (4/50) stay unscored until their attack phase is backfilled --
# HARD RULE: never score a partial cell.
#
# HARD RULE: optimization prompt == evaluation prompt. Both are the transcribe
# prompt below; cross_model_transfer_eval.py also aborts on a prompt mismatch
# against any existing JSON.
#
# Usage: bash run_s1_txprompt_scoring.sh
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
PROMPT="What does the person say in this audio?"
BUNDLES=0_all_combined/results_transcribe_prompt/01_finance_voice_agent
LOG=logs/s1_txprompt_scoring.log

py_for() {
    case "$1" in
        qwen2_audio)                echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)             echo "$ENVROOT/flamingo3/bin/python" ;;
    esac
}

mkdir -p logs
echo "================ SCORING START $(date -Is) ================" | tee -a "$LOG"

for ATK in qwen2_audio qwen25_omni; do
    B="$BUNDLES/$ATK/eps_1.0"
    N=$(ls "$B"/audio/*_attacked.wav 2>/dev/null | wc -l)
    if [[ "$N" -lt 50 ]]; then
        echo "!! SKIP $ATK -- partial bundle ($N/50 wavs)" | tee -a "$LOG"
        continue
    fi
    # white-box control first, then the three transfer evaluators
    for EV in "$ATK" qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
        OUT="$B/transfer_eval_${EV}__transcribe.json"
        [[ -f "$OUT" ]] && { echo "-- have $OUT, skipping" | tee -a "$LOG"; continue; }
        ROLE=transfer; [[ "$ATK" == "$EV" ]] && ROLE="WHITE-BOX (control)"
        echo "=== [B] $ATK -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
        "$(py_for "$EV")" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
            --bundle-dir "$B" \
            --target-model "$EV" \
            --prompt-text "$PROMPT" \
            --out-tag transcribe \
            --opus-bitrates 32,64 >> "$LOG" 2>&1
        echo "=== [B] $ATK -> $EV exit=$?  $(date -Is) ===" | tee -a "$LOG"
    done
done

echo "================ SCORING DONE $(date -Is) ================" | tee -a "$LOG"
