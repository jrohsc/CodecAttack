#!/bin/bash
# Backfill the AudioFlamingo3 row of the transcribe-prompt S1 attack (0/50 on
# 2026-08-14: run_s1_attack.py launched its inner run_attack.py in the codec-attack
# env, which cannot import AudioFlamingo3ForConditionalGeneration). The fix is the
# CODEC_ATTACK_PY export below -- the same idiom every other driver in this tree uses.
#
# Waits for any in-flight scoring run to release the GPU, then attacks 50 pairs and
# scores the bundle on all four evaluators. Resumable: pairs whose attacked wav
# already exists are skipped.
#
# HARD RULE: optimization prompt == evaluation prompt (the transcribe prompt below).
#
# Usage: nohup bash run_s1_txprompt_af3_backfill.sh > logs/s1_af3_backfill.nohup 2>&1 &
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
AF3=$ENVROOT/flamingo3/bin/python
PROMPT="What does the person say in this audio?"
OUTROOT=results_transcribe_prompt/01_finance_voice_agent
B=0_all_combined/$OUTROOT/audio_flamingo/eps_1.0
LOG=logs/s1_af3_backfill.log

py_for() {
    case "$1" in
        qwen2_audio)                echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)             echo "$AF3" ;;
    esac
}

mkdir -p logs
# Don't collide with the scoring run on the same card.
while pgrep -f run_s1_txprompt_scoring.sh >/dev/null; do
    echo "waiting for scoring to finish $(date -Is)" | tee -a "$LOG"; sleep 120
done

echo "================ AF3 BACKFILL START $(date -Is) ================" | tee -a "$LOG"
env CODEC_ATTACK_PY="$AF3" "$AF3" -u 0_all_combined/scripts/run_s1_attack.py \
    --target-model audio_flamingo \
    --eps 1.0 \
    --prompt-text "$PROMPT" \
    --results-root "$OUTROOT" >> "$LOG" 2>&1
N=$(ls "$B"/audio/*_attacked.wav 2>/dev/null | wc -l)
echo "=== [A] audio_flamingo done: $N/50 wavs  $(date -Is) ===" | tee -a "$LOG"

if [[ "$N" -lt 50 ]]; then
    echo "!! SKIP scoring -- partial bundle ($N/50); never score a partial cell" | tee -a "$LOG"
    exit 1
fi

for EV in audio_flamingo qwen2_audio qwen25_omni qwen25_omni_3b; do
    OUT="$B/transfer_eval_${EV}__transcribe.json"
    [[ -f "$OUT" ]] && { echo "-- have $OUT, skipping" | tee -a "$LOG"; continue; }
    ROLE=transfer; [[ "$EV" == "audio_flamingo" ]] && ROLE="WHITE-BOX (control)"
    echo "=== [B] audio_flamingo -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
    "$(py_for "$EV")" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
        --bundle-dir "$B" \
        --target-model "$EV" \
        --prompt-text "$PROMPT" \
        --out-tag transcribe \
        --opus-bitrates 32,64 >> "$LOG" 2>&1
done
echo "================ AF3 BACKFILL DONE $(date -Is) ================" | tee -a "$LOG"
