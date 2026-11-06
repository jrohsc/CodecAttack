#!/bin/bash
# EXPERIMENT #1, run entirely on the local GPU (no SLURM).
#
# Phase A: re-optimize the S1 attack with the TRANSCRIBE prompt in context,
#          3 models x 50 carrier/target pairs, eps=1.0, 1000 PGD steps.
# Phase B: evaluate every bundle against all 4 evaluators under that same prompt.
#
# Why: the published bundles were optimized with the bank prompt, where the
# transfer ceiling (clean speech of the target) is only ~7% -- so every transfer
# zero there is floored by the prompt, not by transfer. Under the transcribe
# prompt the ceiling is 98.5%, but the bank-prompt-optimized wavs score ~0
# white-box there. This run puts optimization and evaluation on the SAME prompt
# so the diagonal is high AND the ceiling has headroom.
#
# Resumable: run_s1_attack.py SKIPs any pair whose attacked wav already exists,
# so re-running this script after an interruption picks up where it stopped.
#
# Usage:  bash run_s1_transcribe_prompt_local.sh

set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
PROMPT="What does the person say in this audio?"
OUTROOT=results_transcribe_prompt/01_finance_voice_agent
BUNDLES=0_all_combined/$OUTROOT
LOG=logs/s1_txprompt_local.log

mkdir -p logs
echo "================ START $(date -Is) ================" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

py_for() {
    case "$1" in
        qwen2_audio)                  echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b)   echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)               echo "$ENVROOT/flamingo3/bin/python" ;;
    esac
}

# ---------------- Phase A: attack ----------------
for MODEL in qwen2_audio qwen25_omni audio_flamingo; do
    PY=$(py_for "$MODEL")
    echo "=== [A] attack @ transcribe prompt: $MODEL  $(date -Is) ===" | tee -a "$LOG"
    # run_s1_attack.py re-launches dataset/interview/run_attack.py as a subprocess
    # with CODEC_ATTACK_PY (default: the codec-attack env). Without this env var the
    # inner process runs in codec-attack regardless of $PY, which import-errors on
    # AudioFlamingo3ForConditionalGeneration and killed the AF3 row on 2026-08-14.
    env CODEC_ATTACK_PY="$PY" "$PY" -u 0_all_combined/scripts/run_s1_attack.py \
        --target-model "$MODEL" \
        --eps 1.0 \
        --prompt-text "$PROMPT" \
        --results-root "$OUTROOT" >> "$LOG" 2>&1
    N=$(ls "$BUNDLES/$MODEL/eps_1.0/audio"/*attacked*.wav 2>/dev/null | wc -l)
    echo "=== [A] $MODEL done: $N/50 wavs  $(date -Is) ===" | tee -a "$LOG"
done

# ---------------- Phase B: eval ----------------
for ATK in qwen2_audio qwen25_omni audio_flamingo; do
    B="$BUNDLES/$ATK/eps_1.0"
    N=$(ls "$B/audio"/*attacked*.wav 2>/dev/null | wc -l)
    if [[ "$N" -lt 50 ]]; then
        echo "!! [B] SKIP $ATK -- partial bundle ($N/50 wavs); re-run phase A" | tee -a "$LOG"
        continue
    fi
    for EV in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo; do
        PY=$(py_for "$EV")
        ROLE=transfer; [[ "$ATK" == "$EV" ]] && ROLE="WHITE-BOX (control)"
        echo "=== [B] $ATK -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
        "$PY" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
            --bundle-dir "$B" \
            --target-model "$EV" \
            --prompt-text "$PROMPT" \
            --out-tag transcribe \
            --opus-bitrates 32,64 >> "$LOG" 2>&1
    done
done

echo "================ ALL DONE $(date -Is) ================" | tee -a "$LOG"
