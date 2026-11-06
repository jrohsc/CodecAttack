#!/bin/bash
# Resume + score ONE ensemble rung: ens_q2a_omni (victims Qwen2-Audio + Qwen2.5-Omni-7B),
# matched transcribe prompt. The 2026-08-14 run was killed at 4/50; run_s1_ensemble_bundleA.py
# skips any pair whose attacked wav already exists, so this picks up at pair 5.
#
# Measured cost: 12.4 min/pair -> ~9.5 h for the remaining 46 pairs, then 4 eval cells.
# The other three rungs (ens_q2a_af3, ens_af3_omni, ens_all3) are deliberately NOT run
# here; use run_s1_ensemble3_transcribe_local.sh if the ladder is wanted later.
#
# Evaluators: Q2A and Omni-7B are IN-ensemble (white-box, not transfer -- see the
# ensemble_victim lesson); Omni-3B and AF3 are held-out transfer.
#
# HARD RULE: optimization prompt == evaluation prompt (the transcribe prompt below).
#
# Usage: nohup bash run_s1_ens_q2a_omni_resume.sh > logs/s1_ens_q2a_omni_resume.nohup 2>&1 &
set -uo pipefail
cd ${REPO_ROOT}
export HF_HUB_OFFLINE=1
export ENSEMBLE_PERSISTENT_WORKER=1

ENVROOT=${WORK_ROOT}/miniconda3/envs
PY=$ENVROOT/codec-attack/bin/python
PROMPT="What does the person say in this audio?"
OUTBASE=results_transcribe_prompt_ensemble/01_finance_voice_agent
NAME=ens_q2a_omni
MODELS="qwen2_audio,qwen25_omni"
B=0_all_combined/$OUTBASE/$NAME/eps_1_seed1
LOG=logs/s1_ens_q2a_omni_resume.log

py_for() {
    case "$1" in
        qwen2_audio)                echo "$ENVROOT/codec-attack/bin/python" ;;
        qwen25_omni|qwen25_omni_3b) echo "$ENVROOT/qwen-omni/bin/python" ;;
        audio_flamingo)             echo "$ENVROOT/flamingo3/bin/python" ;;
    esac
}

mkdir -p logs
# One card: queue behind the scoring run and the AF3 backfill.
while pgrep -f 'run_s1_txprompt_scoring.sh|run_s1_txprompt_kimi_evals.sh' >/dev/null; do
    echo "waiting for GPU (scoring / af3 backfill) $(date -Is)" | tee -a "$LOG"; sleep 300
done

echo "================ ENS q2a_omni RESUME START $(date -Is) ================" | tee -a "$LOG"
echo "have $(ls "$B"/audio/*_attacked.wav 2>/dev/null | wc -l)/50 wavs before resume" | tee -a "$LOG"
"$PY" -u 0_all_combined/scripts/run_s1_ensemble_bundleA.py \
    --ensemble-models "$MODELS" \
    --prompt-text "$PROMPT" \
    --out-base "$OUTBASE" \
    --bundle-name "$NAME" \
    --eps 1.0 --steps 1000 --seed 1 --rand-init >> "$LOG" 2>&1
N=$(ls "$B"/audio/*_attacked.wav 2>/dev/null | wc -l)
echo "=== [E-A] $NAME done: $N/50 wavs  $(date -Is) ===" | tee -a "$LOG"

if [[ "$N" -lt 50 ]]; then
    echo "!! SKIP scoring -- partial bundle ($N/50); never score a partial cell. Re-run to resume." | tee -a "$LOG"
    exit 1
fi

# kimi_audio is the FOURTH ARCHITECTURE and can never be an ensemble member
# (inference-only wrapper, no differentiable path), so it is the cleanest held-out
# victim in the set. Measured ceiling under this prompt: 10/10 clean, 10/10 opus32,
# 9/10 opus64 STRICT -- higher than any other evaluator, so a 0 there needs no
# normalization caveat. It runs through kimi_eval.sh, which carries three env fixes
# its conda env needs (see that script).
for EV in qwen2_audio qwen25_omni qwen25_omni_3b audio_flamingo kimi_audio; do
    OUT="$B/transfer_eval_${EV}__transcribe.json"
    [[ -f "$OUT" ]] && { echo "-- have $OUT, skipping" | tee -a "$LOG"; continue; }
    ROLE="HELD-OUT (transfer)"
    [[ ",$MODELS," == *",$EV,"* ]] && ROLE="in-ensemble (white-box)"
    echo "=== [E-B] $NAME -> $EV [$ROLE]  $(date -Is) ===" | tee -a "$LOG"
    if [[ "$EV" == "kimi_audio" ]]; then
        bash kimi_eval.sh \
            --bundle-dir "$B" \
            --prompt-text "$PROMPT" \
            --out-tag transcribe \
            --opus-bitrates 32,64 >> "$LOG" 2>&1
    else
        "$(py_for "$EV")" -u 0_all_combined/scripts/cross_model_transfer_eval.py \
            --bundle-dir "$B" \
            --target-model "$EV" \
            --prompt-text "$PROMPT" \
            --out-tag transcribe \
            --opus-bitrates 32,64 >> "$LOG" 2>&1
    fi
done
echo "================ ENS q2a_omni DONE $(date -Is) ================" | tee -a "$LOG"
