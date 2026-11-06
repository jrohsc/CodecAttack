#!/usr/bin/env bash
# Cross-model transfer eval queue for S1 + S3a at eps=1.0.
#
# For each (attacker x evaluator) cross pair (3 attackers x 2 evaluators each
# = 6 pairs per scenario, 12 evals total), run cross_model_transfer_eval.py.
# Output: transfer_eval_<evaluator>.json sibling-written to each attacker
# bundle dir.
#
# Waits for the queued sidecar fix script (PID arg) to release the GPU first.
#
# Usage:
#   bash queue_cross_model_eval.sh                 # default wait list
#   bash queue_cross_model_eval.sh --skip-wait     # start immediately
set -u

SKIP_WAIT=0
[[ "${1:-}" == "--skip-wait" ]] && SKIP_WAIT=1

PROJ=${REPO_ROOT}
cd "$PROJ"

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

py_for() {
    case "$1" in
        qwen2_audio)    echo "$CA" ;;
        qwen25_omni)    echo "$OMNI" ;;
        audio_flamingo) echo "$AF3" ;;
    esac
}

S1_BUNDLE_ROOT=$PROJ/0_all_combined/results_codec_robust/01_finance_voice_agent
S3A_BUNDLE_ROOT=$PROJ/0_all_combined/results_codec_robust/03a_ai_detection_bypass

S1_PROMPT=""  # cross_model_transfer_eval.py defaults to BANK_PROMPT for S1
S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"

WATCH_PIDS=(2800379)

if [[ $SKIP_WAIT -eq 0 ]]; then
    echo "=== [$(date)] Cross-model queue waiting for PIDs: ${WATCH_PIDS[*]} ==="
    for pid in "${WATCH_PIDS[@]}"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 60
        done
        echo "  PID $pid done at $(date)"
    done
    sleep 30
fi

run_one() {
    # $1 attacker, $2 evaluator, $3 bundle_dir, $4 prompt_text (may be empty)
    local atk="$1" ev="$2" bundle="$3" prompt="$4"
    local py; py="$(py_for "$ev")"
    echo
    echo "######################################################"
    echo "###  attacker=$atk  evaluator=$ev"
    echo "###  bundle=$bundle"
    echo "######################################################"
    local args=(--bundle-dir "$bundle" --target-model "$ev")
    [[ -n "$prompt" ]] && args+=(--prompt-text "$prompt")
    "$py" -u 0_all_combined/scripts/cross_model_transfer_eval.py "${args[@]}"
    echo "[$(date)] done: $atk -> $ev"
}

echo
echo "######################################################"
echo "###  S1 finance (eps=1.0) cross-model x6           ###"
echo "######################################################"
for ATK in qwen2_audio qwen25_omni audio_flamingo; do
    BUNDLE="$S1_BUNDLE_ROOT/$ATK/eps_1.0"
    [[ -d "$BUNDLE/audio" ]] || { echo "skip (no bundle): $BUNDLE"; continue; }
    for EV in qwen2_audio qwen25_omni audio_flamingo; do
        [[ "$EV" == "$ATK" ]] && continue
        run_one "$ATK" "$EV" "$BUNDLE" "$S1_PROMPT"
    done
done

echo
echo "######################################################"
echo "###  S3a ai-detect (eps=1.0) cross-model x6        ###"
echo "######################################################"
for ATK in qwen2_audio qwen25_omni audio_flamingo; do
    BUNDLE="$S3A_BUNDLE_ROOT/$ATK/eps_1.0_multibitrate"
    [[ -d "$BUNDLE/audio" ]] || { echo "skip (no bundle): $BUNDLE"; continue; }
    for EV in qwen2_audio qwen25_omni audio_flamingo; do
        [[ "$EV" == "$ATK" ]] && continue
        run_one "$ATK" "$EV" "$BUNDLE" "$S3A_PROMPT"
    done
done

echo
echo "######################################################"
echo "###  CROSS-MODEL EVAL COMPLETE                     ###"
echo "###  Re-run to fill the table:                     ###"
echo "###  python 0_all_combined/scripts/build_cross_model_table.py"
echo "######################################################"
