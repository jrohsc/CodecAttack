#!/usr/bin/env bash
# Interactive single-cell launcher: pick (scenario, model, eps) and run
# attack + codec sidecar eval in the foreground. Skip-existing resume is
# handled by the underlying drivers (adv wavs and *_codecs.json skipped
# if already present).
#
# Usage: bash 0_all_combined/scripts/run_one.sh
set -u

cd ${REPO_ROOT}

# Format a command array for display, truncating --prompt-text content.
format_cmd() {
    local args=("$@")
    local out=""
    local i=0
    while (( i < ${#args[@]} )); do
        if [[ "${args[$i]}" == "--prompt-text" && $((i+1)) -lt ${#args[@]} ]]; then
            out+=" --prompt-text <${#args[$((i+1))]}-char prompt>"
            i=$((i+2))
        else
            out+=" ${args[$i]}"
            i=$((i+1))
        fi
    done
    echo "${out# }"
}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S2_QA_PROMPT="Listen to the audio and follow the instruction."
S3A_PROMPT_FILE=0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt
S3B_PROMPT_FILE=0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt

########################################
# Scenario menu
########################################
echo "Select scenario:"
echo "  1  -- S1  finance voice agent  (25 EN banking pairs)"
echo "  2  -- S2  interview screening  (25 EN / 24 ZH pairs)"
echo "  3a -- S3a AI-detection bypass  (40 pairs)"
echo "  3b -- S3b copyright bypass     (45 pairs)"
read -r -p "> " SCEN_IN

# S2_LANG is set in the S2 branch below (en|zh). Other scenarios leave it empty.
S2_LANG=""
case "$SCEN_IN" in
    1)  SCEN=s1;  SCEN_LABEL="S1 finance voice agent (EN carriers)";  SCEN_DIR=01_finance_voice_agent; IS_MBR=0; TOTAL=25 ;;
    2)  SCEN=s2;  SCEN_DIR=02_interview_screening; IS_MBR=1
        echo ""
        echo "S2 carrier language:"
        echo "  en -- English Edge-TTS interview (25 pairs, primary) [default]"
        echo "  zh -- Mandarin ablation (24 pairs, legacy)"
        read -r -p "> " S2_LANG
        S2_LANG="${S2_LANG:-en}"
        case "$S2_LANG" in
            en) SCEN_LABEL="S2 interview screening (EN carriers)"; TOTAL=25 ;;
            zh) SCEN_LABEL="S2 interview screening (ZH ablation)"; TOTAL=24 ;;
            *)  echo "Unknown S2 lang '$S2_LANG' (expected en or zh)"; exit 1 ;;
        esac ;;
    3a) SCEN=s3a; SCEN_LABEL="S3a AI-detection bypass"; SCEN_DIR=03a_ai_detection_bypass; IS_MBR=1; TOTAL=40 ;;
    3b) SCEN=s3b; SCEN_LABEL="S3b copyright bypass";    SCEN_DIR=03b_copyright_bypass;   IS_MBR=1; TOTAL=45 ;;
    *)  echo "Unknown scenario '$SCEN_IN' (expected 1, 2, 3a, or 3b)"; exit 1 ;;
esac

########################################
# Model menu
########################################
echo ""
echo "Select model:"
echo "  1 -- qwen2_audio     (codec-attack env)"
echo "  2 -- qwen25_omni     (qwen-omni env)"
echo "  3 -- audio_flamingo  (flamingo3 env)"
read -r -p "> " MODEL_IN

case "$MODEL_IN" in
    1) MODEL=qwen2_audio;    PY="$CA";   ENV_NAME="codec-attack" ;;
    2) MODEL=qwen25_omni;    PY="$OMNI"; ENV_NAME="qwen-omni" ;;
    3) MODEL=audio_flamingo; PY="$AF3";  ENV_NAME="flamingo3" ;;
    *) echo "Unknown model '$MODEL_IN' (expected 1, 2, or 3)"; exit 1 ;;
esac

########################################
# Eps
########################################
echo ""
read -r -p "Eps (e.g. 0.5 / 1.0 / 1.5): " EPS
if ! [[ "$EPS" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "Invalid eps '$EPS' (must be a positive number)"; exit 1
fi

########################################
# Resolve bundle + counts
########################################
if (( IS_MBR )); then
    BUNDLE_SUFFIX="eps_${EPS}_multibitrate"
else
    BUNDLE_SUFFIX="eps_${EPS}"
fi
# S2 EN carriers live in a parallel bundle dir so they don't collide with the
# legacy ZH results on disk. ZH keeps the historical path (no suffix).
if [[ "$SCEN" == "s2" && "$S2_LANG" == "en" ]]; then
    BUNDLE_SUFFIX="${BUNDLE_SUFFIX}_en"
fi
BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/${BUNDLE_SUFFIX}"

declare -a PAIR_NAMES=()
declare -a PAIR_HAS_SIDECAR=()
WAV_DIR="$BUNDLE/audio"
if [[ -d "$WAV_DIR" ]]; then
    while IFS= read -r wav; do
        base="${wav##*/}"
        base="${base%_attacked.wav}"
        PAIR_NAMES+=("$base")
        if [[ -f "$WAV_DIR/${base}_codecs.json" ]]; then
            PAIR_HAS_SIDECAR+=(1)
        else
            PAIR_HAS_SIDECAR+=(0)
        fi
    done < <(find "$WAV_DIR" -maxdepth 1 -name '*_attacked.wav' 2>/dev/null | sort)
fi
WAV_DONE=${#PAIR_NAMES[@]}
SIDECARS_DONE=0
if (( WAV_DONE > 0 )); then
    for s in "${PAIR_HAS_SIDECAR[@]}"; do (( s == 1 )) && SIDECARS_DONE=$((SIDECARS_DONE + 1)); done
fi
WAV_REMAIN=$(( TOTAL - WAV_DONE )); (( WAV_REMAIN < 0 )) && WAV_REMAIN=0
# Sidecars still to run = existing wavs missing sidecars + new wavs from attack stage
SIDECAR_REMAIN_EXISTING=$(( WAV_DONE - SIDECARS_DONE ))
(( SIDECAR_REMAIN_EXISTING < 0 )) && SIDECAR_REMAIN_EXISTING=0
SIDECAR_REMAIN_TOTAL=$(( SIDECAR_REMAIN_EXISTING + WAV_REMAIN ))

########################################
# Build commands
########################################
case "$SCEN" in
    s1)
        ATTACK_CMD=("$PY" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
                     --target-model "$MODEL" --eps "$EPS")
        SIDECAR_CMD=("$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
                     --bundle-dir "$BUNDLE" --target-model "$MODEL")
        ;;
    s2)
        ATTACK_CMD=("$PY" -u 0_all_combined/scripts/run_s2_attack.py \
                     --target-model "$MODEL" --eps "$EPS" \
                     --channel-mode multi_bitrate --skip-eval \
                     --lang "$S2_LANG")
        SIDECAR_CMD=("$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
                     --bundle-dir "$BUNDLE" --target-model "$MODEL" \
                     --prompt-text "$S2_QA_PROMPT")
        ;;
    s3a)
        if [[ ! -f "$S3A_PROMPT_FILE" ]]; then
            echo "Missing prompt file: $S3A_PROMPT_FILE"; exit 1
        fi
        S3A_PROMPT="$(cat "$S3A_PROMPT_FILE")"
        ATTACK_CMD=("$PY" -u 0_all_combined/scripts/run_s3_attack.py \
                     --sub-scenario 03a --target-model "$MODEL" --eps "$EPS" \
                     --channel-mode multi_bitrate --skip-eval)
        SIDECAR_CMD=("$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
                     --bundle-dir "$BUNDLE" --target-model "$MODEL" \
                     --prompt-text "$S3A_PROMPT")
        ;;
    s3b)
        if [[ ! -f "$S3B_PROMPT_FILE" ]]; then
            echo "Missing prompt file: $S3B_PROMPT_FILE"; exit 1
        fi
        S3B_PROMPT="$(cat "$S3B_PROMPT_FILE")"
        ATTACK_CMD=("$PY" -u 0_all_combined/scripts/run_s3_attack.py \
                     --sub-scenario 03b --target-model "$MODEL" --eps "$EPS" \
                     --channel-mode multi_bitrate --skip-eval)
        SIDECAR_CMD=("$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
                     --bundle-dir "$BUNDLE" --target-model "$MODEL" \
                     --prompt-text "$S3B_PROMPT")
        ;;
esac

########################################
# Pre-flight overview
########################################
echo ""
echo "======================================================"
echo " Scenario : $SCEN_LABEL"
echo " Model    : $MODEL  ($ENV_NAME env)"
echo " Eps      : $EPS"
echo " Bundle   : $BUNDLE"
echo "======================================================"
echo ""
echo "-- Per-pair status --"
echo "   legend: [train ✓/✗] [eval ✓/✗]  pair_basename"
if (( WAV_DONE > 0 )); then
    for i in "${!PAIR_NAMES[@]}"; do
        s="${PAIR_HAS_SIDECAR[$i]}"
        if (( s == 1 )); then eval_mark="✓"; else eval_mark="✗"; fi
        echo "   [train ✓] [eval $eval_mark]  ${PAIR_NAMES[$i]}"
    done
fi
if (( WAV_REMAIN > 0 )); then
    echo "   [train ✗] [eval ✗]  ($WAV_REMAIN pair(s) not yet generated -- names chosen by driver)"
fi
echo ""
echo "-- Plan --"
echo "   Training (attack) :  $WAV_DONE / $TOTAL done  ->  $WAV_REMAIN pair(s) to run"
echo "   Eval (sidecar)    :  $SIDECARS_DONE / $TOTAL done  ->  $SIDECAR_REMAIN_TOTAL pair(s) to run"
if (( SIDECAR_REMAIN_EXISTING > 0 && WAV_REMAIN > 0 )); then
    echo "                        ($SIDECAR_REMAIN_EXISTING for existing wavs + $WAV_REMAIN for new wavs after training)"
fi
echo ""
echo "-- Commands --"
echo "   attack  : $(format_cmd "${ATTACK_CMD[@]}")"
echo "             (env CODEC_ATTACK_PY=$PY)"
echo "   sidecar : $(format_cmd "${SIDECAR_CMD[@]}")"
echo ""
if (( WAV_REMAIN == 0 && SIDECAR_REMAIN_TOTAL == 0 )); then
    echo "Nothing to run -- both stages fully populated. (Re-running is a no-op.)"
fi
read -r -p "Proceed? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

########################################
# Run attack stage
########################################
echo ""
echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE START ==="
env CODEC_ATTACK_PY="$PY" "${ATTACK_CMD[@]}"
RC=$?
if (( RC != 0 )); then
    echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE FAILED rc=$RC -- skipping sidecars ==="
    exit $RC
fi
echo "=== [$(date +%F\ %H:%M:%S)] ATTACK STAGE OK ==="

########################################
# Run sidecar stage
########################################
echo ""
echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE START ==="
"${SIDECAR_CMD[@]}"
RC=$?
if (( RC != 0 )); then
    echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE FAILED rc=$RC ==="
    exit $RC
fi
echo "=== [$(date +%F\ %H:%M:%S)] SIDECAR STAGE OK ==="

echo ""
echo "######################################################"
echo "### DONE: $SCEN_LABEL / $MODEL / eps=$EPS"
echo "######################################################"
