#!/usr/bin/env bash
# eps=1.5 sweep across S3a + S3b × {qwen2_audio, qwen25_omni, audio_flamingo}.
# Serial (one GPU). Each model bundle = attack then codec sidecar eval.
# Waits for $BLOCKER_PID (current codec-attack env holder) to exit first.
#
# Stage logs: 0_all_combined/scripts/logs/<stage>.log
# Master log: 0_all_combined/scripts/chain_s3ab_eps1.5.log (via nohup)
# Idempotent: attack drivers skip existing wavs, sidecar eval skips existing *_codecs.json.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S3A_ROOT=0_all_combined/results_codec_robust/03a_ai_detection_bypass
S3B_ROOT=0_all_combined/results_codec_robust/03b_copyright_bypass
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

EPS=1.5
S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"
S3B_PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    echo "=== [$(date +%F\ %H:%M:%S)] STAGE: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- abort chain"
        exit $rc
    fi
    echo "  OK"
}

BLOCKER_PID="${BLOCKER_PID:-}"
if [[ -n "$BLOCKER_PID" ]]; then
    echo "[$(date +%F\ %H:%M:%S)] waiting for PID $BLOCKER_PID (codec-attack env holder) to finish..."
    while kill -0 "$BLOCKER_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date +%F\ %H:%M:%S)] blocker exited; starting eps=${EPS} S3a+S3b chain"
fi

################################################################################
# S3a AI-detect (40 carriers/model)
################################################################################

s3a_attack() {
    local model="$1" py="$2"
    run_stage "s3a_${model}_eps${EPS}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario 03a --target-model "$model" --eps "$EPS" \
            --channel-mode multi_bitrate --skip-eval
}
s3a_sidecars() {
    local model="$1" py="$2"
    run_stage "s3a_${model}_eps${EPS}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3A_ROOT/$model/eps_${EPS}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S3A_PROMPT"
}

s3a_attack qwen2_audio    "$CA"  ; s3a_sidecars qwen2_audio    "$CA"
s3a_attack qwen25_omni    "$OMNI"; s3a_sidecars qwen25_omni    "$OMNI"
s3a_attack audio_flamingo "$AF3" ; s3a_sidecars audio_flamingo "$AF3"

echo ""; echo "###### S3a eps=${EPS} DONE -- moving to S3b ######"; echo ""

################################################################################
# S3b copyright (20 carriers/model)
################################################################################

s3b_attack() {
    local model="$1" py="$2"
    run_stage "s3b_${model}_eps${EPS}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario 03b --target-model "$model" --eps "$EPS" \
            --channel-mode multi_bitrate --skip-eval
}
s3b_sidecars() {
    local model="$1" py="$2"
    run_stage "s3b_${model}_eps${EPS}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3B_ROOT/$model/eps_${EPS}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S3B_PROMPT"
}

s3b_attack qwen2_audio    "$CA"  ; s3b_sidecars qwen2_audio    "$CA"
s3b_attack qwen25_omni    "$OMNI"; s3b_sidecars qwen25_omni    "$OMNI"
s3b_attack audio_flamingo "$AF3" ; s3b_sidecars audio_flamingo "$AF3"

echo ""
echo "#############################################"
echo "### S3a+S3b eps=${EPS} SWEEP COMPLETE     ###"
echo "#############################################"
