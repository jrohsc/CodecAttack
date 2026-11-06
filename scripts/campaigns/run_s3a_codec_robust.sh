#!/usr/bin/env bash
# S3a codec-robust sweep (extracted from chain_s1_s2_s3_codec_robust.sh).
#
# Runs the 3-cell S3a matrix sequentially on one GPU:
#   models = {qwen2_audio, qwen25_omni, audio_flamingo}
#   eps    = {1.0}
# Each cell = attack stage (multi_bitrate) + codec sidecars (mp3/aac).
# 6 stages total, logged to 0_all_combined/scripts/logs/<stage>.log.
# Exits non-zero on any stage failure. Safe to re-launch: every stage is
# idempotent (attack driver skips existing wavs, sidecar eval skips
# existing *_codecs.json).
#
# Do NOT launch while any attack/sidecar job is already using the GPU --
# these run sequentially on one GPU.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S3A_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/03a_ai_detection_bypass
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"

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

s3a_attack() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3a_${model}_eps${eps}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario 03a --target-model "$model" --eps "$eps" \
            --channel-mode multi_bitrate --skip-eval
}

s3a_codec_sidecars() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3a_${model}_eps${eps}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3A_ROOT/$model/eps_${eps}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S3A_PROMPT"
}

s3a_attack qwen2_audio 1.0 "$CA"
s3a_codec_sidecars qwen2_audio 1.0 "$CA"

s3a_attack qwen25_omni 1.0 "$OMNI"
s3a_codec_sidecars qwen25_omni 1.0 "$OMNI"

s3a_attack audio_flamingo 1.0 "$AF3"
s3a_codec_sidecars audio_flamingo 1.0 "$AF3"

echo ""
echo "######################################################"
echo "###  S3a CODEC-ROBUST COMPLETE (3 cells, eps=1.0)   ###"
echo "######################################################"
