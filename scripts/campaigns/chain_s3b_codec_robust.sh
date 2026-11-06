#!/usr/bin/env bash
# S3b copyright-bypass codec-robust sweep (--channel-mode multi_bitrate).
# Runs attack + MP3/AAC codec sidecars for 3 cells
# (3 models x eps=1.0) into
# 0_all_combined/results_codec_robust/03b_copyright_bypass.
#
# Prompt + carrier map come from 0_all_combined/scenarios/03b_copyright_bypass
# and 0_all_combined/data/music/03b_copyright_bypass/carrier_to_target.json
# (same 20 pairs as the baseline results/03b_copyright_bypass bundle).
#
# Logs: 0_all_combined/scripts/logs/s3b_<cell>.log
# Idempotent: skips wavs that already exist; skips codec sidecars that
# already have a *_codecs.json.
#
# Do NOT launch while any other attack/sidecar job is using the GPU --
# this runs sequentially on one GPU.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S3B_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/03b_copyright_bypass
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

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

s3b_attack() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3b_${model}_eps${eps}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario 03b --target-model "$model" --eps "$eps" \
            --channel-mode multi_bitrate --skip-eval
}

s3b_codec_sidecars() {
    local model="$1" eps="$2" py="$3"
    run_stage "s3b_${model}_eps${eps}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3B_ROOT/$model/eps_${eps}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S3B_PROMPT"
}

s3b_attack qwen2_audio 1.0 "$CA"
s3b_codec_sidecars qwen2_audio 1.0 "$CA"

s3b_attack qwen25_omni 1.0 "$OMNI"
s3b_codec_sidecars qwen25_omni 1.0 "$OMNI"

s3b_attack audio_flamingo 1.0 "$AF3"
s3b_codec_sidecars audio_flamingo 1.0 "$AF3"

echo ""
echo "######################################################"
echo "###  S3b codec-robust sweep COMPLETE                ###"
echo "######################################################"
