#!/usr/bin/env bash
# Master chain: finish S1 (remaining cells) + full S2 + full S3a + full S3b
# codec-robust sweeps (--channel-mode multi_bitrate) at eps={0.5, 1.0}.
# Codec sidecars (mp3/aac) run after each attack bundle completes.
#
# Each stage logs to 0_all_combined/scripts/logs/<stage>.log.
# Exits non-zero on any stage failure. Safe to re-launch: every stage is
# idempotent (attack drivers skip existing wavs, sidecar eval skips
# existing *_codecs.json).
#
# Do NOT launch while any attack/sidecar job is already using the GPU
# -- these run sequentially on one GPU.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S1_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/01_finance_voice_agent
S2_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/02_interview_screening
S3A_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/03a_ai_detection_bypass
S3B_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/03b_copyright_bypass
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

S2_QA_PROMPT="Listen to the audio and follow the instruction."
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

################################################################################
# S1 finance -- finish remaining cells
################################################################################

# 1a. AF3 eps=1.0 attack fill (8 carriers missing as of 2026-04-21)
run_stage s1_af3_eps1.0_attack_fill \
    env CODEC_ATTACK_PY="$AF3" \
    "$AF3" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model audio_flamingo --eps 1.0

# 1b. AF3 eps=1.0 codec sidecars
run_stage s1_af3_eps1.0_codec_sidecars \
    "$AF3" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/audio_flamingo/eps_1.0" \
        --target-model audio_flamingo

# 1c. Omni eps=0.5 attack fill (5 q3_transactions carriers)
run_stage s1_omni_eps0.5_attack_fill \
    env CODEC_ATTACK_PY="$OMNI" \
    "$OMNI" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model qwen25_omni --eps 0.5

# 1d. Omni eps=0.5 codec sidecars (idempotent; only new clips processed)
run_stage s1_omni_eps0.5_codec_sidecars \
    "$OMNI" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/qwen25_omni/eps_0.5" \
        --target-model qwen25_omni

echo ""
echo "######################################################"
echo "###  S1 COMPLETE -- moving to S2                    ###"
echo "######################################################"
echo ""

################################################################################
# S2 interview -- all 6 cells
################################################################################

s2_attack() {
    local model="$1" eps="$2" py="$3"
    run_stage "s2_${model}_eps${eps}_attack" \
        env CODEC_ATTACK_PY="$py" \
        "$py" -u 0_all_combined/scripts/run_s2_attack.py \
            --target-model "$model" --eps "$eps" \
            --channel-mode multi_bitrate --skip-eval
}

s2_codec_sidecars() {
    local model="$1" eps="$2" py="$3"
    run_stage "s2_${model}_eps${eps}_codec_sidecars" \
        "$py" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S2_ROOT/$model/eps_${eps}_multibitrate" \
            --target-model "$model" \
            --prompt-text "$S2_QA_PROMPT"
}

s2_attack qwen2_audio 0.5 "$CA"
s2_codec_sidecars qwen2_audio 0.5 "$CA"
s2_attack qwen2_audio 1.0 "$CA"
s2_codec_sidecars qwen2_audio 1.0 "$CA"

s2_attack qwen25_omni 0.5 "$OMNI"
s2_codec_sidecars qwen25_omni 0.5 "$OMNI"
s2_attack qwen25_omni 1.0 "$OMNI"
s2_codec_sidecars qwen25_omni 1.0 "$OMNI"

s2_attack audio_flamingo 0.5 "$AF3"
s2_codec_sidecars audio_flamingo 0.5 "$AF3"
s2_attack audio_flamingo 1.0 "$AF3"
s2_codec_sidecars audio_flamingo 1.0 "$AF3"

echo ""
echo "######################################################"
echo "###  S2 COMPLETE -- moving to S3a                   ###"
echo "######################################################"
echo ""

################################################################################
# S3a AI-detection bypass -- all 6 cells (40 carriers each)
################################################################################

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

s3a_attack qwen2_audio 0.5 "$CA"
s3a_codec_sidecars qwen2_audio 0.5 "$CA"
s3a_attack qwen2_audio 1.0 "$CA"
s3a_codec_sidecars qwen2_audio 1.0 "$CA"

s3a_attack qwen25_omni 0.5 "$OMNI"
s3a_codec_sidecars qwen25_omni 0.5 "$OMNI"
s3a_attack qwen25_omni 1.0 "$OMNI"
s3a_codec_sidecars qwen25_omni 1.0 "$OMNI"

s3a_attack audio_flamingo 0.5 "$AF3"
s3a_codec_sidecars audio_flamingo 0.5 "$AF3"
s3a_attack audio_flamingo 1.0 "$AF3"
s3a_codec_sidecars audio_flamingo 1.0 "$AF3"

echo ""
echo "######################################################"
echo "###  S3a COMPLETE -- moving to S3b                  ###"
echo "######################################################"
echo ""

################################################################################
# S3b copyright bypass -- all 6 cells (20 carriers each)
################################################################################

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

s3b_attack qwen2_audio 0.5 "$CA"
s3b_codec_sidecars qwen2_audio 0.5 "$CA"
s3b_attack qwen2_audio 1.0 "$CA"
s3b_codec_sidecars qwen2_audio 1.0 "$CA"

s3b_attack qwen25_omni 0.5 "$OMNI"
s3b_codec_sidecars qwen25_omni 0.5 "$OMNI"
s3b_attack qwen25_omni 1.0 "$OMNI"
s3b_codec_sidecars qwen25_omni 1.0 "$OMNI"

s3b_attack audio_flamingo 0.5 "$AF3"
s3b_codec_sidecars audio_flamingo 0.5 "$AF3"
s3b_attack audio_flamingo 1.0 "$AF3"
s3b_codec_sidecars audio_flamingo 1.0 "$AF3"

echo ""
echo "######################################################"
echo "###  FULL CHAIN COMPLETE: S1 + S2 + S3a + S3b       ###"
echo "######################################################"
