#!/usr/bin/env bash
# Sequential chain: remaining S1 (after current AF3 eps=1.0 attack) + full S2.
# Each stage logs to 0_all_combined/scripts/logs/<stage>.log.
# Exits non-zero on any stage failure. Designed to be launched AFTER the
# in-flight AF3 eps=1.0 attack finishes -- do not launch in parallel with it.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S1_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/01_finance_voice_agent
S2_ROOT=${REPO_ROOT}/0_all_combined/results_codec_robust/02_interview_screening
LOGDIR=0_all_combined/scripts/logs
mkdir -p "$LOGDIR"

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    echo "=== [$(date +%H:%M:%S)] STAGE: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- abort chain"
        exit $rc
    fi
    echo "  OK"
}

# ===== S1 remaining =====

# 1. AF3 eps=1.0 codec sidecars
run_stage s1_af3_eps1.0_codec_sidecars \
    "$AF3" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/audio_flamingo/eps_1.0" \
        --target-model audio_flamingo

# 2. Omni eps=0.5 attack fill (5 q3_transactions carriers)
run_stage s1_omni_eps0.5_attack \
    env CODEC_ATTACK_PY="$OMNI" \
    "$OMNI" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
        --target-model qwen25_omni --eps 0.5

# 3. Omni eps=0.5 codec sidecars (idempotent; only new clips processed)
run_stage s1_omni_eps0.5_codec_sidecars \
    "$OMNI" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$S1_ROOT/qwen25_omni/eps_0.5" \
        --target-model qwen25_omni

echo ""
echo "######################################################"
echo "###  S1 COMPLETE -- moving to S2                    ###"
echo "######################################################"
echo ""

# ===== S2 =====
# NOTE: run_s2_attack.py adds _multibitrate suffix automatically when
# --channel-mode multi_bitrate. Bundle dirs end up at eps_{X}_multibitrate/.

S2_QA_PROMPT="Listen to the audio and follow the instruction."

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

# Order: finish qwen2 (12 missing eps=0.5 + full eps=1.0), then Omni, then AF3.
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
echo "###  FULL CHAIN COMPLETE -- S1 + S2 done            ###"
echo "######################################################"
