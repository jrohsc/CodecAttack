#!/usr/bin/env bash
# Master "complete everything" chain for results_codec_robust/.
# Covers:
#   * S1 finance       (eps=1.5 omni + af3; qwen2 eps=1.5 sidecars-only)
#   * S2 interview     (eps=0.5,1.0,1.5 x 3 models -- fills gaps, idempotent)
#   * S3a ai-detect    (eps=0.5,1.0,1.5 x 3 models -- fills gaps, idempotent)
#   * S3b copyright    (eps=0.5,1.0,1.5 x 3 models -- fills gaps, idempotent)
#   * S4 physical OTA  -- SKIPPED (no run_s4_attack.py driver exists)
#
# Attack drivers skip carriers whose *_attacked.wav already exists.
# Codec sidecar eval skips clips whose *_codecs.json already exists.
# Therefore this chain can be re-run safely to resume.
#
# Serial on 1 GPU. Stage logs in 0_all_combined/scripts/logs/.
set -u

cd ${REPO_ROOT}

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python

S1_ROOT=0_all_combined/results_codec_robust/01_finance_voice_agent
S2_ROOT=0_all_combined/results_codec_robust/02_interview_screening
S3A_ROOT=0_all_combined/results_codec_robust/03a_ai_detection_bypass
S3B_ROOT=0_all_combined/results_codec_robust/03b_copyright_bypass

LOGDIR=0_all_combined/scripts/logs/chain_complete
mkdir -p "$LOGDIR"

S2_QA_PROMPT="Listen to the audio and follow the instruction."
S3A_PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)"
S3B_PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)"

STAGE_IDX=0
run_stage() {
    STAGE_IDX=$((STAGE_IDX + 1))
    local name="$1"; shift
    local log="$LOGDIR/$(printf '%02d' "$STAGE_IDX")_${name}.log"
    echo "=== [$(date +%F\ %H:%M:%S)] STAGE $STAGE_IDX: $name ==="
    echo "  log: $log"
    "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED rc=$rc -- continuing to next stage"
        echo "$name FAILED rc=$rc" >> "$LOGDIR/FAILURES.txt"
    else
        echo "  OK"
    fi
}

# Attack drivers (model-scoped helpers)
s1_attack() {  # s1_attack <model> <py> <eps>
    run_stage "s1_$1_eps$3_attack" \
        env CODEC_ATTACK_PY="$2" \
        "$2" -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
            --target-model "$1" --eps "$3"
}
s1_sidecars() {  # s1_sidecars <model> <py> <eps>
    run_stage "s1_$1_eps$3_codec_sidecars" \
        "$2" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S1_ROOT/$1/eps_$3" \
            --target-model "$1"
}

s2_attack() {  # s2_attack <model> <py> <eps>
    run_stage "s2_$1_eps$3_attack" \
        env CODEC_ATTACK_PY="$2" \
        "$2" -u 0_all_combined/scripts/run_s2_attack.py \
            --target-model "$1" --eps "$3" \
            --channel-mode multi_bitrate --skip-eval
}
s2_sidecars() {  # s2_sidecars <model> <py> <eps>
    run_stage "s2_$1_eps$3_codec_sidecars" \
        "$2" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S2_ROOT/$1/eps_$3_multibitrate" \
            --target-model "$1" \
            --prompt-text "$S2_QA_PROMPT"
}

s3_attack() {  # s3_attack <sub> <model> <py> <eps>
    run_stage "s$1_$2_eps$4_attack" \
        env CODEC_ATTACK_PY="$3" \
        "$3" -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario "$1" --target-model "$2" --eps "$4" \
            --channel-mode multi_bitrate --skip-eval
}
s3a_sidecars() {  # s3a_sidecars <model> <py> <eps>
    run_stage "s03a_$1_eps$3_codec_sidecars" \
        "$2" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3A_ROOT/$1/eps_$3_multibitrate" \
            --target-model "$1" \
            --prompt-text "$S3A_PROMPT"
}
s3b_sidecars() {  # s3b_sidecars <model> <py> <eps>
    run_stage "s03b_$1_eps$3_codec_sidecars" \
        "$2" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
            --bundle-dir "$S3B_ROOT/$1/eps_$3_multibitrate" \
            --target-model "$1" \
            --prompt-text "$S3B_PROMPT"
}

################################################################################
# Phase 0: cheap sidecar-only catchup for bundles whose attacks already finished
################################################################################
echo "###### Phase 0: sidecar-only catchup ######"
s1_sidecars qwen2_audio    "$CA"   1.5
s2_sidecars qwen2_audio    "$CA"   1.0
s2_sidecars qwen25_omni    "$OMNI" 1.0
s3b_sidecars qwen25_omni   "$OMNI" 1.0

################################################################################
# Phase 1: resume partial attacks (idempotent drivers fill only missing carriers)
################################################################################
echo "###### Phase 1: resume partial attacks + their sidecars ######"
# S3a qwen2 eps=0.5 / 1.5 partial -> finish + sidecars
s3_attack      03a qwen2_audio   "$CA"   0.5
s3a_sidecars       qwen2_audio   "$CA"   0.5
s3_attack      03a qwen2_audio   "$CA"   1.5
s3a_sidecars       qwen2_audio   "$CA"   1.5
# S3a omni eps=1.0 partial -> finish + sidecars
s3_attack      03a qwen25_omni   "$OMNI" 1.0
s3a_sidecars       qwen25_omni   "$OMNI" 1.0
# S3a af3 eps=1.0 partial -> finish + sidecars
s3_attack      03a audio_flamingo "$AF3" 1.0
s3a_sidecars       audio_flamingo "$AF3" 1.0
# S3b qwen2 eps=0.5 partial -> finish + sidecars
s3_attack      03b qwen2_audio   "$CA"   0.5
s3b_sidecars       qwen2_audio   "$CA"   0.5
# S3b af3 eps=1.0 partial -> finish + sidecars
s3_attack      03b audio_flamingo "$AF3" 1.0
s3b_sidecars       audio_flamingo "$AF3" 1.0

################################################################################
# Phase 2: S1 finance eps=1.5 (omni + af3)  [qwen2 eps=1.5 attack+sidecar already]
################################################################################
echo "###### Phase 2: S1 eps=1.5 (omni + af3) ######"
s1_attack   qwen25_omni    "$OMNI" 1.5 ; s1_sidecars qwen25_omni    "$OMNI" 1.5
s1_attack   audio_flamingo "$AF3"  1.5 ; s1_sidecars audio_flamingo "$AF3"  1.5

################################################################################
# Phase 3: S2 interview gap fill (eps=0.5 all 3, eps=1.0 af3, eps=1.5 all 3)
################################################################################
echo "###### Phase 3: S2 interview ######"
# eps=0.5 all 3 models
s2_attack qwen2_audio     "$CA"   0.5 ; s2_sidecars qwen2_audio     "$CA"   0.5
s2_attack qwen25_omni     "$OMNI" 0.5 ; s2_sidecars qwen25_omni     "$OMNI" 0.5
s2_attack audio_flamingo  "$AF3"  0.5 ; s2_sidecars audio_flamingo  "$AF3"  0.5
# eps=1.0 af3 only (qwen2 + omni attacks already done)
s2_attack audio_flamingo  "$AF3"  1.0 ; s2_sidecars audio_flamingo  "$AF3"  1.0
# eps=1.5 all 3 models
s2_attack qwen2_audio     "$CA"   1.5 ; s2_sidecars qwen2_audio     "$CA"   1.5
s2_attack qwen25_omni     "$OMNI" 1.5 ; s2_sidecars qwen25_omni     "$OMNI" 1.5
s2_attack audio_flamingo  "$AF3"  1.5 ; s2_sidecars audio_flamingo  "$AF3"  1.5

################################################################################
# Phase 4: S3a ai_detection gap fill (eps=0.5 omni+af3, eps=1.5 omni+af3)
################################################################################
echo "###### Phase 4: S3a ai_detection_bypass ######"
s3_attack 03a qwen25_omni    "$OMNI" 0.5 ; s3a_sidecars qwen25_omni    "$OMNI" 0.5
s3_attack 03a audio_flamingo "$AF3"  0.5 ; s3a_sidecars audio_flamingo "$AF3"  0.5
s3_attack 03a qwen25_omni    "$OMNI" 1.5 ; s3a_sidecars qwen25_omni    "$OMNI" 1.5
s3_attack 03a audio_flamingo "$AF3"  1.5 ; s3a_sidecars audio_flamingo "$AF3"  1.5

################################################################################
# Phase 5: S3b copyright gap fill (eps=0.5 omni+af3, eps=1.5 all 3)
################################################################################
echo "###### Phase 5: S3b copyright_bypass ######"
s3_attack 03b qwen25_omni    "$OMNI" 0.5 ; s3b_sidecars qwen25_omni    "$OMNI" 0.5
s3_attack 03b audio_flamingo "$AF3"  0.5 ; s3b_sidecars audio_flamingo "$AF3"  0.5
s3_attack 03b qwen2_audio    "$CA"   1.5 ; s3b_sidecars qwen2_audio    "$CA"   1.5
s3_attack 03b qwen25_omni    "$OMNI" 1.5 ; s3b_sidecars qwen25_omni    "$OMNI" 1.5
s3_attack 03b audio_flamingo "$AF3"  1.5 ; s3b_sidecars audio_flamingo "$AF3"  1.5

echo ""
echo "######################################################"
echo "### chain_complete_codec_robust.sh DONE            ###"
echo "### failures (if any): $LOGDIR/FAILURES.txt        ###"
echo "######################################################"
if [[ -f "$LOGDIR/FAILURES.txt" ]]; then
    echo ""
    echo "Stages that failed:"
    cat "$LOGDIR/FAILURES.txt"
fi
