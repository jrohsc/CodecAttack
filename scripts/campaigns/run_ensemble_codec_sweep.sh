#!/usr/bin/env bash
# Codec sweep over the ensemble-victim bundle (AC Q1 + Q5 channel story).
# Inference only on the 180 already-crafted wavs -- no new optimization.
#
# Channels: clean + Opus{64,128} + MP3{64,128} + AAC-LC{64,128} (+ DAC/EnCodec
# neural re-encode where the env supports it). AF3's flamingo3 env has no `dac`
# module, so AF3 runs the 6 ffmpeg channels only -- acceptable because neural
# re-encode is a DESTRUCTIVE DEFENSE and cannot raise an already-zero held-out
# number; it is informative on the members, where success is 90-100%.
#
# Roles: qwen2_audio + qwen25_omni = ensemble MEMBERS (co-optimized).
#        audio_flamingo + qwen25_omni_3b = HELD OUT (never in the loss).
# Omni-3B is the same family as member Omni-7B, so a null there is a stronger
# result than the cross-architecture AF3 null.
set -u

PROJ=${REPO_ROOT}
cd "$PROJ"

B=results/benchmark_ensemble_transcribe_eps10.0_jazz_1
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
EVAL=0_all_combined/scripts/ensemble_heldout_eval.py

LOSSY=opus64,opus128,mp3_64,mp3_128,aac_64,aac_128
ALL=$LOSSY,dac,encodec

run() {  # $1 python  $2 model  $3 out-name  $4 channels
    echo; echo "=== [$(date)] $2 -> $3 ==="
    # Merge stderr into stdout and FILTER it -- do not discard it. An earlier
    # version used 2>/dev/null to hide progress bars; that also swallowed an
    # argparse error, so a whole model silently produced nothing while the log
    # looked like it had simply moved on. Filter noise, keep failures.
    "$1" -u "$EVAL" --bundle-dir "$B" --target-model "$2" \
        --out-name "$3" --channels "$4" 2>&1 \
        | grep -vE "Loading weights|it/s\]|^Loading checkpoint"
    local rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && echo "!!! FAILED: $2 -> $3 (exit $rc)"
    return 0
}

# Held-out first -- that is the result that matters.
run "$AF3"  audio_flamingo   af3_heldout_allcodecs        "$LOSSY"
run "$OMNI" qwen25_omni_3b   omni3b_heldout_allcodecs     "$ALL"
# Members, for the channel-robustness picture.
run "$OMNI" qwen25_omni      omni7b_member_allcodecs      "$ALL"
run "$CA"   qwen2_audio      qwen2_member_allcodecs       "$ALL"

echo; echo "=== [$(date)] CODEC SWEEP DONE ==="
