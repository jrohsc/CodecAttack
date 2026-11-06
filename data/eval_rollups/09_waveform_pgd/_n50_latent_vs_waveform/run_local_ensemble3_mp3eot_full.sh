#!/bin/bash
# E2 completion, local A100: finish crafting the MP3-EoT 3-model ensemble shards
# and score every crafted wav on all 3 members + held-out Omni-3B across
# clean + Opus/MP3/AAC {64,128} -- the same 7-channel readout E1 (no-EoT
# ensemble3) has, under the same out-names.
#
# Scoring runs on a symlinked MERGED bundle so each model loads once per pass;
# the eval is resumable per clip, so each pass only scores newly crafted wavs.
#
# Order: score the 89 surviving wavs first (early numbers), then craft
# shard1 -> shard2 -> shard0, re-scoring after each.  Create
#   $N50/SKIP_LOCAL_GEN<k>
# before a shard starts to hand that shard to another producer (e.g. SLURM);
# never let two producers write the same shard.
#
# DO NOT EDIT WHILE RUNNING (bash reads scripts incrementally).
set -u

REPO=${REPO_ROOT}
cd "$REPO"
N50=0_all_combined/n50_latent_vs_waveform
EVAL=0_all_combined/scripts/ensemble_heldout_eval.py
ENVS=${WORK_ROOT}/miniconda3/envs
CA=$ENVS/codec-attack/bin/python
OMNI=$ENVS/qwen-omni/bin/python
AF3=$ENVS/flamingo3/bin/python
CH=opus64,opus128,mp3_64,mp3_128,aac_64,aac_128

STEM=results/benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1
MERGED=${STEM}_MERGED

export CUDA_VISIBLE_DEVICES=0
export ENSEMBLE_WORKER_GPU=0
NOISE="WARNING:root:System prompt modified|Loading weights|it/s\]|s/it\]|Loading checkpoint"

link_merged() {
    mkdir -p "$MERGED/audio" "$MERGED/qwen2_audio"
    cp -f "${STEM}_shard3of4/config.json" "$MERGED/config.json"
    for s in 0 1 2 3; do
        d=$REPO/${STEM}_shard${s}of4
        for w in "$d"/audio/*.wav; do
            [ -e "$w" ] || continue
            stem=$(basename "$w" .wav)
            # only link clips whose primary record exists (target_text source)
            [ -f "$d/qwen2_audio/$stem/record.json" ] || continue
            ln -sfn "$w" "$MERGED/audio/$stem.wav"
            ln -sfn "$d/qwen2_audio/$stem" "$MERGED/qwen2_audio/$stem"
        done
    done
    echo "[merge] $(ls "$MERGED/audio" | wc -l) wavs linked"
}

ev() {  # $1 python  $2 model  $3 out-name
    echo "=== [$(date -u +%FT%TZ)] EVAL $2 -> $3"
    "$1" -u "$EVAL" --bundle-dir "$MERGED" --target-model "$2" \
        --out-name "$3" --channels "$CH" 2>&1 | grep -vE "$NOISE"
    echo "=== [$(date -u +%FT%TZ)] EVAL $2 rc=${PIPESTATUS[0]}"
}

eval_all() {
    link_merged
    ev "$CA"   qwen2_audio    qwen2_MEMBER_allcodecs
    ev "$OMNI" qwen25_omni    omni7b_MEMBER
    ev "$AF3"  audio_flamingo af3_MEMBER
    ev "$OMNI" qwen25_omni_3b omni3b_HELDOUT
}

gen() {  # $1 shard id
    if [ -f "$N50/SKIP_LOCAL_GEN$1" ]; then
        echo "=== [$(date -u +%FT%TZ)] GEN shard$1 SKIPPED (skip-file present)"; return 0
    fi
    echo "=== [$(date -u +%FT%TZ)] GEN shard$1 start"
    SHARD_ID=$1 NUM_SHARDS=4 bash "$N50/run_local_ensemble3_mp3_shard.sh" 2>&1 | grep -vE "$NOISE"
    echo "=== [$(date -u +%FT%TZ)] GEN shard$1 rc=${PIPESTATUS[0]} wavs=$(ls ${STEM}_shard$1of4/audio 2>/dev/null | wc -l)/56"
}

echo "=== [$(date -u +%FT%TZ)] E2 local chain start on $(hostname)"
eval_all
for s in 1 2 0; do
    gen $s
    eval_all
done
echo "=== [$(date -u +%FT%TZ)] E2 local chain DONE"
