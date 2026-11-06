#!/bin/bash
# Finish the preempted MP3-EoT three-model ensemble shards on the local A100.
# Shards 0/1/2 were preempted on gpu-preempt with 38/32/38 of 56 done; shard 3
# completed on SLURM. --output-dir resumes: load_completed() skips any command
# that already has qwen2_audio/<exp_key>/record.json, so this only crafts the
# 60 missing experiments. One shard at a time -- AF3 shares the single card.
set -uo pipefail

REPO=${REPO_ROOT}
LOGDIR="$REPO/0_all_combined/n50_latent_vs_waveform/logs"
cd "$REPO"

for SHARD in 0 1 2; do
    OUT="results/benchmark_ensemble3_mp3eot_transcribe_eps10.0_jazz_1_shard${SHARD}of4"
    DONE=$(find "$OUT/qwen2_audio" -name record.json 2>/dev/null | wc -l)
    echo "=== $(date '+%F %T') shard ${SHARD}: ${DONE}/56 done, starting ==="
    if (( DONE >= 56 )); then
        echo "shard ${SHARD} already complete; skipping"
        continue
    fi
    SHARD_ID=$SHARD NUM_SHARDS=4 \
        bash 0_all_combined/n50_latent_vs_waveform/run_local_ensemble3_mp3_shard.sh \
        > "$LOGDIR/ens3m_s${SHARD}_local.out" 2> "$LOGDIR/ens3m_s${SHARD}_local.err"
    RC=$?
    N=$(find "$OUT/qwen2_audio" -name record.json 2>/dev/null | wc -l)
    W=$(find "$OUT/audio" -name '*.wav' 2>/dev/null | wc -l)
    FAIL=$(grep -c "subprocess gradient requests FAILED" "$LOGDIR/ens3m_s${SHARD}_local.out")
    echo "=== $(date '+%F %T') shard ${SHARD}: rc=${RC} records=${N}/56 wavs=${W} af3_grad_failures=${FAIL} ==="
done
echo "=== $(date '+%F %T') ALL SHARDS DONE ==="
