#!/usr/bin/env bash
# Finish the AF3 S3a waveform cell on the local GPU.
#
# What is left after run_s3a_local_chain.sh:
#   seed 42  40/40 + sidecars + neural eval        -- done
#   seed 7   40 wavs but 39 summaries / 39 codecs  -- one clip short
#   seed 1   14/40, no sidecars, no neural eval    -- lost when the interactive
#            allocation timed out at 2026-08-12 01:59
#
# Seed 7 is the 39/40 bug the chain's own header warns about: stage 1 resume
# keys on the attacked wav alone, so pair013's wav (written 02:46, summary never
# written before preemption) makes that clip invisible to every resubmit. The
# bundle can never reach 40 while that wav sits there. Quarantining it -- moving,
# not deleting, so a mistake is reversible -- is what lets stage 1 redo the clip.
#
# The same quarantine runs before AND after every cell, because this script is
# itself running against a wall clock: the interactive allocation ends hours
# before seed 1 can finish, and a kill between the wav write and the summary
# write manufactures a fresh orphan. Cleaning up on the way in means the next
# run of this script repairs whatever this one leaves behind.
#
#   nohup bash run_s3a_af3_finish_local.sh > logs/s3a_af3_finish_local.log 2>&1 &
set -uo pipefail

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
cd "$HERE" || exit 1
mkdir -p logs
ts() { date '+%F %T'; }

BASE=${REPO_ROOT}/0_all_combined/results_codec_robust_waveform/03a_ai_detection_bypass/audio_flamingo
QUAR=${REPO_ROOT}/0_all_combined/quarantine/s3a_af3_orphan_wavs
EPS=0.06211

# An attacked wav with no matching summary.json. Never scored, but always
# skipped -- the worst of both.
quarantine_orphans() {
    # One name per `local`: bash marks every name in a single `local` local
    # before running the assignments, so `local seed=$1 d=${seed}` expands an
    # unset `seed` and `set -u` kills the script outright.
    local seed="$1"
    local d="$BASE/eps_${EPS}_seed${seed}/audio"
    local n=0
    [ -d "$d" ] || return 0
    mkdir -p "$QUAR"
    for w in "$d"/*_attacked.wav; do
        [ -e "$w" ] || continue
        local b="${w%_attacked.wav}"
        if [ ! -f "${b}_summary.json" ]; then
            mv "$w" "$QUAR/seed${seed}__$(basename "$w")"
            echo "[$(ts)]   quarantined orphan: $(basename "$w")"
            n=$((n + 1))
        fi
    done
    [ "$n" -gt 0 ] && echo "[$(ts)]   seed$seed: $n orphan wav(s) moved to $QUAR"
    return 0
}

counts() {
    local d="$BASE/eps_${EPS}_seed$1"
    printf '[%s]   seed%s: wav=%s summary=%s codecs=%s neural=%s\n' "$(ts)" "$1" \
        "$(ls "$d"/audio/*_attacked.wav 2>/dev/null | wc -l)" \
        "$(ls "$d"/audio/*_summary.json 2>/dev/null | wc -l)" \
        "$(ls "$d"/audio/*_codecs.json  2>/dev/null | wc -l)" \
        "$(ls "$d"/neural_delivery_eval_*.json 2>/dev/null | wc -l)"
}

# Counts, not SLURM state: preemption has repeatedly produced COMPLETED jobs on
# top of partial bundles, so completion is only ever read off the disk.
complete() {
    local d="$BASE/eps_${EPS}_seed$1"
    [ -d "$d" ] || return 1
    [ "$(ls "$d"/audio/*_summary.json 2>/dev/null | wc -l)" -eq 40 ] \
        && [ "$(ls "$d"/audio/*_codecs.json 2>/dev/null | wc -l)" -eq 40 ] \
        && ls "$d"/neural_delivery_eval_*.json > /dev/null 2>&1
}

echo "[$(ts)] === AF3 S3a waveform finish, local GPU on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# Seed 7 first: one clip plus sidecars, so the cell that is one file short is
# closed out before the long one starts and risks eating the whole allocation.
for SEED in 7 1; do
    if complete "$SEED"; then
        echo "[$(ts)] SKIP seed$SEED -- already 40/40 with sidecars and neural eval"
        continue
    fi

    # Both paths write the same directory; two writers corrupt it.
    STATE=$(squeue -u "$USER" -h -o "%T" -n "af3_s3a_wav_s${SEED}" 2>/dev/null | head -1)
    if [ "$STATE" = "RUNNING" ]; then
        echo "[$(ts)] SKIP seed$SEED -- af3_s3a_wav_s${SEED} is RUNNING under SLURM"
        continue
    fi
    if [ -n "$STATE" ]; then
        echo "[$(ts)] cancelling queued af3_s3a_wav_s${SEED} before running it locally"
        scancel -n "af3_s3a_wav_s${SEED}"; sleep 5
        if squeue -u "$USER" -h -o "%j" | grep -qx "af3_s3a_wav_s${SEED}"; then
            echo "[$(ts)] !! still queued after scancel -- skipping to avoid a double writer"
            continue
        fi
    fi

    echo "[$(ts)] === START seed$SEED (eps=$EPS) ==="
    counts "$SEED"
    quarantine_orphans "$SEED"
    bash run_grid_cell_local.sh audio_flamingo s3a waveform "$SEED" "$EPS" \
        > "logs/af3_s3a_wav_s${SEED}_LOCAL.out" 2>&1
    rc=$?
    quarantine_orphans "$SEED"
    counts "$SEED"
    complete "$SEED" && echo "[$(ts)] === END seed$SEED rc=$rc COMPLETE ===" \
                     || echo "[$(ts)] === END seed$SEED rc=$rc STILL SHORT ==="
done

echo "[$(ts)] chain done"
