#!/usr/bin/env bash
# Submit the AF3 S3a waveform cell as soon as its SNR calibration lands.
#
# Why a separate watcher: submit_waveform_after_calib.sh has a hardcoded GRID
# covering s1 + s3b only, and it exits as soon as that grid is resolved. S3a was
# added to the waveform driver later (2026-08-09), and only AF3 is still missing
# -- q2a and omni already have their eps' and their bundles.
#
# The AF3 S3a calibration died mid-grid on 2026-08-10 01:57 (local run, killed
# after the eps=0.1 point) without writing snr_audio_flamingo_s3a.json, so no
# eps' ever existed and no waveform job was ever submitted. It is being rerun on
# the local idle A100 via run_calib_s3a_local.sh, so the liveness check watches
# that process rather than squeue.
#
# Run detached:
#   nohup bash submit_af3_s3a_after_calib.sh > logs/submit_af3_s3a.log 2>&1 &
set -uo pipefail

HERE=${REPO_ROOT}/0_all_combined/n50_latent_vs_waveform
CALIB=${REPO_ROOT}/0_all_combined/calibration
cd "$HERE" || exit 1

JSON="$CALIB/snr_audio_flamingo_s3a.json"
# Five eps points x 3 carriers at ~7 min each ran ~2 h last time; 24 h leaves
# room for a retry without the watcher expiring under it.
DEADLINE=$(( $(date +%s) + 24*3600 ))

echo "[$(date '+%F %T')] waiting for $JSON"

while [ ! -s "$JSON" ]; do
    # Liveness = the local calibration process, not squeue. It dies with the
    # node's interactive session, which is exactly how the 2026-08-10 run was
    # lost silently -- so notice it here rather than waiting out the deadline.
    if ! pgrep -f "run_calib_s3a_local.sh" > /dev/null; then
        echo "[$(date '+%F %T')] !! run_calib_s3a_local.sh is no longer running and $JSON was never written -- giving up."
        exit 1
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "[$(date '+%F %T')] !! deadline reached, no calibration."; exit 1
    fi
    sleep 120
done

EPS=$(python3 -c "
import json
d = json.load(open('$JSON'))
e = d.get('recommended_eps')
print('' if e is None else f'{float(e):g}')
" 2>/dev/null)

if [ -z "$EPS" ]; then
    echo "[$(date '+%F %T')] !! $JSON has no recommended_eps -- not submitting."; exit 1
fi
echo "[$(date '+%F %T')] audio_flamingo/s3a eps'=$EPS -- submitting 3 seeds"

# Two Claude sessions have submitted into this tree before; guard on the job
# name so a re-run of this watcher cannot double-submit a live cell.
RUNNING=$(squeue -u "$USER" -h -o "%j")
for SEED in 1 7 42; do
    NAME="af3_s3a_wav_s${SEED}"
    if echo "$RUNNING" | grep -qx "$NAME"; then
        echo "  SKIP $NAME (already in queue)"; continue
    fi
    sbatch --export=ALL,MODEL=audio_flamingo,SCEN=s3a,ARM=waveform,SEED="$SEED",EPS="$EPS" \
           --job-name="$NAME" sbatch_grid_cell.sh
done
echo "[$(date '+%F %T')] submitter done."
