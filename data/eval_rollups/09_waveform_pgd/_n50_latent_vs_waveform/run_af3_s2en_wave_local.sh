#!/bin/bash
# Drive ONE seed of the AF3 S2-English waveform cell on this interactive GPU,
# while SLURM carries the other two (see sbatch_launch_af3_s2en_wave.sh, which
# is submitted with SEEDS="7 42" so it never touches this seed's bundle).
#
#   SEED=1 bash run_af3_s2en_wave_local.sh
#
# Waits for the SNR calibration to land, then runs the identical three-stage
# chain sbatch_grid_cell.sh runs, so the local bundle and the two SLURM bundles
# are produced by the same code path and are directly comparable.
set -uo pipefail

REPO=${REPO_ROOT}
HERE="$REPO/0_all_combined/n50_latent_vs_waveform"
CAL="$REPO/0_all_combined/calibration/snr_audio_flamingo_s2en.json"
SEED="${SEED:-1}"
CAL_JOB="${CAL_JOB:-62489642}"
cd "$HERE" || exit 1
mkdir -p logs

# The broken ffmpeg first on the interactive PATH (kimi-audio's, missing
# libx264.so.138) returns 127 on the Opus encode and silently killed the first
# calibration attempt. Pin the flamingo3 toolchain, which is the env this cell
# runs under anyway.
export PATH=${WORK_ROOT}/miniconda3/envs/flamingo3/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[local] seed=$SEED host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[local] ffmpeg: $(command -v ffmpeg)"
ffmpeg -hide_banner -version >/dev/null 2>&1 \
    || { echo "[local] FATAL: ffmpeg is not runnable on PATH"; exit 1; }

# ---- wait for the calibration -------------------------------------------
while [ ! -f "$CAL" ]; do
    st=$(sacct -j "$CAL_JOB" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
            if [ ! -f "$CAL" ]; then
                echo "[local] FATAL: calibration $CAL_JOB ended as $st with no JSON."
                echo "        Refusing to invent an eps'. See logs/cal_af3_s2en_${CAL_JOB}.err"
                exit 1
            fi ;;
    esac
    echo "[local] waiting for calibration ($st) ..."
    sleep 60
done

read -r EPS PROBE_MAX <<EOF
$(python3 -c "
import json
j = json.load(open('$CAL'))
print(j['recommended_eps'], max(float(e) for e in j['per_eps']))
")
EOF
[ -n "${EPS:-}" ] || { echo "[local] FATAL: no recommended_eps in $CAL"; exit 1; }

# Same guard the SLURM launcher applies: an eps' fitted past the probe grid is
# what cost us the Omni S3b rerun. Fail before spending GPU hours, not after.
if python3 -c "import sys; sys.exit(0 if float('$EPS') > float('$PROBE_MAX') else 1)"; then
    echo "[local] FATAL: eps'=$EPS is EXTRAPOLATED beyond probe max $PROBE_MAX."
    exit 1
fi

BUNDLE="$REPO/0_all_combined/results_codec_robust_waveform/02_interview_screening/audio_flamingo/eps_${EPS}_en_seed${SEED}"
echo "[local] eps'=$EPS (probe max $PROBE_MAX)"
echo "[local] bundle=$BUNDLE"

n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
[ "$n" -ge 25 ] && { echo "[local] already 25/25 -- nothing to do"; exit 0; }

# One writer per bundle: bail out if SLURM somehow also owns this seed.
if squeue -u "$USER" -h -o "%j" | grep -qx "af3_s2en_wave_s${SEED}"; then
    echo "[local] FATAL: SLURM already owns af3_s2en_wave_s${SEED}. Two writers"
    echo "        on one bundle corrupts it. Cancel that job or pick another seed."
    exit 1
fi

# sbatch_grid_cell.sh is plain bash (the #SBATCH lines are comments) but runs
# under `set -u`, so give it the SLURM vars it echoes. Running the same script
# rather than reimplementing its three stages is the whole point.
export SLURM_JOB_ID="local-$$"
export SLURM_NODELIST="$(hostname)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL=audio_flamingo SCEN=s2en ARM=waveform SEED="$SEED" EPS="$EPS"

LOG="logs/local_af3_s2en_wave_s${SEED}.log"
echo "[local] === starting at $(date -Is); log -> $LOG ==="
bash sbatch_grid_cell.sh 2>&1 | tee -a "$LOG"

n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
echo "[local] === finished at $n/25 wavs ($(date -Is)) ==="
# The bundle is resumable: if the interactive session's walltime cuts this
# short, rerunning this script (or submitting the seed to SLURM) picks up the
# remaining pairs rather than restarting from zero.
[ "$n" -ge 25 ] || echo "[local] INCOMPLETE -- resume by rerunning this script or sbatch the seed"
