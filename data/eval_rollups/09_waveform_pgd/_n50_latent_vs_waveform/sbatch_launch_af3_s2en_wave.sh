#!/bin/bash
# Fan the AF3 S2-English waveform cell out over its three seeds, once the SNR
# calibration for that cell exists.
#
# Submitted with --dependency=afterok:<calibration job>, so by the time this
# runs, 0_all_combined/calibration/snr_audio_flamingo_s2en.json is on disk and
# carries the eps' whose clean SNR matches the AF3 S2-EN latent reference
# (7.680/7.654/7.736 dB over seeds 1/7/42, mean 7.690).
#
#   sbatch --dependency=afterok:<calid> sbatch_launch_af3_s2en_wave.sh
#
# The three seeds are submitted as three independent jobs, so they occupy three
# GPUs concurrently rather than draining one after another.
#SBATCH --partition=cpu
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
HERE="$REPO/0_all_combined/n50_latent_vs_waveform"
CAL="$REPO/0_all_combined/calibration/snr_audio_flamingo_s2en.json"
cd "$HERE" || exit 1

if [ ! -f "$CAL" ]; then
    echo "ERROR: no calibration at $CAL -- refusing to guess an eps'"; exit 1
fi

# Read eps' AND the probe grid it came from. Omni S3b shipped a budget that was
# extrapolated past the largest probed eps; the prediction was ~1.0 dB off and
# two of three seeds fell outside the 1 dB SNR gate, costing a full rerun.
# Refuse to launch on an extrapolated budget rather than discover it afterwards.
read -r EPS PROBE_MAX <<EOF
$(python3 -c "
import json
j = json.load(open('$CAL'))
print(j['recommended_eps'], max(float(e) for e in j['per_eps']))
")
EOF

if [ -z "${EPS:-}" ]; then
    echo "ERROR: could not read recommended_eps from $CAL"; exit 1
fi

echo "=== AF3 S2-EN waveform launch ==="
echo "    calibration: $CAL"
echo "    eps'       = $EPS"
echo "    probe max  = $PROBE_MAX"

if python3 -c "import sys; sys.exit(0 if float('$EPS') > float('$PROBE_MAX') else 1)"; then
    echo "ERROR: eps'=$EPS is EXTRAPOLATED beyond the probe grid (max $PROBE_MAX)."
    echo "       This is the exact fault that broke Omni S3b. Re-probe with a"
    echo "       grid that brackets the answer instead of launching three seeds"
    echo "       against a budget that will miss the SNR gate."
    exit 1
fi

# SEEDS defaults to the full set. Narrow it when some seeds are being run
# elsewhere (e.g. one seed driven locally on an interactive GPU): the bundle dir
# takes exactly one writer, so a seed handled outside SLURM must not also be
# submitted here.
for SEED in ${SEEDS:-1 7 42}; do
    JOB="af3_s2en_wave_s${SEED}"
    BUNDLE="$REPO/0_all_combined/results_codec_robust_waveform/02_interview_screening/audio_flamingo/eps_${EPS}_en_seed${SEED}"
    n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
    if [ "$n" -ge 25 ]; then
        echo "  $JOB already 25/25 -- skip"; continue
    fi
    # One writer per bundle dir: never start a seed SLURM already owns.
    if squeue -u "$USER" -h -o "%j" | grep -qx "$JOB"; then
        echo "  $JOB already queued/running -- skip"; continue
    fi
    sbatch --export=ALL,MODEL=audio_flamingo,SCEN=s2en,ARM=waveform,SEED=$SEED,EPS=$EPS \
           --job-name="$JOB" sbatch_grid_cell.sh
done

echo "=== submitted; three seeds run concurrently ==="
squeue -u "$USER" -o "%.10i %.24j %.8T" | grep af3_s2en_wave || true
