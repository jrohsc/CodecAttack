#!/usr/bin/env bash
# ============================================================================
# SNR calibration probes for NEW band-limited cutoffs (800 Hz, 1500 Hz).
#
# The lp400 arm in Table 4 is SNR-matched, not L-inf-matched: eps'=0.30 was
# chosen so the band-limited arm lands within 1 dB of the latent arm on clean
# carrier SNR. A wider passband lets more of delta through the projection, so
# 800 Hz and 1500 Hz each need their OWN eps' -- reusing 0.30 would make them
# several dB louder than lp400 and the falsification would be unfalsifiable
# ("of course it worked, you gave it more budget").
#
# PROTOCOL NOTE. The original lp400 calibration probed at STEPS=100, but the
# shipped bundles ran 1000 steps on 50 carriers, and those two settings do NOT
# give the same SNR (probe 8.38 dB vs bundle 5.41 dB for the same eps=0.30).
# So this script probes at the SHIPPING setting (1000 steps) on 3 carriers and
# re-measures lp400@0.30 under that same protocol as the anchor. Match the new
# cutoffs to the ANCHOR's probe SNR, not to the 5.41 dB bundle number.
#
# Seed 998 is deliberately fresh: seed999 already holds 100-step lp400 probes
# and the driver skips-when-exists, so reusing it would silently return the
# old 100-step wavs.
#
# Usage:
#   bash run_lp_cutoff_calib_local.sh 400:0.30 800:0.20 1500:0.12
#   (each arg is CUTOFF_HZ:EPS)
# ============================================================================
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1

OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEED="${SEED:-998}"
STEPS="${STEPS:-1000}"
PAIRS="${PAIRS:-3}"
# MODE=lp  -> --lowpass-hz       (causal sosfilt re-applied every step)
# MODE=fft -> --spectral-cutoff-hz (exact, idempotent rFFT subspace)
#
# Use fft for anything that is meant to READ as a cutoff sweep. The lp path is
# not a projection: sosfilt is causal, so re-applying it for 1000 steps drags
# energy toward DC no matter what the nominal cutoff is. Measured on the real
# attack, --lowpass-hz 800 leaves 99.68% of delta below 400 Hz -- i.e. it is a
# 400 Hz arm wearing an 800 Hz label, and lp1500 is roughly an 800 Hz arm.
# Comparing lp400/lp800/lp1500 therefore compares three copies of one arm.
MODE="${MODE:-lp}"
case "$MODE" in
    lp)  FLAG="--lowpass-hz";         TAGPFX="lp"    ;;
    fft) FLAG="--spectral-cutoff-hz"; TAGPFX="fftlp" ;;
    *) echo "ERROR: MODE must be lp or fft (got '$MODE')"; exit 2 ;;
esac
BASE=0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni
LOG_DIR=0_all_combined/n50_latent_vs_waveform/logs
OUT=0_all_combined/n50_latent_vs_waveform/lp_cutoff_calib.jsonl
mkdir -p "$LOG_DIR"

echo "=== lp-cutoff calibration  seed=$SEED steps=$STEPS pairs=$PAIRS on $(hostname) ==="
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

for SPEC in "$@"; do
    HZ="${SPEC%%:*}"
    EPS="${SPEC##*:}"
    # The driver names the bundle with python's f"{eps:g}", so 0.30 -> "0.3".
    # Building the path from the raw string instead silently points the wav
    # count and the SNR read at a directory that never gets written -- the same
    # bug makes sbatch_omni_waveform_eps.sh print "attacked wavs: 0" for runs
    # that in fact completed 50/50.
    EPS_TAG=$(printf '%g' "$EPS")
    BUNDLE="$BASE/eps_${EPS_TAG}_${TAGPFX}${HZ}_seed${SEED}"
    TAG="${TAGPFX}${HZ}_e${EPS_TAG}_s${SEED}"
    echo
    echo "=== [$(date '+%F %H:%M:%S')] PROBE $MODE=${HZ}Hz eps=${EPS} -> $BUNDLE ==="

    $OMNI -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
        --target-model qwen25_omni --eps "$EPS" --seed "$SEED" --rand-init \
        --steps "$STEPS" --max-pairs "$PAIRS" "$FLAG" "$HZ" \
        > "$LOG_DIR/calib_${TAG}.log" 2>&1
    RC=$?
    N=$(find "$BUNDLE/audio" -name '*_attacked.wav' 2>/dev/null | wc -l)
    echo "  rc=$RC  wavs=$N"
    if [ "$RC" -ne 0 ] || [ "$N" -lt "$PAIRS" ]; then
        echo "  !! INCOMPLETE -- not recording SNR (see $LOG_DIR/calib_${TAG}.log)"
        continue
    fi

    # Clean-channel carrier SNR. `soundfile` is the loader the waveform driver
    # actually attacked under (_load_carrier_24k = read, resample, trim, NO
    # peak-normalize), so it is the correct convention for these bundles.
    # load_music is recorded alongside only as a cross-check -- do not mix the
    # two in one comparison.
    SNR_SF=$($OMNI 0_all_combined/scripts/measure_bundle_snr.py \
                 --bundle "$BUNDLE" --loader soundfile 2>/dev/null)
    SNR_LM=$($OMNI 0_all_combined/scripts/measure_bundle_snr.py \
                 --bundle "$BUNDLE" --loader load_music 2>/dev/null)
    echo "  soundfile : $SNR_SF"
    echo "  load_music: $SNR_LM"
    $OMNI -c '
import json, sys
sf, lm = json.loads(sys.argv[1]), json.loads(sys.argv[2])
print(json.dumps({"mode": sys.argv[7], "cutoff_hz": float(sys.argv[3]),
                  "eps": float(sys.argv[4]),
                  "seed": int(sys.argv[5]), "steps": int(sys.argv[6]),
                  "n": sf["n"], "bundle": sf["bundle"],
                  "snr_soundfile": sf["snr_mean"], "snr_sd_soundfile": sf["snr_sd"],
                  "snr_load_music": lm["snr_mean"]}))' \
        "$SNR_SF" "$SNR_LM" "$HZ" "$EPS_TAG" "$SEED" "$STEPS" "$MODE" >> "$OUT"
done

echo
echo "=== ALL PROBES DONE  $(date '+%F %H:%M:%S') ==="
cat "$OUT" 2>/dev/null
