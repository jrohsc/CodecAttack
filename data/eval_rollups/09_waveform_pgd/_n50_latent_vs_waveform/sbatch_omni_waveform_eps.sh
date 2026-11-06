#!/bin/bash
# ============================================================================
# SLURM job: Qwen2.5-Omni S1 waveform-space baseline with a PARAMETERIZED eps.
# Generalizes sbatch_omni_waveform.sh (which hardcodes EPS=0.075) so the
# band-limited arm can be run at an SNR-MATCHED budget instead of an
# L-infinity-matched one.
#
# Why: at eps'=0.075 the <=400 Hz arm lands at 16.5 dB SNR, ~10 dB quieter than
# the unconstrained waveform arm (6.34 dB) and the latent arm (6.77 dB), because
# band-limiting strips most of the perturbation's energy at a fixed L-inf bound.
# The paper's own protocol (appendix.tex:201) matches arms by carrier SNR to
# within 1 dB, so the lp400 arm must be re-run at a larger eps' to be a fair
# baseline.
#
#   EPS     float        perturbation budget (required)
#   LP      0 | <HZ>     0 = unconstrained; else band-limited <=LP Hz
#   MODE    lp | fft     default lp. lp  -> --lowpass-hz       (causal sosfilt)
#                                    fft -> --spectral-cutoff-hz (rFFT subspace)
#            Use fft for any CUTOFF SWEEP. Measured on this tree's own probes,
#            --lowpass-hz 800 leaves 99.3% of delta below 400 Hz and lp1500 is
#            roughly an 800 Hz arm, so an lp-mode sweep compares three copies
#            of the 400 Hz arm. fft800 puts 43.7%/99.2% below 400/800 Hz.
#   SEED    integer      default 1
#   STEPS   integer      default 1000 (set small for an SNR calibration probe)
#   PAIRS   integer      optional --max-pairs (probe only)
#
# Examples:
#   sbatch --export=ALL,EPS=0.24,LP=400,SEED=1 --job-name=omni_lp400_e024_s1 sbatch_omni_waveform_eps.sh
#   sbatch --export=ALL,EPS=0.24,LP=400,SEED=999,STEPS=100,PAIRS=3 --job-name=probe sbatch_omni_waveform_eps.sh
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
export CODEC_ATTACK_PY="$OMNI"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPS="${EPS:?set EPS}"
LP="${LP:-0}"
MODE="${MODE:-lp}"
SEED="${SEED:-1}"
STEPS="${STEPS:-1000}"
LP_FLAG=""; LP_TAG=""
# Tags must match the driver's own bundle_suffix (run_s1_waveform_bundleA_
# multibitrate.py:184-191): _lp<HZ> for --lowpass-hz, _fftlp<HZ> for
# --spectral-cutoff-hz. If they drift apart the wav count at the end of this
# script reads an empty directory and prints 0 for a run that finished 50/50.
if [ "$LP" != "0" ]; then
    case "$MODE" in
        lp)  LP_FLAG="--lowpass-hz $LP";         LP_TAG="_lp${LP}"    ;;
        fft) LP_FLAG="--spectral-cutoff-hz $LP"; LP_TAG="_fftlp${LP}" ;;
        *) echo "ERROR: MODE must be lp or fft (got '$MODE')"; exit 2 ;;
    esac
fi
PAIRS_FLAG=""
if [ -n "${PAIRS:-}" ]; then PAIRS_FLAG="--max-pairs $PAIRS"; fi
BUNDLE="0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen25_omni/eps_${EPS}${LP_TAG}_seed${SEED}"

echo "=== OMNI S1 waveform  EPS=$EPS LP=$LP MODE=$MODE SEED=$SEED STEPS=$STEPS  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "  bundle: $BUNDLE"
$OMNI -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

echo "=== STAGE 1: waveform attack ==="
$OMNI -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --target-model qwen25_omni --eps "$EPS" --seed "$SEED" --rand-init \
    --steps "$STEPS" $LP_FLAG $PAIRS_FLAG
STAGE1=$?

# The sibling wrappers use `set -uo pipefail` WITHOUT -e, so a preempted stage 1
# falls through to stage 2, which runs on the partial wav set, prints DONE, and
# exits 0 -- sacct then records COMPLETED for a truncated bundle. Guard it here.
if [ "$STAGE1" -ne 0 ]; then
    echo "!!! STAGE 1 exited $STAGE1 -- NOT running the sidecar. Bundle is PARTIAL."
    ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs so far: {}"
    exit "$STAGE1"
fi

# Skip the codec sidecar for calibration probes -- they only need SNR.
if [ -z "${PAIRS:-}" ]; then
    echo "=== STAGE 2: MP3+AAC held-out sidecar ==="
    $OMNI -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$BUNDLE" --target-model qwen25_omni
else
    echo "=== STAGE 2 skipped (calibration probe) ==="
fi

echo "=== DONE  EPS=$EPS LP=$LP MODE=$MODE seed=$SEED  bundle=$BUNDLE ==="
# Count with find, not `ls <glob>`: an empty glob makes ls exit 2, and under
# `pipefail` that becomes the script's exit status -> SLURM records FAILED for a
# run that actually succeeded. Keep the final status a true reflection of stage 1.
echo "  attacked wavs:  $(find "$BUNDLE/audio" -name '*_attacked.wav' 2>/dev/null | wc -l)"
echo "  codec sidecars: $(find "$BUNDLE/audio" -name '*_codecs.json'  2>/dev/null | wc -l)"
exit 0
