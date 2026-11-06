#!/bin/bash
# ============================================================================
# Stage 3 ONLY: DAC/EnCodec neural delivery eval on a bundle that already has
# its attacked wavs.
#
# Exists because sbatch_codec_eot.sh never had a stage 3, so the Qwen2-Audio
# MP3/AAC/ensemble arms have neural evals for seed 1 only. That is what renders
# the DAC/EnCodec columns of those rows as "0†"/"2†" -- single-run daggers on an
# otherwise three-seed table. Filling them needs no re-attack, just this eval.
#
#   sbatch --export=ALL,RESULTS_DIR=results_codec_robust_mp3,MODEL=qwen2_audio,SEED=7 \
#          --job-name=ncd_mp3_q2a_s7 sbatch_ncd_eval_only.sh
#
# Refuses to run on an incomplete bundle: a partial neural eval scores as a
# biased-LOW ASR and there is no way to tell it from a real one afterwards.
# Idempotent -- skips if the eval already exists unless FORCE=1.
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR=results_codec_robust_mp3|_aac|_ensemble|...}"
MODEL="${MODEL:?set MODEL=qwen2_audio|qwen25_omni}"
SEED="${SEED:-1}"
FORCE="${FORCE:-0}"
EPS=1.0
SCEN=01_finance_voice_agent
EXPECT=50

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio) PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni) PY="$ENVROOT/qwen-omni/bin/python" ;;
    *) echo "ERROR: unsupported MODEL=$MODEL (audio_flamingo needs the"
       echo "       precompute_neural_roundtrip.py two-step; use that path)"; exit 1 ;;
esac
export CODEC_ATTACK_PY="$PY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUNDLE="0_all_combined/${RESULTS_DIR}/${SCEN}/${MODEL}/eps_${EPS}_seed${SEED}"
OUT="$BUNDLE/neural_delivery_eval_${MODEL}.json"

echo "=========================================="
echo "  stage-3 neural delivery eval (DAC + EnCodec)"
echo "  MODEL=$MODEL  ARM=$RESULTS_DIR  SEED=$SEED"
echo "  Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-$(hostname)}"
echo "  bundle: $BUNDLE"
echo "=========================================="

if [ ! -d "$BUNDLE" ]; then
    echo "ERROR: bundle does not exist: $BUNDLE"; exit 1
fi
n=$(ls "$BUNDLE"/audio/*_attacked.wav 2>/dev/null | wc -l)
if [ "$n" -lt "$EXPECT" ]; then
    echo "REFUSING: bundle is $n/$EXPECT wavs. A neural eval over a partial"
    echo "  bundle yields a biased-low ASR that is indistinguishable from a"
    echo "  real one. Finish the attack first."
    exit 1
fi
if [ -s "$OUT" ] && [ "$FORCE" != "1" ]; then
    echo "SKIP: $OUT already exists (FORCE=1 to overwrite)"; exit 0
fi

# Default --prompt-text is BANK_PROMPT, byte-identical to the S1 prompt the
# seed-1 evals used. Left implicit so the two can never drift apart.
$PY -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
    --bundle-dir "$BUNDLE" --target-model "$MODEL" --codecs dac,encodec

echo "=== DONE  $RESULTS_DIR $MODEL s$SEED ==="
# Counts only. Never trust an exit status as evidence a bundle is whole.
$PY - <<PYEOF
import json, os
p = "$OUT"
if not os.path.exists(p):
    print("  FAILED: no output written"); raise SystemExit(1)
j = json.load(open(p))
r = j.get("results", [])
print(f"  results rows: {len(r)}/$EXPECT")
for c in ("clean", "dac", "encodec"):
    print(f"    {c:8s} present on {sum(1 for x in r if c in x)} rows")
PYEOF
