#!/bin/bash
# ============================================================================
# SLURM job: held-out NEURAL codec delivery eval (DAC + EnCodec) swept over
# every bundle of one model that does not have it yet.
#
# Inference only. Per BUNDLE granularity: each bundle writes its own
# neural_delivery_eval_<model>.json the moment it finishes, and bundles that
# already have that file are skipped. So a preempt/requeue resumes cleanly and
# never leaves a half-written result (cf. the silent-truncation failure mode).
#
# The prompt is taken from each bundle's own *_codecs.json "prompt" field, so
# the neural columns use the identical system prompt as the MP3/AAC sidecar.
#
#   sbatch --export=ALL,MODEL=qwen25_omni,GROUP=speech  --job-name=ncd_omni_speech  sbatch_heldout_neural_sweep.sh
#   sbatch --export=ALL,MODEL=qwen25_omni,GROUP=music   --job-name=ncd_omni_music   sbatch_heldout_neural_sweep.sh
#   sbatch --export=ALL,MODEL=audio_flamingo,GROUP=speech --job-name=ncd_af3_speech sbatch_heldout_neural_sweep.sh
#   sbatch --export=ALL,MODEL=audio_flamingo,GROUP=music  --job-name=ncd_af3_music  sbatch_heldout_neural_sweep.sh
#
# GROUP: speech = 01_finance + 02_interview ; music = 03a + 03b ; all = both.
# MIN_WAVS (default 20) skips the tiny probe/diagnostic bundles.
# ============================================================================
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

MODEL="${MODEL:?set MODEL=qwen25_omni|audio_flamingo|qwen2_audio}"
GROUP="${GROUP:-all}"
MIN_WAVS="${MIN_WAVS:-20}"

case "$MODEL" in
  qwen25_omni)    PY=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python ;;
  audio_flamingo) PY=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python ;;
  qwen2_audio)    PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$GROUP" in
  speech) SCEN_RE='(01_finance_voice_agent|02_interview_screening)' ;;
  music)  SCEN_RE='(03a_ai_detection_bypass|03b_copyright_bypass)' ;;
  all)    SCEN_RE='(01_finance_voice_agent|02_interview_screening|03a_ai_detection_bypass|03b_copyright_bypass)' ;;
  *) echo "unknown GROUP=$GROUP"; exit 2 ;;
esac

echo "=========================================="
echo "  HELD-OUT NEURAL CODEC SWEEP  model=$MODEL group=$GROUP"
echo "  Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local}  GPU=${CUDA_VISIBLE_DEVICES:-?}"
echo "=========================================="
$PY -c 'import torch;print("  torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))' || exit 1

# --- build the work list: bundles for this model, this group, >=MIN_WAVS wavs,
#     no neural_delivery_eval_<model>.json yet -------------------------------
mapfile -t BUNDLES < <(
  find 0_all_combined/results_codec_robust* -type d -name audio 2>/dev/null \
    | sed 's#/audio$##' | sort -u \
    | grep -E "/$SCEN_RE/" | grep -E "/${MODEL}/" \
    | while read -r d; do
        n=$(ls "$d"/audio/*_attacked.wav 2>/dev/null | wc -l)
        [ "$n" -lt "$MIN_WAVS" ] && continue
        [ -f "$d/neural_delivery_eval_${MODEL}.json" ] && continue
        echo "$d"
      done
)

echo "--- ${#BUNDLES[@]} bundle(s) to do ---"
printf '    %s\n' "${BUNDLES[@]}"
[ "${#BUNDLES[@]}" -eq 0 ] && { echo "nothing to do"; exit 0; }

FAILED=0
for B in "${BUNDLES[@]}"; do
    # prompt straight from the bundle's own codec sidecar (falls back to the
    # scenario system_prompt.txt if no sidecar exists)
    PROMPT=$($PY - "$B" <<'PYEOF'
import glob, json, os, sys
b = sys.argv[1]
cj = sorted(glob.glob(os.path.join(b, "audio", "*_codecs.json")))
p = ""
if cj:
    p = (json.load(open(cj[0])).get("prompt") or "").strip()
if not p:
    scen = [s for s in b.split(os.sep) if s[:2] in ("01", "02", "03")]
    if scen:
        f = os.path.join("0_all_combined/scenarios", scen[0], "system_prompt.txt")
        if os.path.isfile(f):
            p = open(f).read().strip()
print(p)
PYEOF
)
    if [ -z "$PROMPT" ]; then echo "!! no prompt for $B -- SKIPPED"; FAILED=$((FAILED+1)); continue; fi

    echo ""
    echo ">>> $B"
    echo "    prompt: ${PROMPT:0:80}..."
    $PY -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
        --bundle-dir "$B" --target-model "$MODEL" \
        --codecs dac,encodec --prompt-text "$PROMPT"
    rc=$?
    if [ $rc -ne 0 ]; then echo "!! FAILED rc=$rc on $B"; FAILED=$((FAILED+1)); fi
done

echo ""
echo "=== DONE model=$MODEL group=$GROUP  failures=$FAILED ==="
for B in "${BUNDLES[@]}"; do
    f="$B/neural_delivery_eval_${MODEL}.json"
    if [ -f "$f" ]; then echo "  ✅ $B"; else echo "  ❌ $B"; fi
done
exit $FAILED
