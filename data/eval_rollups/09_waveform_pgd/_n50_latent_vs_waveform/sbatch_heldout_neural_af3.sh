#!/bin/bash
# ============================================================================
# Held-out NEURAL codec delivery eval (DAC + EnCodec) for AUDIO FLAMINGO 3.
#
# AF3 needs a SPLIT path: the flamingo3 env has no `dac` package, and installing
# the descript-audio-codec stack beside its pinned transformers 5.0.0.dev0 is not
# worth the risk. So per bundle:
#   STAGE A  codec-attack env  -> DAC/EnCodec round-trip, wavs to <bundle>/neural_roundtrip/
#   STAGE B  flamingo3 env     -> AF3 scores those wavs, writes neural_delivery_eval_audio_flamingo.json
# Round-trip maths is imported from neural_codec_delivery_eval.py in both paths,
# so a split-path bundle is indistinguishable from a single-env one downstream.
#
#   sbatch --export=ALL,GROUP=speech --job-name=ncd_af3_speech sbatch_heldout_neural_af3.sh
#   sbatch --export=ALL,GROUP=music  --job-name=ncd_af3_music  sbatch_heldout_neural_af3.sh
#
# Resumable at both stages: existing round-trip wavs are skipped, and bundles that
# already have the final JSON are skipped entirely.
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

MODEL=audio_flamingo
GROUP="${GROUP:-all}"
MIN_WAVS="${MIN_WAVS:-20}"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python   # has dac
F3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python      # has AF3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$GROUP" in
  speech) SCEN_RE='(01_finance_voice_agent|02_interview_screening)' ;;
  music)  SCEN_RE='(03a_ai_detection_bypass|03b_copyright_bypass)' ;;
  all)    SCEN_RE='(01_finance_voice_agent|02_interview_screening|03a_ai_detection_bypass|03b_copyright_bypass)' ;;
  *) echo "unknown GROUP=$GROUP"; exit 2 ;;
esac

echo "=========================================="
echo "  AF3 HELD-OUT NEURAL CODEC SWEEP (split path)  group=$GROUP"
echo "  Job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local}  GPU=${CUDA_VISIBLE_DEVICES:-?}"
echo "=========================================="

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
    PROMPT=$($CA - "$B" <<'PYEOF'
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

    echo "--- STAGE A: DAC/EnCodec round-trip (codec-attack env) ---"
    $CA -u 0_all_combined/scripts/precompute_neural_roundtrip.py \
        --bundle-dir "$B" --codecs dac,encodec
    if [ $? -ne 0 ]; then echo "!! STAGE A FAILED on $B"; FAILED=$((FAILED+1)); continue; fi

    echo "--- STAGE B: AF3 scoring (flamingo3 env) ---"
    $F3 -u 0_all_combined/scripts/neural_delivery_eval_from_wavs.py \
        --bundle-dir "$B" --target-model "$MODEL" \
        --codecs dac,encodec --prompt-text "$PROMPT"
    if [ $? -ne 0 ]; then echo "!! STAGE B FAILED on $B"; FAILED=$((FAILED+1)); fi
done

echo ""
echo "=== DONE AF3 group=$GROUP  failures=$FAILED ==="
for B in "${BUNDLES[@]}"; do
    if [ -f "$B/neural_delivery_eval_${MODEL}.json" ]; then echo "  ✅ $B"; else echo "  ❌ $B"; fi
done
exit $FAILED
