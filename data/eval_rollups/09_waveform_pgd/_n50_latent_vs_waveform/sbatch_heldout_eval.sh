#!/bin/bash
# Score an already-crafted ensemble bundle on one model (member or held-out).
#
# Eval only -- no attack, no gradients. The ensemble attack driver scores ONLY
# its primary member (qwen2_audio), so without this every ensemble bundle on
# disk carries a lone self-score that reads as a transfer result if quoted
# carelessly. out-name is mandatory here and must say MEMBER or HELDOUT.
#
#   sbatch --export=ALL,BUNDLE=<dir>,MODEL=qwen25_omni_3b,OUTNAME=omni3b_HELDOUT \
#          --job-name=ho_e3_omni3b sbatch_heldout_eval.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1

BUNDLE="${BUNDLE:?set BUNDLE=<bundle dir>}"
MODEL="${MODEL:?set MODEL=qwen2_audio|qwen25_omni|qwen25_omni_3b|audio_flamingo}"
OUTNAME="${OUTNAME:?set OUTNAME=<name>_MEMBER or <name>_HELDOUT}"
# Same channel set the launcher's own hint uses, so these rows line up with the
# committed §1.2 table. DAC/EnCodec need the `dac` package (codec-attack env
# only) and are left to the neural-delivery path.
CHANNELS="${CHANNELS:-opus64,opus128,mp3_64,mp3_128,aac_64,aac_128}"

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio)                  PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni|qwen25_omni_3b)   PY="$ENVROOT/qwen-omni/bin/python" ;;
    audio_flamingo)               PY="$ENVROOT/flamingo3/bin/python" ;;
    # Kimi is the 4th-architecture HELD-OUT victim; `kimia_infer` is not
    # pip-installed, it lives in the sibling project's checkout.
    kimi_audio)                   PY="$ENVROOT/kimi-audio/bin/python"
                                  export PYTHONNOUSERSITE=1
                                  export KIMI_INFER_ROOT="${KIMI_INFER_ROOT:-${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio}"
                                  export PYTHONPATH="$KIMI_INFER_ROOT:${PYTHONPATH:-}" ;;
    phi4)                         PY="$ENVROOT/audio-llm/bin/python" ;;
    *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
export PATH="$ENVROOT/codec-attack/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== heldout/member eval ==="
echo "    bundle:   $BUNDLE"
echo "    model:    $MODEL   -> $OUTNAME"
echo "    channels: clean,$CHANNELS"
echo "    job $SLURM_JOB_ID on $SLURM_NODELIST"

LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
GEN_ARGS=()
[[ "$MODEL" == kimi_audio ]] && GEN_ARGS=(--max-new-tokens 100)

$PY -u 0_all_combined/scripts/ensemble_heldout_eval.py \
    --bundle-dir "$BUNDLE" \
    --target-model "$MODEL" \
    --out-name "$OUTNAME" \
    --channels "$CHANNELS" "${GEN_ARGS[@]}" $LIMIT_ARG

echo "=== DONE  $BUNDLE / $OUTNAME ==="
# Counts only -- SLURM COMPLETED is not evidence the eval covered every clip.
echo "  crafted wavs:  $(find "$BUNDLE/audio" -name '*.wav' 2>/dev/null | wc -l)"
echo "  eval records:  $(find "$BUNDLE/$OUTNAME" -name 'record.json' 2>/dev/null | wc -l)"
N_WAV=$(find "$BUNDLE/audio" -maxdepth 1 -name '*.wav' | wc -l)
N_EVAL=$(find "$BUNDLE/$OUTNAME" -name 'record.json' | wc -l)
(( N_WAV == N_EVAL )) || { echo "ERROR: evaluation incomplete ($N_EVAL/$N_WAV)" >&2; exit 1; }
