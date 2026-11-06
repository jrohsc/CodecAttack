#!/bin/bash
# One S2-English cell, run in a terminal instead of through sbatch.
#
#   bash run_s2en_local.sh MODEL SEED
#   bash run_s2en_local.sh qwen2_audio 1
#
# Same three stages as sbatch_grid_cell.sh SCEN=s2en, same bundle path, so a
# locally-started cell and a later sbatch of the same cell resume each other.
#
# REFUSES to start if a live SLURM job already owns the cell. Two writers on one
# bundle dir is the failure mode TODO.md:300 warns about; scancel first.
set -uo pipefail

MODEL="${1:?usage: run_s2en_local.sh MODEL SEED}"
SEED="${2:?usage: run_s2en_local.sh MODEL SEED}"

REPO=${REPO_ROOT}
cd "$REPO" || exit 1

declare -A TAG=( [qwen2_audio]=q2a [qwen25_omni]=omni [audio_flamingo]=af3 )
JOBNAME="${TAG[$MODEL]}_s2en_lat_s${SEED}"
if squeue -u "$USER" -h -o "%j %T" | grep -q "^${JOBNAME} "; then
    echo "REFUSING: SLURM job $JOBNAME is live and owns this bundle."
    echo "  scancel it first, then re-run."
    exit 1
fi

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio)    PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni)    PY="$ENVROOT/qwen-omni/bin/python" ;;
    audio_flamingo) PY="$ENVROOT/flamingo3/bin/python" ;;
    *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
CA="$ENVROOT/codec-attack/bin/python"
export CODEC_ATTACK_PY="$PY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCEN_DIR=02_interview_screening
# Verbatim from run_s2_attack.py QA_PROMPT: stages 2-3 must score under the same
# prompt the attack optimized against.
PROMPT="Listen to the audio and follow the instruction."
BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/eps_1.0_multibitrate_en_seed${SEED}"

echo "=== LOCAL cell MODEL=$MODEL SCEN=s2en SEED=$SEED ==="
echo "    host $(hostname)  bundle $BUNDLE"
$PY -c 'import torch;print("    torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

echo "=== STAGE 1: latent attack (eps=1.0 seed=$SEED lang=en) ==="
$PY -u 0_all_combined/scripts/run_s2_attack.py \
    --target-model "$MODEL" --eps 1.0 --lang en \
    --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval

echo "=== STAGE 2: MP3/AAC sidecar ==="
$PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model "$MODEL" --prompt-text "$PROMPT"

echo "=== STAGE 3: DAC/EnCodec neural delivery eval ==="
if [ "$MODEL" = "audio_flamingo" ]; then
    $CA -u 0_all_combined/scripts/precompute_neural_roundtrip.py \
        --bundle-dir "$BUNDLE" --codecs dac,encodec
    $PY -u 0_all_combined/scripts/neural_delivery_eval_from_wavs.py \
        --bundle-dir "$BUNDLE" --target-model "$MODEL" \
        --codecs dac,encodec --prompt-text "$PROMPT"
else
    $PY -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
        --bundle-dir "$BUNDLE" --target-model "$MODEL" \
        --codecs dac,encodec --prompt-text "$PROMPT"
fi

# Counts only. Never trust an exit status as evidence a bundle is whole.
echo "=== DONE  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs:  {}/25"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
ls "$BUNDLE"/neural_delivery_eval_*.json 2>/dev/null | wc -l | xargs -I{} echo "  neural evals:   {}"
