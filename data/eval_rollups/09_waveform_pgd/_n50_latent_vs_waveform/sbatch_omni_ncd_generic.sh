#!/bin/bash
# ============================================================================
# Generic held-out NEURAL-CODEC delivery eval (DAC + EnCodec) on ANY Qwen2.5-Omni
# bundle. Inference-only: feeds already-attacked wavs through the neural codec
# round-trip and re-queries Omni. Non-preempt `gpu` partition so it can't be evicted.
#
#   sbatch --export=ALL,BUNDLE=<relpath-under-repo> --job-name=<name> sbatch_omni_ncd_generic.sh
# ============================================================================
#SBATCH --partition=gpu
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
cd "$REPO"; mkdir -p 0_all_combined/n50_latent_vs_waveform/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUNDLE="${BUNDLE:?set BUNDLE (relpath under repo)}"
# Each target model lives in its own conda env — qwen25_omni in qwen-omni,
# qwen2_audio in codec-attack. Picking the wrong interpreter fails at import.
MODEL="${MODEL:-qwen25_omni}"
case "$MODEL" in
    qwen25_omni) PYBIN=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python ;;
    qwen2_audio) PYBIN=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python ;;
    *) echo "ERROR: MODEL must be qwen25_omni|qwen2_audio (got '$MODEL')"; exit 1 ;;
esac
[ -x "$PYBIN" ] || { echo "ERROR: interpreter not found: $PYBIN"; exit 1; }

echo "=== neural DAC/EnCodec delivery eval  job $SLURM_JOB_ID on $SLURM_NODELIST ==="
echo "  bundle: $BUNDLE"
echo "  model:  $MODEL  ($PYBIN)"
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs present: {}"

# The eval script defaults --prompt-text to BANK_PROMPT, which is only correct for
# 01_finance bundles. Any other scenario MUST pass PROMPT_FILE or the clean control
# collapses to 0/N and the whole bundle is an artifact.
PROMPT_ARGS=()
if [ -n "${PROMPT_FILE:-}" ]; then
    [ -f "$PROMPT_FILE" ] || { echo "ERROR: PROMPT_FILE not found: $PROMPT_FILE"; exit 1; }
    PROMPT_ARGS=(--prompt-text "$(cat "$PROMPT_FILE")")
    echo "  prompt-file: $PROMPT_FILE"
else
    case "$BUNDLE" in
        *01_finance_voice_agent*) echo "  prompt: default BANK_PROMPT (matches 01_finance)" ;;
        *) echo "ERROR: non-finance bundle with no PROMPT_FILE — refusing to run with BANK_PROMPT"; exit 1 ;;
    esac
fi

$PYBIN -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
    --bundle-dir "$BUNDLE" --target-model "$MODEL" --codecs dac,encodec \
    ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}

echo "=== DONE  wrote neural_delivery_eval_${MODEL}.json in $BUNDLE ==="
