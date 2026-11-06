#!/bin/bash
# One cell of the multi-seed x waveform grid for the 27809 paper revision.
#
# A "cell" = (MODEL, SCEN, ARM, SEED) and produces one resumable bundle:
#   stage 1  attack
#   stage 2  MP3/AAC transport-codec sidecar
#   stage 3  DAC/EnCodec neural-delivery eval  (split path for AF3, which
#            lives in an env without the `dac` package)
#
#   MODEL  qwen2_audio | qwen25_omni | audio_flamingo
#   SCEN   s1 | s2 | s2en | s3a | s3b
#   ARM    latent | waveform | lp400
#   SEED   1 | 7 | 42
#   EPS    waveform/lp400 only; required for those, ignored for latent
#          (latent is always eps=1.0, the paper's budget)
#
# The waveform + lp400 arms pass --loader load_music so the carrier
# preprocessing matches the latent arm. On s1 that differs from the scenario
# default and the driver tags the bundle _norm, so these never collide with the
# published unnormalized bundles. On s3b load_music is already the default.
#
#   sbatch --export=ALL,MODEL=audio_flamingo,SCEN=s1,ARM=latent,SEED=7 \
#          --job-name=af3_s1_lat_s7 sbatch_grid_cell.sh
#   sbatch --export=ALL,MODEL=qwen2_audio,SCEN=s3b,ARM=waveform,SEED=1,EPS=0.075 \
#          --job-name=q2a_s3b_wav_s1 sbatch_grid_cell.sh
#SBATCH --partition=gpu-preempt
#SBATCH --account=${PI_GROUP}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail

REPO=${REPO_ROOT}
cd "$REPO" || exit 1
mkdir -p 0_all_combined/n50_latent_vs_waveform/logs

MODEL="${MODEL:?set MODEL=qwen2_audio|qwen25_omni|audio_flamingo}"
SCEN="${SCEN:?set SCEN=s1|s2|s2en|s3a|s3b}"
ARM="${ARM:?set ARM=latent|waveform|lp400}"
SEED="${SEED:?set SEED}"

ENVROOT=${WORK_ROOT}/miniconda3/envs
case "$MODEL" in
    qwen2_audio)    PY="$ENVROOT/codec-attack/bin/python" ;;
    qwen25_omni)    PY="$ENVROOT/qwen-omni/bin/python" ;;
    audio_flamingo) PY="$ENVROOT/flamingo3/bin/python" ;;
    *) echo "ERROR: unknown MODEL=$MODEL"; exit 1 ;;
esac
# precompute_neural_roundtrip.py needs the `dac` package, which only
# codec-attack has. AF3 scoring needs flamingo3. Hence the split in stage 3.
CA="$ENVROOT/codec-attack/bin/python"
export CODEC_ATTACK_PY="$PY"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$SCEN" in
    s1)  SCEN_DIR=01_finance_voice_agent; PROMPT="" ;;
    s3b) SCEN_DIR=03b_copyright_bypass
         PROMPT="$(cat 0_all_combined/scenarios/03b_copyright_bypass/system_prompt.txt)" ;;
    s3a) SCEN_DIR=03a_ai_detection_bypass
         PROMPT="$(cat 0_all_combined/scenarios/03a_ai_detection_bypass/system_prompt.txt)" ;;
    # S2 attacks under the driver's own QA_PROMPT, not the scenario
    # system_prompt.txt. Stages 2-3 must score under the same prompt the attack
    # optimized against, so it is repeated here verbatim from
    # run_s2_attack.py:57 -- if that constant changes, change it here too.
    s2)  SCEN_DIR=02_interview_screening
         PROMPT="Listen to the audio and follow the instruction." ;;
    # s2en = the 25-pair English carrier set, a different set from s2's 24-pair
    # Mandarin one. Same driver, --lang en, and the driver tags the bundle _en.
    s2en) SCEN_DIR=02_interview_screening
          PROMPT="Listen to the audio and follow the instruction." ;;
    *) echo "ERROR: unknown SCEN=$SCEN"; exit 1 ;;
esac

# The waveform driver's SCENARIOS map defines s1, s3b, s2en (since 2026-07-30)
# and s3a (since 2026-08-09). s2 (Mandarin) still has no waveform entry. Fail
# loudly rather than building a bundle path that never fills.
if [ "$ARM" != "latent" ] \
   && [ "$SCEN" != "s1" ] && [ "$SCEN" != "s3b" ] && [ "$SCEN" != "s2en" ] \
   && [ "$SCEN" != "s3a" ]; then
    echo "ERROR: ARM=$ARM is only supported for SCEN=s1|s3a|s3b|s2en (got $SCEN)"; exit 1
fi

echo "=== cell MODEL=$MODEL SCEN=$SCEN ARM=$ARM SEED=$SEED ==="
echo "    job $SLURM_JOB_ID on $SLURM_NODELIST GPU=$CUDA_VISIBLE_DEVICES"
echo "    python: $PY"
$PY -c 'import torch;print("    torch",torch.__version__,"| gpu",torch.cuda.get_device_name(0))'

# ---------------------------------------------------------------- stage 1
if [ "$ARM" = "latent" ]; then
    EPS=1.0
    echo "=== STAGE 1: latent attack (eps=1.0 seed=$SEED) ==="
    case "$SCEN" in
      s1)
        BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/eps_1.0_seed${SEED}"
        $PY -u 0_all_combined/scripts/run_s1_attack_bundleA.py \
            --target-model "$MODEL" --eps 1.0 --seed "$SEED" --rand-init
        ;;
      s3b|s3a)
        [ "$SCEN" = "s3b" ] && SUB=03b || SUB=03a
        BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/eps_1.0_multibitrate_seed${SEED}"
        $PY -u 0_all_combined/scripts/run_s3_attack.py \
            --sub-scenario "$SUB" --target-model "$MODEL" --eps 1.0 \
            --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval
        ;;
      s2)
        # --lang zh: the paper's cross_s2 table is the 24-pair Mandarin set
        # (the _en bundles are a separate 25-pair carrier set).
        BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/eps_1.0_multibitrate_seed${SEED}"
        $PY -u 0_all_combined/scripts/run_s2_attack.py \
            --target-model "$MODEL" --eps 1.0 --lang zh \
            --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval
        ;;
      s2en)
        # The published eps_1.0_multibitrate_en bundle has no --rand-init and no
        # seed, so it is not one draw from this distribution. Seed 1 is re-run
        # here under the seeded protocol, exactly as S1 re-ran eps_1.0_seed1
        # beside its published eps_1.0 bundle.
        BUNDLE="0_all_combined/results_codec_robust/${SCEN_DIR}/${MODEL}/eps_1.0_multibitrate_en_seed${SEED}"
        $PY -u 0_all_combined/scripts/run_s2_attack.py \
            --target-model "$MODEL" --eps 1.0 --lang en \
            --channel-mode multi_bitrate --seed "$SEED" --rand-init --skip-eval
        ;;
    esac
else
    EPS="${EPS:?waveform/lp400 arms need EPS=<calibrated eps>}"
    LP_FLAG=""; LP_TAG=""
    if [ "$ARM" = "lp400" ]; then LP_FLAG="--lowpass-hz 400"; LP_TAG="_lp400"; fi
    # s1's scenario default is soundfile, so --loader load_music adds _norm.
    NORM_TAG=""; [ "$SCEN" = "s1" ] && NORM_TAG="_norm"
    # Must match SCENARIOS[scen]["bundle_tag"] in the driver. s2en carries _en so
    # English and Mandarin bundles cannot collide under 02_interview_screening.
    SCEN_TAG=""; [ "$SCEN" = "s2en" ] && SCEN_TAG="_en"
    BUNDLE="0_all_combined/results_codec_robust_waveform/${SCEN_DIR}/${MODEL}/eps_${EPS}${SCEN_TAG}${LP_TAG}${NORM_TAG}_seed${SEED}"
    echo "=== STAGE 1: $ARM attack (eps=$EPS seed=$SEED loader=load_music) ==="
    $PY -u 0_all_combined/scripts/run_waveform_bundleA_multibitrate.py \
        --scenario "$SCEN" --target-model "$MODEL" \
        --eps "$EPS" --seed "$SEED" --rand-init \
        --loader load_music $LP_FLAG
fi

# ---------------------------------------------------------------- stage 2
echo "=== STAGE 2: MP3/AAC sidecar on $BUNDLE ==="
if [ -n "$PROMPT" ]; then
    $PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$BUNDLE" --target-model "$MODEL" --prompt-text "$PROMPT"
else
    $PY -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
        --bundle-dir "$BUNDLE" --target-model "$MODEL"
fi

# ---------------------------------------------------------------- stage 3
# Missing from the earlier S3b-waveform launcher, which is why that row shipped
# with empty DAC/EnCodec cells. Always run it here.
echo "=== STAGE 3: DAC/EnCodec neural delivery eval ==="
if [ "$MODEL" = "audio_flamingo" ]; then
    echo "--- 3A: round-trip (codec-attack env, has dac) ---"
    $CA -u 0_all_combined/scripts/precompute_neural_roundtrip.py \
        --bundle-dir "$BUNDLE" --codecs dac,encodec
    echo "--- 3B: AF3 scoring (flamingo3 env) ---"
    if [ -n "$PROMPT" ]; then
        $PY -u 0_all_combined/scripts/neural_delivery_eval_from_wavs.py \
            --bundle-dir "$BUNDLE" --target-model "$MODEL" \
            --codecs dac,encodec --prompt-text "$PROMPT"
    else
        $PY -u 0_all_combined/scripts/neural_delivery_eval_from_wavs.py \
            --bundle-dir "$BUNDLE" --target-model "$MODEL" --codecs dac,encodec
    fi
else
    if [ -n "$PROMPT" ]; then
        $PY -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
            --bundle-dir "$BUNDLE" --target-model "$MODEL" \
            --codecs dac,encodec --prompt-text "$PROMPT"
    else
        $PY -u 0_all_combined/scripts/neural_codec_delivery_eval.py \
            --bundle-dir "$BUNDLE" --target-model "$MODEL" --codecs dac,encodec
    fi
fi

# ---------------------------------------------------------------- report
# Counts only. SLURM COMPLETED is not evidence a bundle is whole: preemption
# has repeatedly produced COMPLETED jobs with partial bundles.
echo "=== DONE  bundle=$BUNDLE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l | xargs -I{} echo "  attacked wavs:  {}"
ls "$BUNDLE/audio/"*_codecs.json  2>/dev/null | wc -l | xargs -I{} echo "  codec sidecars: {}"
ls "$BUNDLE"/neural_delivery_eval_*.json 2>/dev/null | wc -l | xargs -I{} echo "  neural evals:   {}"
