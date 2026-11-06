#!/usr/bin/env bash
# Mechanistic-control variant of run_waveform.sh:
# adds --lowpass-hz 400 to project the waveform-PGD perturbation into
# the [0, 400 Hz] sub-band after every step. Tests whether the latent
# attack's codec-survival advantage is *causally* driven by spectral
# confinement (the bark-band fingerprint shows latent puts ~32% of its
# energy below 400 Hz vs waveform's 14%).
#
# Bundle is auto-tagged eps_0.075_lp400 by the driver, so this runs in
# parallel with the unconstrained waveform bundle without overwriting.
#
# Resumable: same skip-when-exists semantics as run_waveform.sh.
set -euo pipefail

REPO=${REPO_ROOT}
cd "$REPO"

PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
SCEN_DIR=01_finance_voice_agent
EPS=0.075
LP_HZ=400
BUNDLE="0_all_combined/results_codec_robust_waveform/${SCEN_DIR}/qwen2_audio/eps_${EPS}_lp${LP_HZ}"
LOG_DIR="0_all_combined/n50_latent_vs_waveform/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ts() { date "+%F %H:%M:%S"; }

echo "=== [$(ts)] WAVEFORM-LP${LP_HZ} n=50  bundle=$BUNDLE ==="
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "=== [$(ts)] STAGE 1: attack (lowpass=${LP_HZ} Hz) ==="
"$PY" -u 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --eps "$EPS" \
    --lowpass-hz "$LP_HZ" \
    2>&1 | tee -a "$LOG_DIR/waveform_lp${LP_HZ}_attack.log"

echo
echo "=== [$(ts)] STAGE 2: MP3+AAC codec sidecar eval ==="
"$PY" -u 0_all_combined/scripts/eval_mp3_aac_bundle.py \
    --bundle-dir "$BUNDLE" --target-model qwen2_audio \
    2>&1 | tee -a "$LOG_DIR/waveform_lp${LP_HZ}_sidecar.log"

echo
echo "=== [$(ts)] WAVEFORM-LP${LP_HZ} DONE ==="
ls "$BUNDLE/audio/"*_attacked.wav 2>/dev/null | wc -l \
    | xargs -I{} echo "  attacked wavs: {}"
ls "$BUNDLE/audio/"*_codecs.json 2>/dev/null | wc -l \
    | xargs -I{} echo "  codec sidecars: {}"
