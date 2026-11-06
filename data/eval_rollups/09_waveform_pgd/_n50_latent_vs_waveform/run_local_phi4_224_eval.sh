#!/bin/bash
# Held-out Phi-4 evaluation on the saved 224-target three-model Opus-EoT audio.
set -euo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
ENVROOT=${WORK_ROOT}/miniconda3/envs
export PATH="$ENVROOT/codec-attack/bin:$PATH"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONNOUSERSITE=1
PREFIX=benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600
for SHARD in 0 1 2 3; do
  B="results/${PREFIX}_shard${SHARD}of4"
  [[ $(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l) -eq 56 ]]
  echo "$(date -u +%FT%TZ) Phi-4 shard $SHARD"
  "$ENVROOT/audio-llm/bin/python" -u 0_all_combined/scripts/ensemble_heldout_eval.py \
    --bundle-dir "$B" --target-model phi4 --out-name phi4_HELDOUT \
    --channels opus64,opus128,mp3_64,mp3_128,aac_64,aac_128 \
    > "logs/local_phi4_224_shard${SHARD}.log" 2>&1
done
"$ENVROOT/codec-attack/bin/python" \
  0_all_combined/n50_latent_vs_waveform/aggregate_extra_heldout.py \
  --prefix "$PREFIX" --out-name phi4_HELDOUT --model phi4
