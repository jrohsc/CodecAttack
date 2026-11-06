#!/bin/bash
# Extend the saved 224-target ensemble attack to the full paper delivery grid.
# Existing per-clip results are reused; only missing channels are inferred.
set -euo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
ENVROOT=${WORK_ROOT}/miniconda3/envs
export PATH="$ENVROOT/codec-attack/bin:$PATH"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1
export KIMI_INFER_ROOT=${WORK_ROOT}/audio_benign_finetuning/Kimi-Audio
export PYTHONPATH="$KIMI_INFER_ROOT:${PYTHONPATH:-}"
PREFIX=benchmark_ensemble3_opuseot_transcribe_eps10.0_jazz_1_bestpostcodec_steps600
CHANNELS=opus16,opus24,opus32,opus64,opus128,opus192,mp3_64,mp3_96,mp3_128,mp3_192,aac_64,aac_96,aac_128,aac_192,dac,encodec
for SPEC in 'qwen2_audio qwen2_MEMBER codec-attack' \
            'qwen25_omni omni7b_MEMBER qwen-omni' \
            'audio_flamingo af3_MEMBER flamingo3' \
            'qwen25_omni_3b omni3b_HELDOUT qwen-omni' \
            'kimi_audio kimi_HELDOUT_100tok kimi-audio' \
            'phi4 phi4_HELDOUT audio-llm'; do
  read -r MODEL OUT ENV <<< "$SPEC"
  for SHARD in 0 1 2 3; do
    B="results/${PREFIX}_shard${SHARD}of4"
    [[ $(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l) -eq 56 ]]
    [[ $(find "$B/neural_roundtrip" -maxdepth 1 -name '*.wav' | wc -l) -eq 112 ]]
    EXTRA=()
    [[ "$MODEL" == kimi_audio ]] && EXTRA=(--max-new-tokens 100)
    PY_CMD=("$ENVROOT/$ENV/bin/python")
    if [[ "$ENV" == qwen-omni ]]; then
      # This env gets its pure-Python tqdm from the user site. The other envs
      # keep PYTHONNOUSERSITE=1 to avoid conflicting user-installed PyTorch.
      PY_CMD=(env -u PYTHONNOUSERSITE "$ENVROOT/$ENV/bin/python")
    fi
    echo "$(date -u +%FT%TZ) full grid $MODEL shard $SHARD" >&2
    "${PY_CMD[@]}" -u 0_all_combined/scripts/ensemble_heldout_eval.py \
      --bundle-dir "$B" --target-model "$MODEL" --out-name "$OUT" \
      --channels "$CHANNELS" --resume-missing-channels \
      --neural-audio-dir "$B/neural_roundtrip" "${EXTRA[@]}" \
      > "logs/fullgrid_${MODEL}_shard${SHARD}.log" 2>&1
  done
done
"$ENVROOT/codec-attack/bin/python" \
  0_all_combined/n50_latent_vs_waveform/aggregate_ensemble3_full_codec_grid.py
