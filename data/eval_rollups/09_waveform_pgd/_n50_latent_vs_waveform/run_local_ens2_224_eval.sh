#!/bin/bash
# Evaluate the three already-crafted 224-target Omni-7B + AF3 Opus-EoT bundles.
set -euo pipefail
REPO=${REPO_ROOT}
cd "$REPO"
export PATH="$REPO/../../miniconda3/envs/codec-attack/bin:$PATH"
export HF_HUB_OFFLINE=1 PYTHONUTF8=1 PYTHONNOUSERSITE=1
ENVROOT=${WORK_ROOT}/miniconda3/envs
EVAL=0_all_combined/scripts/ensemble_heldout_eval.py
CHANNELS=opus64,opus128,mp3_64,mp3_128,aac_64,aac_128
for SEED in 1 7 42; do
  B="results/benchmark_ens2_omni_af3_opuseot_transcribe_eps10.0_jazz_1_seed${SEED}"
  [[ $(find "$B/audio" -maxdepth 1 -name '*.wav' | wc -l) -eq 224 ]]
  for SPEC in \
      "qwen-omni qwen25_omni omni7b_MEMBER" \
      "flamingo3 audio_flamingo af3_MEMBER" \
      "codec-attack qwen2_audio qwen2_HELDOUT" \
      "qwen-omni qwen25_omni_3b omni3b_HELDOUT"; do
    read -r ENV MODEL NAME <<< "$SPEC"
    echo "$(date -u +%FT%TZ) seed=$SEED $MODEL"
    "$ENVROOT/$ENV/bin/python" -u "$EVAL" \
      --bundle-dir "$B" --target-model "$MODEL" --out-name "$NAME" \
      --channels "$CHANNELS" > "logs/local_ens2_seed${SEED}_${NAME}.log" 2>&1
    "$ENVROOT/codec-attack/bin/python" - "$B/$NAME/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
assert d['stats']['n']==224 and d['stats']['n_skipped']==0, sys.argv[1]
print(sys.argv[1], {k:v['exact_pct'] for k,v in d['stats']['channels'].items()})
PY
  done
done
"$ENVROOT/codec-attack/bin/python" \
  0_all_combined/n50_latent_vs_waveform/aggregate_ens2_224_seeds.py
