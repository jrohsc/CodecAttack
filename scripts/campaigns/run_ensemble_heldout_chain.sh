#!/usr/bin/env bash
# Finish the ENSEMBLE-VICTIM transfer probe (AC Q1): score the 180 already-crafted
# ensemble wavs on (1) the HELD-OUT model AF3 -- the point of the probe -- and
# (2) the second ensemble member Omni, which the attack never scored separately.
# Inference only, no optimization. Sequential so the two models never share the GPU.
set -u

PROJ=${REPO_ROOT}
cd "$PROJ"

BUNDLE=results/benchmark_ensemble_transcribe_eps10.0_jazz_1
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python
EVAL=0_all_combined/scripts/ensemble_heldout_eval.py

echo "=== [$(date)] HELD-OUT: audio_flamingo (n=180) ==="
"$AF3" -u "$EVAL" --bundle-dir "$BUNDLE" \
    --target-model audio_flamingo --out-name audio_flamingo_HELDOUT \
    2>/dev/null | grep -v "Loading weights"

echo
echo "=== [$(date)] MEMBER 2: qwen25_omni (n=180) ==="
"$OMNI" -u "$EVAL" --bundle-dir "$BUNDLE" \
    --target-model qwen25_omni --out-name qwen25_omni_MEMBER \
    2>/dev/null | grep -v "Loading weights"

echo
echo "=== [$(date)] DONE ==="
