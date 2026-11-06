#!/bin/bash
# Phase 1+1.5 sweep: channel-eval on existing S3 adv wavs.
# Resumable: audio_agent_bench.runner keys on (channel_tag, adv_stem).
# Phase 2 (fresh attacks for 03b AF3 eps_1.0 pair013-019) handled separately.
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA_PY=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3_PY=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
PP="src:${REPO_ROOT}"

SCN=scenarios
RES=results
CMAP_03A=data/music/03a_ai_detection_bypass/carrier_to_target.json
CMAP_03B=data/music/03b_copyright_bypass/carrier_to_target.json

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_chan() {
  local tag="$1"; local py="$2"; local scn_dir="$3"; local model="$4"; local out="$5"; local cmap="$6"
  local log="scripts/phase1_${tag}.log"
  local env_bin="$(dirname "$py")"
  echo "[phase1 $(ts)] START ${tag}"
  env PATH="${env_bin}:${PATH}" PYTHONPATH="${PP}" AUDIO_AGENT_BENCH_CARRIER_MAP="$cmap" \
    "$py" -u -m audio_agent_bench run "$scn_dir" --model "$model" --out "$out" \
    >> "${log}" 2>&1
  echo "[phase1 $(ts)] DONE  ${tag}"
}

# ---- Phase 1: qwen2_audio channel eval on 40/40/20/20 existing wavs ----
run_chan 03a_qwen2_eps0.5 $CA_PY  $SCN/03a_ai_detection_bypass qwen2_audio     $RES/03a_ai_detection_bypass/qwen2_audio/eps_0.5     $CMAP_03A
run_chan 03a_qwen2_eps1.0 $CA_PY  $SCN/03a_ai_detection_bypass qwen2_audio     $RES/03a_ai_detection_bypass/qwen2_audio/eps_1.0     $CMAP_03A
run_chan 03b_qwen2_eps0.5 $CA_PY  $SCN/03b_copyright_bypass    qwen2_audio     $RES/03b_copyright_bypass/qwen2_audio/eps_0.5        $CMAP_03B
run_chan 03b_qwen2_eps1.0 $CA_PY  $SCN/03b_copyright_bypass    qwen2_audio     $RES/03b_copyright_bypass/qwen2_audio/eps_1.0        $CMAP_03B

# ---- Phase 1.5: AF3 resumable evals ----
# 03b AF3 eps_0.5: already has clean/opus16k/opus24k/opus32k -> fill 64k/128k/192k/g722_64k
run_chan 03b_af3_eps0.5   $AF3_PY $SCN/03b_copyright_bypass    audio_flamingo  $RES/03b_copyright_bypass/audio_flamingo/eps_0.5     $CMAP_03B
# 03b AF3 eps_1.0: eval on 13 existing pairs (pair000-012); missing 013-019 handled in Phase 2
run_chan 03b_af3_eps1.0_partial $AF3_PY $SCN/03b_copyright_bypass audio_flamingo $RES/03b_copyright_bypass/audio_flamingo/eps_1.0 $CMAP_03B

echo "[phase1 $(ts)] SWEEP COMPLETE"
