#!/bin/bash
# Re-run S2 bench eval for AF3 + Omni after 2026-04-17 adapter prompt-fix
# (dropped "Listen to the audio and respond." suffix; it misaligned with
# attack-time prompt and collapsed success to 0%). Attacks are unchanged;
# adv wavs already on disk. This only re-runs the eval pass.
set -euo pipefail
cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

SCEN=scenarios/02_interview_screening
export AUDIO_AGENT_BENCH_CARRIER_MAP=data/speech/02_interview/carrier_to_target.json
export PYTHONPATH="src:${REPO_ROOT}:${PYTHONPATH:-}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run() {
  local tag="$1"; shift
  local py="$1"; shift
  local model="$1"; shift
  local eps="$1"; shift
  local bundle="results/02_interview_screening/${model}/eps_${eps}"
  local log="scripts/re_eval_s2_${tag}.log"
  echo "[sweep $(ts)] START ${tag}"
  # Prepend env bin to PATH for ffmpeg (g722 / g711 / chain channels)
  local envbin=$(dirname "$py")
  PATH="${envbin}:$PATH" "$py" -m audio_agent_bench run "$SCEN" \
      --model "$model" --out "$bundle" > "$log" 2>&1
  echo "[sweep $(ts)] DONE  ${tag}  -> $bundle/results.jsonl"
}

run af3_eps0.5  "$AF3"  audio_flamingo  0.5
run af3_eps1.0  "$AF3"  audio_flamingo  1.0
run omni_eps0.5 "$OMNI" qwen25_omni     0.5
run omni_eps1.0 "$OMNI" qwen25_omni     1.0

echo "[sweep $(ts)] SWEEP COMPLETE — refreshing rollups"
"$CA" scripts/rollup_results.py
