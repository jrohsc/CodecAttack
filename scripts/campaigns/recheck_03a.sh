#!/bin/bash
# Re-run channel inference for the 4 existing 03a bundles after the regex +
# carrier_map env-var fixes. Old eval rows already stripped from results.jsonl.
# Adv wavs are correct (attack succeeded) — only re-running model inference.
set -euo pipefail

cd ${REPO_ROOT}/0_all_combined

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
PP="src:${REPO_ROOT}"
SCN=scenarios/03a_ai_detection_bypass
RES=results/03a_ai_detection_bypass
CMAP=data/music/03a_ai_detection_bypass/carrier_to_target.json

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_chan() {
  local tag="$1"; local py="$2"; local model="$3"; local out="$4"
  local log="scripts/recheck_${tag}.log"
  local env_bin="$(dirname "$py")"
  echo "[recheck $(ts)] START ${tag}" | tee -a scripts/recheck_03a.master.log
  env PATH="${env_bin}:${PATH}" PYTHONPATH="${PP}" \
      AUDIO_AGENT_BENCH_CARRIER_MAP="$CMAP" \
    "$py" -u -m audio_agent_bench run "$SCN" --model "$model" --out "$out" \
    > "${log}" 2>&1 || echo "[recheck $(ts)] ${tag} EXIT=$?" | tee -a scripts/recheck_03a.master.log
  echo "[recheck $(ts)] DONE  ${tag}" | tee -a scripts/recheck_03a.master.log
}

# qwen2 first (faster)
run_chan 03a_qwen2_eps0.5 $CA  qwen2_audio    $RES/qwen2_audio/eps_0.5
run_chan 03a_qwen2_eps1.0 $CA  qwen2_audio    $RES/qwen2_audio/eps_1.0
# AF3 (slower)
run_chan 03a_af3_eps0.5   $AF3 audio_flamingo $RES/audio_flamingo/eps_0.5
run_chan 03a_af3_eps1.0   $AF3 audio_flamingo $RES/audio_flamingo/eps_1.0

echo "[recheck $(ts)] ALL 03a RE-CHECKS COMPLETE" | tee -a scripts/recheck_03a.master.log
"$CA" -u scripts/rollup_results.py >> scripts/recheck_03a.master.log 2>&1 || true
echo "[recheck $(ts)] ROLLUP REFRESHED" | tee -a scripts/recheck_03a.master.log
