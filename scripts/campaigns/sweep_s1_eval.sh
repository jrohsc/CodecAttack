#!/usr/bin/env bash
# Complete S1 eval sweep — 5 remaining bundles (qwen2 eps=0.5 already done).
# Uses env-specific python directly so PYTHONPATH propagates (conda run strips it).
set -uo pipefail
cd "$(dirname "$0")/.."

CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
AF3=${WORK_ROOT}/miniconda3/envs/flamingo3/bin/python
OMNI=${WORK_ROOT}/miniconda3/envs/qwen-omni/bin/python

SCEN_DIR="scenarios/01_finance_voice_agent"
RESULTS_BASE="results/01_finance_voice_agent"
CARRIER_MAP="data/speech/01_finance/carrier_to_target.json"
LOG_DIR="scripts"

export AUDIO_AGENT_BENCH_CARRIER_MAP="$CARRIER_MAP"
export PYTHONPATH="$PWD/src:$PWD/..:${PYTHONPATH:-}"

run_bundle() {
  local py="$1"; local model="$2"; local eps="$3"; local tag="$4"
  local bundle="$RESULTS_BASE/$model/eps_$eps"
  local log="$LOG_DIR/run_s1_eval_${tag}_eps${eps}.log"
  local env_bin="$(dirname "$py")"
  if [[ ! -d "$bundle/audio" ]]; then
    echo "[skip] no adv wavs at $bundle/audio"
    return 0
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START eval $tag eps=$eps" | tee -a scripts/sweep_s1_eval.log
  PATH="$env_bin:$PATH" "$py" -u -m audio_agent_bench run "$SCEN_DIR" \
    --model "$model" --out "$bundle" \
    > "$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  eval $tag eps=$eps rc=$rc" | tee -a scripts/sweep_s1_eval.log
  return $rc
}

run_bundle "$CA"   qwen2_audio   1.0 qwen2
run_bundle "$AF3"  audio_flamingo 0.5 af3
run_bundle "$AF3"  audio_flamingo 1.0 af3
run_bundle "$OMNI" qwen25_omni    0.5 omni
run_bundle "$OMNI" qwen25_omni    1.0 omni

echo "[$(date '+%Y-%m-%d %H:%M:%S')] summarize + rollup" | tee -a scripts/sweep_s1_eval.log
"$CA" scripts/summarize.py --scenario 01_finance_voice_agent || true
"$CA" scripts/rollup_results.py || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1 eval sweep done" | tee -a scripts/sweep_s1_eval.log
