#!/usr/bin/env bash
# Bounded S1 pilot on one GPU, in the order the design requires:
#   1. learning-rate sweep on development examples, training codecs only
#   2. freeze the learning rate
#   3. solve each arm's budget in closed form from distortion alone
#   4. verify the realized SNR match on development examples
#   5. run the frozen configuration on the held-out evaluation examples
# Held-out MP3/AAC results are produced only in step 6 (scoring), after every
# tuning choice is already written to disk.
set -euo pipefail

CA=${WORK_ROOT}/miniconda3/envs/codec-attack
ROOT=${REPO_ROOT}/0_all_combined
OUT="${OUT:-$ROOT/reviewer_control_outputs/pilot_s1_seed1}"
LOGS="$OUT/logs"
STEPS="${STEPS:-1000}"
SEED="${SEED:-1}"
SCEN="${SCEN:-s1}"
export PATH="$CA/bin:$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
mkdir -p "$LOGS"
cd "$ROOT"

run () {  # run <method> <split> <eps> <alpha_ratio> [extra...]
  local m=$1 sp=$2 eps=$3 ar=$4; shift 4
  local tag="${SCEN}_${sp}_${m}_eps${eps}_ar${ar}"
  echo "=== $tag" | tee -a "$LOGS/driver.log"
  "$CA/bin/python" -m reviewer_controls.run_attack \
    --scenario "$SCEN" --split "$sp" --method "$m" \
    --epsilon "$eps" --alpha-ratio "$ar" --steps "$STEPS" --seed "$SEED" \
    --pilot --resume --output-root "$OUT" "$@" >> "$LOGS/$tag.log" 2>&1
}

EPS_C=${EPS_C:-0.06082}     # probe budget, unrestricted tuned arm
EPS_D=${EPS_D:-0.166}       # probe budget, <=4 kHz arm

case "${1:-all}" in
  sweep)
    for ar in 0.02 0.05 0.1 0.2; do
      run tuned_waveform    development "$EPS_C" "$ar"
      run bandlimited_4khz  development "$EPS_D" "$ar"
    done
    ;;
  probe-eval)
    # Distortion probe on the evaluation split at the probe budget and the
    # frozen learning rate. Its OUTCOMES ARE NEVER SCORED; only the realized
    # SNR is read, to solve each example's budget. Matching SNR per example is
    # a measurement requirement, so using evaluation examples here leaks no
    # label and no delivery-codec result.
    : "${AR_C:?set AR_C to the frozen alpha/epsilon for tuned_waveform}"
    : "${AR_D:?set AR_D to the frozen alpha/epsilon for bandlimited_4khz}"
    run tuned_waveform   evaluation "$EPS_C" "$AR_C" --run-tag probe
    run bandlimited_4khz evaluation "$EPS_D" "$AR_D" --run-tag probe
    ;;
  baseline)
    # Arm B on the evaluation split, historical recipe, historical budget.
    run original_waveform evaluation "$EPS_C" 0.2
    ;;
  eval)
    : "${AR_C:?set AR_C to the frozen alpha/epsilon for tuned_waveform}"
    : "${AR_D:?set AR_D to the frozen alpha/epsilon for bandlimited_4khz}"
    run tuned_waveform   evaluation "$EPS_C" "$AR_C" --epsilon-map "$OUT/frozen/epsilon_tuned_waveform.json" --run-tag snrmatched
    run bandlimited_4khz evaluation "$EPS_D" "$AR_D" --epsilon-map "$OUT/frozen/epsilon_bandlimited_4khz.json" --run-tag snrmatched
    ;;
  verify)
    : "${AR_C:?}"; : "${AR_D:?}"
    run tuned_waveform   development "$EPS_C" "$AR_C" --epsilon-map "$OUT/frozen/epsilon_tuned_waveform.json" --run-tag snrmatched
    run bandlimited_4khz development "$EPS_D" "$AR_D" --epsilon-map "$OUT/frozen/epsilon_bandlimited_4khz.json" --run-tag snrmatched
    ;;
  score-dev)
    # Training codecs only. A held-out codec cannot reach a tuning decision
    # because it is never even scored at this stage.
    mkdir -p "$OUT/eval"
    "$CA/bin/python" -m reviewer_controls.evaluate \
      --manifest "$OUT/$SCEN"/development/*/*/manifest.jsonl \
      --channels clean opus16 opus24 opus32 opus64 opus128 \
      --output "$OUT/eval/development.jsonl" --resume 2>&1 | tee -a "$LOGS/score_dev.log"
    "$CA/bin/python" -m reviewer_controls.select_hparams \
      --development-eval "$OUT/eval/development.jsonl" \
      --output "$OUT/frozen/learning_rate.json"
    ;;
  score-eval)
    mkdir -p "$OUT/eval"
    "$CA/bin/python" -m reviewer_controls.evaluate \
      --manifest "$OUT/$SCEN"/evaluation/*/*/manifest.jsonl ${LATENT_MANIFEST:-} \
      --output "$OUT/eval/evaluation.jsonl" --resume 2>&1 | tee -a "$LOGS/score_eval.log"
    "$CA/bin/python" -m reviewer_controls.build_report \
      --records "$OUT/eval/evaluation.jsonl" \
      --reference-method latent_codecattack \
      --output-json "$OUT/report.json" --output-markdown "$OUT/report.md"
    ;;
  *) echo "usage: $0 {sweep|score-dev|verify|baseline|eval|score-eval}" >&2; exit 2;;
esac
echo "done ${1:-all}" | tee -a "$LOGS/driver.log"
