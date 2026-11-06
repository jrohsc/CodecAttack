#!/usr/bin/env bash
# Unattended S1 pilot chain. Waits for the learning-rate sweep, then:
#   score dev (training codecs only) -> freeze LR -> solve budgets -> verify on
#   dev -> probe eval -> solve eval budgets -> run eval -> score all four arms.
# Held-out MP3/AAC are scored only in the final step, after frozen/ is written.
set -euo pipefail
CA=${WORK_ROOT}/miniconda3/envs/codec-attack
ROOT=${REPO_ROOT}/0_all_combined
OUT=$ROOT/reviewer_control_outputs/pilot_s1_seed1
export PATH="$CA/bin:$PATH" PYTHONPATH="$ROOT" HF_HUB_OFFLINE=1
cd "$ROOT"
mkdir -p "$OUT/frozen" "$OUT/eval" "$OUT/logs"
PY="$CA/bin/python"
say () { echo "[chain $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/logs/chain.log"; }

say "waiting for sweep (32 runs)"
while [ "$(find "$OUT" -name manifest.jsonl -exec cat {} + 2>/dev/null | wc -l)" -lt 32 ]; do sleep 60; done
say "sweep complete"

say "scoring development split on training codecs only"
$PY -m reviewer_controls.evaluate \
  --manifest "$OUT"/s1/development/*/*/manifest.jsonl \
  --channels clean opus16 opus24 opus32 opus64 opus128 \
  --output "$OUT/eval/development.jsonl" --resume >> "$OUT/logs/score_dev.log" 2>&1
say "freezing learning rate"
$PY -m reviewer_controls.select_hparams \
  --development-eval "$OUT/eval/development.jsonl" \
  --output "$OUT/frozen/learning_rate.json" >> "$OUT/logs/chain.log" 2>&1

AR_C=$($PY -c "import json;print(json.load(open('$OUT/frozen/learning_rate.json'))['selected']['s1:tuned_waveform']['alpha_over_epsilon'])")
AR_D=$($PY -c "import json;print(json.load(open('$OUT/frozen/learning_rate.json'))['selected']['s1:bandlimited_4khz']['alpha_over_epsilon'])")
say "frozen alpha/epsilon: tuned=$AR_C  bandlimited=$AR_D"

cell () { printf "%s/s1/%s/%s/eps%s_ar%s_steps1000_seed1_pilot%s/manifest.jsonl" \
          "$OUT" "$2" "$1" "$3" "$4" "${5:-}"; }
EPS_C=0.06082; EPS_D=0.166
REF=reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl

for arm in tuned_waveform bandlimited_4khz; do
  if [ "$arm" = tuned_waveform ]; then eps=$EPS_C; ar=$AR_C; else eps=$EPS_D; ar=$AR_D; fi
  say "solving development budget for $arm from the alpha/eps=$ar probe"
  $PY -m reviewer_controls.calibrate_epsilon \
    --probe "$(cell "$arm" development "$eps" "$ar")" --reference "$REF" \
    --mode per_example --output "$OUT/frozen/epsilon_${arm}.dev.json" >> "$OUT/logs/chain.log" 2>&1
done

say "verifying the SNR match on development examples"
for arm in tuned_waveform bandlimited_4khz; do
  if [ "$arm" = tuned_waveform ]; then eps=$EPS_C; ar=$AR_C; else eps=$EPS_D; ar=$AR_D; fi
  $PY -m reviewer_controls.run_attack --scenario s1 --split development --method "$arm" \
    --epsilon "$eps" --alpha-ratio "$ar" --steps 1000 --seed 1 --pilot --resume \
    --epsilon-map "$OUT/frozen/epsilon_${arm}.dev.json" --run-tag snrmatched \
    --output-root "$OUT" >> "$OUT/logs/verify_${arm}.log" 2>&1
  $PY -m reviewer_controls.calibrate_epsilon \
    --probe "$(cell "$arm" development "$eps" "$ar")" --reference "$REF" --mode per_example \
    --verify "$(cell "$arm" development "$eps" "$ar" _snrmatched)" \
    --output "$OUT/frozen/epsilon_${arm}.dev.verified.json" >> "$OUT/logs/chain.log" 2>&1
  say "$arm dev verification: $($PY -c "
import json;v=json.load(open('$OUT/frozen/epsilon_${arm}.dev.verified.json'))['verification']
print(f\"mean {v['mean_db']:+.2f} dB, max|d| {v['max_abs_db']:.2f}, within 1 dB {v['n_within_1db']}/{v['n']}\")")"
done

say "distortion probe on the evaluation split (outcomes never scored)"
for arm in tuned_waveform bandlimited_4khz; do
  if [ "$arm" = tuned_waveform ]; then eps=$EPS_C; ar=$AR_C; else eps=$EPS_D; ar=$AR_D; fi
  $PY -m reviewer_controls.run_attack --scenario s1 --split evaluation --method "$arm" \
    --epsilon "$eps" --alpha-ratio "$ar" --steps 1000 --seed 1 --pilot --resume \
    --run-tag probe --output-root "$OUT" >> "$OUT/logs/probe_${arm}.log" 2>&1
done

# Fixed-point budget solve. SNR(eps) is only locally log-linear -- for the
# <=4 kHz arm the valid-range projection can bind and flatten it -- so each
# round recalibrates from the most recent realized SNR rather than trusting a
# single extrapolation. Distortion only; no outcome is scored here.
say "solving evaluation budgets (iterative, tolerance 1 dB, max 3 rounds)"
for arm in tuned_waveform bandlimited_4khz; do
  if [ "$arm" = tuned_waveform ]; then eps=$EPS_C; ar=$AR_C; else eps=$EPS_D; ar=$AR_D; fi
  src_tag=_probe
  final_tag=""
  for round in 1 2 3; do
    $PY -m reviewer_controls.calibrate_epsilon \
      --probe "$(cell "$arm" evaluation "$eps" "$ar" "$src_tag")" --reference "$REF" \
      --mode per_example --output "$OUT/frozen/epsilon_${arm}.r${round}.json" \
      >> "$OUT/logs/chain.log" 2>&1
    tag="snrmatched${round}"
    $PY -m reviewer_controls.run_attack --scenario s1 --split evaluation --method "$arm" \
      --epsilon "$eps" --alpha-ratio "$ar" --steps 1000 --seed 1 --pilot --resume \
      --epsilon-map "$OUT/frozen/epsilon_${arm}.r${round}.json" --run-tag "$tag" \
      --output-root "$OUT" >> "$OUT/logs/eval_${arm}.log" 2>&1
    $PY -m reviewer_controls.calibrate_epsilon \
      --probe "$(cell "$arm" evaluation "$eps" "$ar" "$src_tag")" --reference "$REF" --mode per_example \
      --verify "$(cell "$arm" evaluation "$eps" "$ar" "_$tag")" \
      --output "$OUT/frozen/epsilon_${arm}.r${round}.verified.json" >> "$OUT/logs/chain.log" 2>&1
    res=$($PY -c "
import json;v=json.load(open('$OUT/frozen/epsilon_${arm}.r${round}.verified.json'))['verification']
print(f\"{v['max_abs_db']:.3f} {v['n_within_1db']}/{v['n']} {v['mean_db']:+.2f}\")")
    say "$arm round $round: max|d| $(echo $res | cut -d' ' -f1) dB, within 1 dB $(echo $res | cut -d' ' -f2), mean $(echo $res | cut -d' ' -f3) dB"
    final_tag="$tag"
    ok=$($PY -c "
import json;v=json.load(open('$OUT/frozen/epsilon_${arm}.r${round}.verified.json'))['verification']
print('1' if v['all_within_1db'] else '0')")
    if [ "$ok" = "1" ]; then say "$arm converged at round $round"; break; fi
    src_tag="_$tag"
  done
  cp "$OUT/frozen/epsilon_${arm}.r"*".json" "$OUT/frozen/" 2>/dev/null || true
  echo "$final_tag" > "$OUT/frozen/final_tag_${arm}.txt"
  say "$arm final manifest tag: $final_tag"
done

say "measuring the two reused arms on the same pilot examples"
$PY -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation --pilot \
  --method latent_codecattack \
  --bundle results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0 \
  --output "$OUT/eval/armA_latent.jsonl" >> "$OUT/logs/chain.log" 2>&1
$PY -m reviewer_controls.measure_reference_snr --scenario s1 --split evaluation --pilot \
  --method original_waveform_published \
  --bundle results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.06082_norm_seed1 \
  --output "$OUT/eval/armB_published.jsonl" >> "$OUT/logs/chain.log" 2>&1

say "scoring all four arms on all eight delivery channels"
$PY -m reviewer_controls.evaluate \
  --manifest $(for a in tuned_waveform bandlimited_4khz; do \
        t=$(cat "$OUT/frozen/final_tag_$a.txt"); \
        if [ "$a" = tuned_waveform ]; then e=$EPS_C; r=$AR_C; else e=$EPS_D; r=$AR_D; fi; \
        printf "%s " "$OUT/s1/evaluation/$a/eps${e}_ar${r}_steps1000_seed1_pilot_${t}/manifest.jsonl"; done) \
             "$OUT/eval/armA_latent.jsonl" "$OUT/eval/armB_published.jsonl" \
  --output "$OUT/eval/evaluation.jsonl" --resume >> "$OUT/logs/score_eval.log" 2>&1

$PY -m reviewer_controls.build_report --records "$OUT/eval/evaluation.jsonl" \
  --reference-method latent_codecattack \
  --output-json "$OUT/report.json" --output-markdown "$OUT/report.md" >> "$OUT/logs/chain.log" 2>&1
say "PILOT COMPLETE -> $OUT/report.md"
