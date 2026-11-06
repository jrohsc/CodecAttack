# Runbook — SNR-matched waveform controls and the ≤4 kHz control

Four arms, one evaluation:

| arm | name in `--method` | what it is |
|---|---|---|
| A | `latent_codecattack` | existing latent CodecAttack bundles, re-measured and re-scored through this pipeline. Not re-optimized. |
| B | `original_waveform` | the historical waveform baseline, unchanged recipe (S1: 20 % warmup, codec on every post-warmup step; S3a: no codec training), kept reproducible. |
| C | `tuned_waveform` | unrestricted waveform, latent's schedule (30 % warmup, alternating clean/Opus-BPDA), learning rate swept, budget calibrated to the latent SNR. |
| D | `bandlimited_4khz` | as C, with the perturbation confined to ≤4 kHz throughout optimization, budget calibrated independently. |

Everything runs in the `codec-attack` env. Prefix every command with:

```bash
CA=${WORK_ROOT}/miniconda3/envs/codec-attack
export PATH="$CA/bin:$PATH" HF_HUB_OFFLINE=1
cd ${REPO_ROOT}/0_all_combined
export PYTHONPATH="$PWD"
```

## 0. Checks that need no GPU

```bash
python -m pytest reviewer_controls/test_controls.py -q
python -m reviewer_controls.audit_audio_path \
  --output reviewer_control_outputs/audit/audio_path_audit.json
```

The second writes the Opus/MP3/AAC alignment table and the PCM round-trip
probe. `"conclusion": "no compensation required"` means the decoders are
already sample-aligned — see AUDIT.md §B1. Do not add delay compensation.

## 1. Measure the reference arm's realized SNR (arm A, CPU only)

```bash
python -m reviewer_controls.measure_reference_snr --scenario s1 \
  --method latent_codecattack \
  --bundle results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0 \
  --output reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl

python -m reviewer_controls.measure_reference_snr --scenario s3a \
  --method latent_codecattack \
  --bundle results_codec_robust/03a_ai_detection_bypass/qwen2_audio/eps_1.0_multibitrate \
  --output reviewer_control_outputs/snr/s3a_latent_eps1.0.jsonl
```

`*.summary.json` lists any example that is missing or ambiguous in the bundle;
those are reported as unavailable rather than dropped silently.

## 2. Learning-rate sweep on development examples (GPU)

α/ε ∈ {0.02, 0.05, 0.1, 0.2}. 0.2 is the existing setting (the drivers use
`alpha = eps/5`), so it is included unchanged and the grid extends downward
only — Adam's per-coordinate normalization means α is close to a pure step size
in waveform amplitude, and α/ε = 0.2 lets the optimizer cross the whole ε-ball
in five steps, which is the end of the range worth testing, not the middle.

```bash
OUT=$PWD/reviewer_control_outputs/pilot_s1_seed1 ./reviewer_controls/run_pilot.sh sweep
```

Full split instead of the pilot subset: drop `--pilot` from `run_pilot.sh`'s
`run()`, or call `run_attack` directly:

```bash
for ar in 0.02 0.05 0.1 0.2; do
  for m in tuned_waveform bandlimited_4khz; do
    eps=$([ $m = bandlimited_4khz ] && echo 0.166 || echo 0.06082)
    python -m reviewer_controls.run_attack --scenario s1 --split development \
      --method $m --epsilon $eps --alpha-ratio $ar --steps 1000 --seed 1 \
      --resume --output-root reviewer_control_outputs/full_s1
  done
done
```

## 3. Score the development runs on training codecs only, then freeze the LR

```bash
python -m reviewer_controls.evaluate \
  --manifest reviewer_control_outputs/pilot_s1_seed1/s1/development/*/*/manifest.jsonl \
  --channels clean opus16 opus24 opus32 opus64 opus128 \
  --output reviewer_control_outputs/pilot_s1_seed1/eval/development.jsonl --resume

python -m reviewer_controls.select_hparams \
  --development-eval reviewer_control_outputs/pilot_s1_seed1/eval/development.jsonl \
  --output reviewer_control_outputs/pilot_s1_seed1/frozen/learning_rate.json
```

`select_hparams.py` refuses to run if a non-development example reaches it, and
its objective only touches clean + Opus 16/24/32/64/128. It does not choose the
budget; §4 does, from distortion alone.

## 4. Solve each arm's budget (no GPU, no attack outcome)

```bash
for m in tuned_waveform bandlimited_4khz; do
  python -m reviewer_controls.calibrate_epsilon \
    --probe reviewer_control_outputs/pilot_s1_seed1/s1/development/$m/<frozen-α-cell>/manifest.jsonl \
    --reference reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl \
    --mode per_example \
    --output reviewer_control_outputs/pilot_s1_seed1/frozen/epsilon_$m.json
done
```

Uses SNR(ε) = SNR(ε₀) − 20 log₁₀(ε/ε₀), which both parameterizations obey (the
unrestricted arm through a pointwise clamp, the ≤4 kHz arm through one global
scale). Verified empirically: 0.06082 → 5.35 dB and 0.075 → 3.56 dB differ by
1.79 dB against a predicted 1.82 dB.

Per-example is the default because latent realized SNR has SD 1.73 dB across S1
examples, so no single global ε can match it example by example
(AUDIT.md §A6). `--mode global` reproduces the manuscript's single-ε′ style.

## 5. Verify the match, then run the held-out evaluation examples

```bash
AR_C=<frozen> AR_D=<frozen> OUT=$PWD/reviewer_control_outputs/pilot_s1_seed1 \
  ./reviewer_controls/run_pilot.sh verify

python -m reviewer_controls.calibrate_epsilon \
  --probe  .../development/tuned_waveform/<probe-cell>/manifest.jsonl \
  --reference reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl \
  --verify .../development/tuned_waveform/<snrmatched-cell>/manifest.jsonl \
  --output reviewer_control_outputs/pilot_s1_seed1/frozen/epsilon_tuned_waveform.verified.json

AR_C=<frozen> AR_D=<frozen> OUT=$PWD/reviewer_control_outputs/pilot_s1_seed1 \
  ./reviewer_controls/run_pilot.sh baseline
AR_C=<frozen> AR_D=<frozen> OUT=$PWD/reviewer_control_outputs/pilot_s1_seed1 \
  ./reviewer_controls/run_pilot.sh eval
```

**The learning rate and the budgets are now written to `frozen/` and must not be
revisited.** Only after this point does anything touch MP3 or AAC.

## 6. Score the evaluation split on all eight channels and build the table

```bash
python -m reviewer_controls.evaluate \
  --manifest reviewer_control_outputs/pilot_s1_seed1/s1/evaluation/*/*/manifest.jsonl \
             reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl \
  --output reviewer_control_outputs/pilot_s1_seed1/eval/evaluation.jsonl --resume

python -m reviewer_controls.build_report \
  --records reviewer_control_outputs/pilot_s1_seed1/eval/evaluation.jsonl \
  --reference-method latent_codecattack \
  --output-json reviewer_control_outputs/pilot_s1_seed1/report.json \
  --output-markdown reviewer_control_outputs/pilot_s1_seed1/report.md
```

Arm A is evaluated from its existing WAVs through the same delivered-PCM path;
it is never re-optimized. Restrict the latent manifest to the evaluation split
with `--split evaluation` in step 1 if you do not want its development rows in
the table.

## 7. Final evaluation configuration (seeds 1, 7, 42)

Freeze the LR and the budgets once, at seed 1, then rerun only steps 5–6 per
seed:

```bash
for s in 1 7 42; do
  for m in tuned_waveform bandlimited_4khz; do
    eps=$([ $m = bandlimited_4khz ] && echo 0.166 || echo 0.06082)
    ar=$(python -c "import json;print(json.load(open('reviewer_control_outputs/full_s1/frozen/learning_rate.json'))['selected']['s1:$m']['alpha_over_epsilon'])")
    python -m reviewer_controls.run_attack --scenario s1 --split evaluation \
      --method $m --epsilon $eps --alpha-ratio $ar --steps 1000 --seed $s \
      --epsilon-map reviewer_control_outputs/full_s1/frozen/epsilon_$m.json \
      --run-tag snrmatched --resume --output-root reviewer_control_outputs/full_s1
  done
done
```

With zeros initialization the seeds differ only through the EoT bitrate draw.
Add `--rand-init` to also vary the starting point; if you do, use it for every
arm in the comparison, and note that the published latent bundles
`eps_1.0_seed{1,7,42}` must be checked for the same flag before they are pooled.

## Measured cost

| quantity | measured | how |
|---|---|---|
| S1 1000-step attack, A100-40GB | **5.3 min** (0.319 s/step, 200-step probe) | `reviewer_control_outputs/TIMING` |
| S1 carrier length | 10.0 s (floor of `max(10, min(25, src))`) | — |
| S3a carrier length | 25.0 s | — |
| 8-channel scoring, S1 example | ~20 s | codec round trips measured; generate not separately timed |

**Scope: S1 only.** S3a is implemented and its config is frozen
(`config.json` `scenarios.s3a`), but it is not being run — its carriers are
25 s against S1's 10 s, so every stage costs about 2.5x. Nothing below includes it.

Per-run cost is flat at **5.1-5.2 min** across all eight swept cells
(both arms, all four learning rates), so the estimate is a straight multiply.

| stage | runs | GPU-h | seeds |
|---|---|---|---|
| learning-rate sweep, 10 dev examples x 4 alpha x 2 arms | 80 | 6.9 | 1 |
| distortion probe on the 40 evaluation examples x 2 arms | 80 | 6.9 | 1 |
| calibrated evaluation runs, 40 examples x 2 arms | 240 | 20.8 | 1, 7, 42 |
| scoring, development, 6 training channels | - | 0.3 | 1 |
| scoring, evaluation, 4 arms x 8 channels | - | 2.7 | 1, 7, 42 |
| **total, S1, four arms, three seeds** | | **~37.6** | |

Arms A and B contribute no optimization cost: both reuse published bundles
(`eps_1.0_seed{1,7,42}` and `eps_0.06082_norm_seed{1,7,42}`, all 50/50 complete).
Only their scoring is counted above.

Add roughly 5.8 GPU-h per extra calibration iteration on the evaluation split
if the first closed-form budget solve leaves a residual outside 1 dB.
