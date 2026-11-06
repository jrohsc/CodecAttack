# Controlled-comparison protocol audit (2026-09-21)

## Scope and non-expansion rule

This audit is restricted to S1, the already-running N=1/N=2/N=3 ensemble-size
comparison, and the four baseline arms requested by the reviewer. No additional
model or scenario sweep is part of this campaign. Pilot outputs remain under
`reviewer_control_outputs/pilot_s1_seed1/` and are not pooled with the 40-example
held-out evaluation.

## Ensemble-size comparison

The three configurations use the same 50 ordered S1 carrier-target pairs and
the same bank system prompt. Their member sets are nested:

- N=1: `qwen2_audio`
- N=2: `qwen2_audio`, `qwen25_omni`
- N=3: `qwen2_audio`, `qwen25_omni`, `audio_flamingo`

`qwen25_omni_3b` is absent from every `--ensemble-models` argument, from the
optimizer, from the training-loss checkpoint rule, and from hyperparameter
selection. It is loaded only by the dependent evaluation jobs.

The following declared settings are identical across N: latent epsilon 10.0; Adam
learning rate 0.2; random latent initialization at scale 0.01 with seed 1; 1,000
steps; 50 primary-only ensemble-warmup steps; 300 clean channel-warmup steps;
then Opus-BPDA on every remaining step with a uniform bitrate draw from
{16,24,32,64,128}; and minimum normalized training loss among eligible steps
301--1000 as the saved checkpoint. The driver names this
`--best-after-warmup`. The selected step itself is not persisted, which is an
auditability limitation.

Compute is deliberately not matched. Each post-ensemble-warmup step evaluates
all N members, so the planned model forward/backward pairs per example are
1,000 (N=1), 1,950 (N=2), and 2,900 (N=3), or 50,000, 97,500, and 145,000 over
50 examples. Periodic primary-model generations and delivery evaluation are
additional. Final realized GPU time and worker-failure counts are emitted by
`reviewer_controls.build_ensemble_report` after the dependent evaluations.

One implementation-level seed mismatch remains. `--seed 1` controls the latent
initialization through a dedicated generator, but `DifferentiableOpusProxy`
draws the bitrate using PyTorch's process-global RNG and the driver never calls
`torch.manual_seed(args.seed)`. Therefore the three arms use the same EoT
distribution and schedule but not a verifiably identical bitrate realization.
The N=3 shard-1 requeue restarted that unlogged global-RNG sequence after
skipping completed files. This does not invalidate the waveforms as attacks,
but it prevents attributing every N difference specifically to ensemble size.

The jobs are retained as descriptive evidence. The interpretation must be
“adding members under the same EoT distribution and increasing compute,” not a
compute- and randomness-matched ensemble-size effect. The smallest correction
is a dedicated seeded codec generator (reset by stable example ID) followed by
a rerun of only the N=1/N=2/N=3 ladder. That correction is not silently mixed
with the current outputs.

## Development/evaluation separation

The S1 split is by source recording. The ten development carriers are listed in
`reviewer_controls/config.json`; all other 40 carriers are evaluation. S1 has
one pair per carrier recording, so there is no cross-split recording reuse.
Stable IDs encode scenario, original pair index, carrier, and target.

The learning-rate selection was frozen before held-out scoring at
`pilot_s1_seed1/frozen/learning_rate.json`. It used four development examples
and only clean plus Opus {16,24,32,64,128}; MP3 and AAC-LC were not exposed.
Both tuned waveform arms selected alpha/epsilon=0.02. The subsequently launched
ten-example development run is confirmatory and cannot revise this file.

Per-example epsilon calibration is outcome-independent: it reads only realized
carrier-referenced SNR and the latent reference SNR. It never reads model text,
matches, or ASR. The held-out arm uses the already-frozen alpha/epsilon and the
distortion-only epsilon map. No designated evaluation example was used for
learning-rate selection.

The initial held-out distortion probes required larger budgets than the nominal
probe for most examples. For arm D, four of 40 epsilon estimates extrapolate
beyond examples that reached the probe projection bound (calibrated epsilon
range 0.192--0.491); arm C had no such projection-bound extrapolations
(range 0.060--0.130). These are distortion calibrations, not hyperparameter
choices. The fixed matched waveforms are accepted only after the independent
per-example SNR gate verifies every residual is within 1 dB.

## Four baseline arms

- A: existing latent CodecAttack, epsilon 1.0.
- B: existing original waveform baseline, epsilon 0.06082, historical recipe.
- C: tuned unrestricted waveform, alpha/epsilon=0.02, frozen on development.
- D: tuned <=4 kHz waveform, alpha/epsilon=0.02, frozen on development.

All final rows use the same 40 held-out S1 IDs, bank prompt, saved/reloaded
24-kHz PCM_16 input, production Opus/MP3/AAC implementations, and exact metric.
The exact metric lowercases and collapses whitespace, retains punctuation, and
requires the complete target as a contiguous substring. One fixed saved
waveform is transformed through every delivery channel; there is no
per-model/per-codec checkpoint reselection.

Arm C was the only missing full held-out arm. Four probe shards and four matched
shards were added, totaling 80 optimizations (approximately 6.9 GPU-hours),
which is the minimum distortion-calibrated completion. No extra seeds, models,
or scenarios were added. Arm D's first six completed probe outputs were
preserved; its slow serial allocation was stopped and only the remaining 34
probe IDs were resumed in four disjoint shards with the identical frozen
configuration. The scheduling change does not alter the statistical protocol.

For the <=4 kHz arm, the optimizer-domain perturbation uses an exact integer-bin
mask. Reports distinguish (i) delivered attack minus full-band carrier, which
includes the common 24->16->24 kHz resampling loss, from (ii) attack minus the
attacker's reloaded reference, which isolates the saved perturbation. PCM_16
quantization can contribute a very small nonzero >4 kHz fraction and is reported
rather than rounded to zero.

## Remaining confounds

1. The published latent arm can restore the last periodic success checkpoint,
   whereas the historical waveform arm B saves the last iterate. C and D both
   use the last iterate. The new four-arm comparison therefore cannot isolate
   parameterization from the published A/B checkpoint asymmetry.
2. Arm B also differs in warmup schedule and carrier normalization from A. Arms
   C/D correct these factors, which is why B and C are both retained.
3. Per-example SNR calibration equalizes realized distortion but creates
   per-example waveform epsilon values; it is not a single-budget comparison.
4. The baseline full evaluation is seed 1. It is a controlled paired estimate,
   not a three-seed uncertainty estimate. Arms B/C/D are labeled seed 1; the
   legacy unsuffixed arm-A bundle does not persist a seed, so its seed is
   reported as unknown rather than inferred from current defaults.
5. Ensemble N changes both member count and optimization compute. The comparison
   is nested but neither compute-matched nor exact-EoT-draw-matched because the
   codec proxy's global RNG is not tied to the declared seed.

## Manuscript statements requiring correction

- `sections/appendix.tex:23`: the historical waveform baseline did not have an
  identical warmup/projection schedule. Restrict this claim to tuned arms C/D or
  describe B accurately.
- `sections/appendix.tex:28` and `tables/tab_latent_vs_waveform.tex:19`: the
  published S1 A/B pair is not matched within 1 dB per example (only 12/50 are;
  common-reference mean waveform-minus-latent difference is -1.24 dB).
- `sections/methodology.tex:117`: Appendix does not currently define every
  checkpoint-selection rule. State “last iterate” for C/D and the actual latent
  periodic-success restore rule for A.
- `sections/experiments.tex:51`: “post-codec checkpoint selection” is undefined
  and does not describe the S1 N ladder precisely. The ladder selects minimum
  randomized codec-training loss after step 300; persist or disclose the chosen
  step in future runs.
- `tables/tab_ensemble_attack.tex` and `sections/experiments.tex:42-139`: these
  report a different 224-target, one-carrier, 600-step experiment. They must not
  be presented as the new 50-pair S1 N=1/N=2/N=3 comparison; replace or clearly
  label them as a separate legacy experiment.
- Any claim that waveform performance was tuned solely by SNR is incorrect for
  arm C/D: epsilon is chosen from distortion only, but alpha/epsilon is chosen
  by development attack effectiveness under that distortion regime.
- The newly controlled evaluator length-normalizes every codec output back to
  the exact input sample count after calling the production encoder/decoder.
  Legacy `*_codecs.json` files predate that common wrapper and, for AAC, retained
  a 640-sample tail (240,640 versus 240,000 samples on S1). The current AAC-LC
  comparison is internally identical across A/B/C/D but is not numerically
  interchangeable with stale manuscript AAC columns; those columns must be
  regenerated or explicitly labeled as the legacy delivery path.
