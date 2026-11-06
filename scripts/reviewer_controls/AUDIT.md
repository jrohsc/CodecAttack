# Baseline-fairness audit — CodecAttack waveform controls

Scope: the latent attack, the waveform baseline, the Opus BPDA proxy, SNR
calibration, checkpoint selection and the evaluation path, as they exist in the
repository — not as the manuscript describes them. Every number below was
measured by a script in this directory; the command is given with each finding.

All measurements use Qwen2-Audio, scenario S1 (`01_finance_voice_agent`), the
bank system prompt, and the published bundles under
`results_codec_robust*/01_finance_voice_agent/qwen2_audio/`.

---

## 1. What the code actually does

| | Latent (`RobustLatentCodecAttacker`) | Waveform S1 (`RobustWaveformAttacker`) | Waveform S3a (`WaveformAttacker`) |
|---|---|---|---|
| Driver | `scripts/run_s1_attack_bundleA.py` → `dataset/interview/run_attack.py` | `scripts/run_s1_waveform_bundleA_multibitrate.py` | `scripts/run_s3a_waveform_attack.py` |
| Optimizer | Adam | Adam | Adam |
| Init | zeros (`--rand-init` opt-in) | zeros (`--rand-init` opt-in) | zeros |
| Steps | 1000 | 1000 | 1000 |
| Step size α | 0.2 (latent units), α/ε = **0.2** | ε/5, i.e. α/ε = **0.2** | ε/5, α/ε = **0.2** |
| Warmup ratio | **0.3** | **0.2** | **n/a (no channel)** |
| Post-warmup schedule | **alternating** clean / Opus-BPDA (`step % 2 == 1`) | **Opus-BPDA on every step** | **no codec training at all** |
| Opus train grid | 16, 24, 32, 64, 128 kbps | 16, 24, 32, 64, 128 kbps | — |
| EoT samples/step | 1 | 1 | — |
| Constraint | ℓ∞ ball radius ε on latents | pointwise clamp to ±ε, then valid-range clamp | same |
| Checkpoint | **last iterate, or the delta from the last periodic channel-eval that hit the target** | last iterate | last iterate |
| Carrier loading | `load_music_by_name` → 24 kHz, **peak-normalized to 0.95** | `_load_carrier_24k` → 24 kHz, **not normalized** | `load_music_by_name`, normalized |
| Saved artifact | 24 kHz PCM_16 | 24 kHz PCM_16 | 24 kHz PCM_16 |

The manuscript's description — 1000 Adam steps, 30 % clean warmup, then
alternating clean and Opus-EoT updates, Opus grid {16, 24, 32, 64, 128} —
is **correct for the latent attack** (`dataset/interview/run_attack.py:484`
hardcodes `warmup_ratio=0.3`; `robust_latent_attack.py:1540-1545` implements the
alternation; `run_s1_attack_bundleA.py:66` sets the grid).

`appendix.tex:23` additionally claims that for the waveform baseline "warmup
ratio, step count, EoT grid, and projection schedule are identical". Findings
A1–A3 below show that the step count and EoT grid match but the warmup ratio and
the projection schedule do not, and that on S3a there is no EoT grid at all.

---

## A. Discrepancies that affect the comparison

### A1 — The latent attack has an oracle checkpoint the waveform attack does not

`robust_latent_attack.py:1982-2002`: when the periodic evaluation (default
`check_every = CHECK_EVERY = 25`, and the production driver does not override it)
finds the target in the channel-evaluated output, the current delta is stored in
`best_ir_delta`; at the end, `final_delta = best_ir_delta` if it was ever set.
Selection therefore uses the success criterion itself.

`robust_waveform_attack.py` has no equivalent: it always returns the last
iterate. It computes `best_delta` (line 496) and never uses it.

Effect: the latent arm gets up to 40 restore points chosen by the evaluation
metric; the waveform arm gets none.

How often this actually fired in the published runs **cannot be read back**: the
saved `*_summary.json` files do not persist `history`, and `best_ir_step` is not
recorded. What is verifiable is that the code path is enabled by default and the
production driver does not disable it. The size of the advantage is therefore
unquantified, not zero and not measured. Quantifying it needs a rerun with
`check_every=0` on the latent arm, which this task did not do because it would
change published latent numbers.

Status: **not fixed** — fixing it would change published latent numbers.
The reviewer-control arms in this directory all use last-iterate selection and
record `"checkpoint_rule": "last iterate"` in every `run_config.json`, so the new
comparison is internally consistent; the asymmetry against the *published*
latent bundles remains and is disclosed rather than silently corrected.

### A2 — Different warmup ratio and different post-warmup schedule

Latent: 30 % warmup, then alternate clean / codec (≈350 codec updates and
≈350 clean updates out of 1000).
Waveform S1: 20 % warmup, then codec on every step (≈800 codec updates, no
clean updates after step 200).

These are different training recipes, not the same recipe in two
parameterizations. The waveform baseline never optimizes the clean objective
after step 200, which plausibly costs it clean-channel ASR — and clean ASR is
one of the reported columns.

Status: **fixed for the new arms.** `run_attack.py` builds
`RobustWaveformAttacker(warmup_ratio=0.3, alternate_after_warmup=True)` for the
tuned arms, which reproduces the latent schedule exactly
(`robust_waveform_attack.py:420-424`). Arm B deliberately keeps the historical
0.2 / no-alternation recipe so the published baseline stays reproducible.

### A3 — The S3a waveform baseline is not codec-aware at all

`scripts/run_s3a_waveform_attack.py:128` constructs a plain `WaveformAttacker`.
There is no Opus BPDA, no EoT grid and no warmup/harden schedule, while the
S3a latent arm it is compared against runs `channel_mode=multi_bitrate`.

Status: **fixed for the new arms** (arms C and D use the codec-aware recipe on
both scenarios). Arm B on S3a is marked in `run_config.json` as
`"opus_train_kbps": []`, `"post_warmup_schedule": "direct only"` so it is never
mistaken for a codec-aware baseline.

### A4 — Carrier normalization differs between the two arms

The latent pipeline loads carriers through `load_music_by_name`, which
peak-normalizes to 0.95 (`music_carrier.py:load_music`, `normalize=True`).
The S1 waveform driver's `_load_carrier_24k` does not normalize. All 50 S1
carriers peak at exactly 0.890, so the latent arm attacks a carrier 0.57 dB
louder than the waveform arm does. At a fixed absolute perturbation this shifts
carrier-referenced SNR by 0.57 dB — more than half the manuscript's 1 dB
tolerance, in the direction that makes the waveform arm look quieter than it is.

Status: **fixed for the new arms.** Every arm in `run_attack.py` loads through
`load_music_by_name`, and the SNR reference is that same normalized carrier.

### A5 — SNR was referenced to a different signal for each arm

`_compute_snr` is called with `original_audio_24k` (the true carrier) in the
latent attack and with `resample(resample(carrier, 24→16), 16→24)` in the
waveform attack, because `AttackResult.original_wav` is built from the 16 kHz
working signal (`robust_waveform_attack.py:590-594`). The two arms therefore
divide by different denominators.

Measured cost of the 24→16→24 round trip (`scratchpad/refsnr.py`):
24.1 and 38.2 dB on two S1 carriers, 19.5 and 24.5 dB on two S3a carriers. At
the ~6.6 dB operating point this is 2–7 % of the noise budget, i.e. 0.1–0.3 dB.
Small, but it is systematic and it is not the same for the two arms.

Status: **fixed for the new arms.** `run_attack.py` uses the common normalized
24 kHz carrier for every arm and additionally records
`snr_db_vs_attacker_reference` and `resample_only_snr_db` per example, so the
size of the effect is visible rather than assumed.

### A6 — The published S1 pair is not SNR-matched within 1 dB

Command:
```
python -m reviewer_controls.measure_reference_snr --scenario s1 \
  --method latent_codecattack \
  --bundle results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0 \
  --output reviewer_control_outputs/snr/s1_latent_eps1.0.jsonl
```
(and the same for the waveform bundle), then differencing per example.

| pairing | n | mean Δ | median Δ | SD | min | max | within 1 dB | within 2 dB |
|---|---|---|---|---|---|---|---|---|
| latent ε=1.0 vs waveform ε′=0.061 (the published pair) | 50 | **−1.24 dB** | −1.51 | 1.48 | −5.02 | +2.76 | **12/50** | 39/50 |
| latent ε=1.0 vs waveform ε=0.075 | 50 | −3.03 dB | −3.33 | 1.42 | −6.82 | +0.88 | 4/50 | 10/50 |

Latent realized SNR is 6.59 dB mean, 6.29 median, SD 1.73, range 3.57–10.71.

Negative Δ means the waveform arm is **louder**, i.e. it is given *more*
distortion than the latent arm, so this does not flatter CodecAttack's win. But
`appendix.tex:28` states the clean-channel SNR "differs by less than 1 dB"; in
aggregate it differs by 1.24 dB under a common carrier reference (≈0.9 dB if
each arm is referenced to its own resampled carrier, from the +0.29 dB offset
measured on a probe run), and per example it differs by more than 1 dB on 38 of
50 examples.

The mechanical reason: latent realized SNR has SD 1.73 dB across examples, so no
single global ε′ can match it per example. `calibrate_epsilon.py` therefore
solves a per-example budget.

Status: **not a code bug**; it is a claim that needs either per-example
calibration (implemented here) or a weaker wording. No published number has been
changed.

### A7 — `matches_target` in the bundles is looser than the manuscript's metric

`scripts/eval_opus_bundle.py:126` sets
`matches_target = exact_substring OR wer <= 0.5`. The manuscript's metric is the
contiguous-substring rule alone. A strict re-score already exists
(`codecattack_audit_20260917/strict_main_table_rescore.csv`), so the published
table is not affected, but any tool reading `matches_target` straight out of a
bundle gets the loose metric.

Status: not applicable to the new pipeline — `common.target_match` implements
lowercase + whitespace-collapse + substring, punctuation retained, and nothing
else.

### A8 — The existing ≤4 kHz and <400 Hz control bundles are unusable as they stand

| bundle | examples present | mean SNR | Δ vs latent |
|---|---|---|---|
| `eps_0.06082_fftlp4000_seed1` | **1 / 50** | 15.90 dB | +9.3 dB |
| `eps_0.06082_lp400_norm_seed1` | **9 / 50** | 15.55 dB | +8.9 dB |

Both are far quieter than the latent arm and neither is complete. The <400 Hz
control is kept available (`--lowpass-hz`) as the manuscript describes, but it
cannot currently support a matched-SNR claim.

Status: superseded. The new ≤4 kHz arm calibrates its own budget
(`epsilon_probe.bandlimited_4khz = 0.166`, solved to ~6.6 dB) and runs the full
split.

---

## B. Things that were checked and are correct — do not "fix" them

### B1 — Opus decoding is already sample-aligned; do not add delay compensation

`reviewer_controls/audit_audio_path.py` plus a cross-correlation check on a real
carrier (`scratchpad/xcorr.py`):

| codec | 16 | 24 | 32 | 64 | 128 kbps |
|---|---|---|---|---|---|
| impulse peak lag (samples) | 0 | 0 | 0 | 0 | 0 |
| xcorr lag on speech (samples) | 0 | — | 0 | 0 | 0 |

MP3 64 and AAC-LC 64 also measure 0 lag. FFmpeg honours the Opus container
pre-skip on decode, and output sample counts equal input sample counts at every
bitrate. Adding a delay correction would *introduce* a misalignment.

`codec_roundtrip` therefore only trims or zero-pads the tail, and it does so
after delegating to the production encoders.

### B2 — The evaluation input is the saved, reloaded PCM

`scripts/eval_opus_bundle.py:221` reads the attacked WAV from disk, applies the
codec, resamples to 16 kHz and generates. The new `evaluate.py` does the same,
from `row["adv_path"]`. The in-memory attack output is stored separately as
`in_memory_attack_output` and is never scored.

### B3 — This ffmpeg build has no `libopus`

`ffmpeg -encoders` lists `opus` (native, requires `-strict -2`), `libmp3lame`
and `aac`, but not `libopus`. A reimplementation of the codec round trip using
`-c:a libopus` fails outright — which is how this was caught. `common.py` now
delegates to the production helpers (`demo_ota.apply_opus_compression`,
`aac_channel.apply_mp3_compression`, `aac_channel.apply_aac_compression`), so the
delivery channel is identical to the one that produced the published numbers.

---

## C. Bug found and fixed in the ≤4 kHz control

`torch.fft.rfftfreq` returns **float32**. At 16 kHz over 74 880 samples it
reports bin 18720 as exactly 4000.0 Hz, while the float64 measurement computes
4000.0000000000005 Hz. A float comparison therefore kept that bin in the
optimizer's mask and counted it as out-of-band in the audit, reporting **1.9e-5
of the perturbation energy above 4 kHz that was not actually there**.

Both sides now decide membership with the exact integer rule
`k · fs ≤ cutoff · n` (`common.inband_rfft_mask`,
`robust_waveform_attack._spectral_delta`). Regression test:
`test_cutoff_bin_membership_is_exact_not_float32_rounded`.

Measured out-of-band energy fraction after the fix (`scratchpad/leak.py`):

| ε | optimizer domain (16 kHz) | after 16→24 kHz resample (float) | after PCM_16 save/reload |
|---|---|---|---|
| 0.010 | 2.2e-14 | 9.8e-07 | 2.1e-05 |
| 0.030 | 2.2e-14 | 9.8e-07 | 3.1e-06 |
| **0.061** | 2.2e-14 | 9.8e-07 | **1.5e-06** |
| 0.120 | 2.2e-14 | 9.8e-07 | 1.1e-06 |
| 0.250 | 2.2e-14 | 9.8e-07 | 1.0e-06 |

Reading: the perturbation is band-limited to machine precision in the domain it
is optimized in. The resampler's finite-length kernel contributes a fixed
9.8e-7. PCM_16 quantization noise is white and of fixed absolute size, so its
relative contribution falls as the perturbation grows; at the operating budget
it is 1.5e-6, i.e. −58 dB. No clipping occurs: the realized peak equals ε
exactly for every ε up to 0.2 (S1) and 0.25 (S3a), above which the valid-range
projection begins to bind on S3a.

---

## D. Results that would need rerunning if the discrepancies were corrected

Nothing has been rerun and nothing has been overwritten. If A1–A4 were corrected
in the published pipeline, these would need regeneration:

| finding | affected results |
|---|---|
| A1 checkpoint asymmetry | every latent-vs-waveform comparison, all scenarios and models (Table `tab_latent_vs_waveform`) — or, cheaper, add the same checkpoint rule to the waveform arm |
| A2 warmup/schedule | every waveform-baseline cell in `results_codec_robust_waveform/` |
| A3 no codec EoT on S3a | the S3a and S3b waveform-baseline columns |
| A4 carrier normalization | the S1 waveform bundles only (S3a already normalizes) |
| A6 SNR match | the ε′ column of `tab_latent_vs_waveform` and the ≤1 dB claim in `appendix.tex:28` |

The claim most exposed is `appendix.tex:23` ("warmup ratio, step count, EoT grid,
and projection schedule are identical") — that sentence is false for the
shipped waveform baseline on every scenario, and false in three ways on S3a.
