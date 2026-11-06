# Mechanistic Analysis: Why Latent Survives Codec Compression

Cross-scenario home for experiments that explain **why** latent-space PGD on
EnCodec survives Opus/MP3/AAC compression while SNR-matched waveform PGD does
not. Lives at the top level of `0_all_combined/` (not nested under any single
scenario) because the same mechanism applies across S1–S3.

## Headline finding

The two attacks place their perturbation energy in essentially **disjoint
spectral regions**: latent concentrates ≥2× more energy below 400 Hz (where
Opus/MP3/AAC always spend bits because of voicing fundamentals); waveform
concentrates ≥2.5× more energy above 4 kHz (where low-bitrate Opus discards
energy under narrowband mode). Spearman ρ between the two per-Bark-band
distributions is 0.017 — they live in different bands.

## Sub-experiments

| Folder | Status | What it shows |
|---|---|---|
| `per_bark_band/` | DONE (n=25 latent, n=21 waveform) | Per-Bark-band δ energy distribution. 3-test gate (sub-400Hz tilt, >4kHz spread, spectral independence) PASSES on Qwen2-Audio S1 finance bundle. |
| `cosine_through_codec/` | DONE (5 Opus bitrates) | Pre/post-codec direction survival cos(δ_pre, δ_post). Both attacks preserve direction similarly *within each band*; the gap is small (~10% at 16 kbps). The asymmetry comes from where the energy was *placed*, not how well it survives once placed. |
| `decoder_freq/` | DONE (n=384 Jacobian columns / 24 random / 21 actual) | *Why* the latent attack lands sub-4 kHz. Per-dim decoder Jacobian columns (J(z)·e_i) and random-direction control with matched per-dim σ both put **~93% of energy below 4 kHz**, within ~1 pp of the actual adversarial spectrum (91.9%). The bias is structural — the EnCodec decoder routes generic latent perturbations to low frequencies — not optimization-driven. Per-dim heatmap shows every one of the 128 dims peaks at ~1.9 or ~2.5 kHz; none peak above 4 kHz. |
| `lowpass_waveform_control/` | IN PROGRESS | Falsification control: train a waveform PGD that is constrained to ≤300 Hz via per-step low-pass projection. If it survives Opus, mechanism = band placement alone. If it fails, latent contributes beyond band placement. |

## How to reproduce

All scripts live in `0_all_combined/scripts/`. They write into the matching
sub-folder under this directory. CPU-only unless noted.

```bash
# Gate experiment (~30 s, CPU)
python 0_all_combined/scripts/per_bark_band_delta_energy.py

# Direction survival through Opus (~5 min, CPU, needs ffmpeg)
python 0_all_combined/scripts/cosine_through_codec.py

# Decoder frequency response — Jacobian + random control + actual adv (~3 min, GPU)
python 0_all_combined/scripts/decoder_freq_response.py \
    --n-jac-carriers 3 --n-random-samples 8 --n-stat-carriers 21

# Lowpass-waveform control attack (GPU, ~4 hr for 25 carriers)
python 0_all_combined/scripts/run_s1_waveform_bundleA_multibitrate.py \
    --eps 0.075 --lowpass-hz 300
```

## Source data

Adversarial wavs from the canonical S1 finance Bundle A:
- Latent: `0_all_combined/results/01_finance_voice_agent/qwen2_audio/eps_1.0/audio/`
- Waveform: `0_all_combined/results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075/audio/`

These are SNR-matched (~5.8 dB) for an apples-to-apples comparison.
