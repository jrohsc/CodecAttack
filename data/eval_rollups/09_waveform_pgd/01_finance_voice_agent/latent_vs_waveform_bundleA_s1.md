# S1 Bundle A — Latent vs Waveform across codecs

- Scenario: `01_finance_voice_agent`, target model `qwen2_audio`.
- **Fair comparison set**: 50 carriers present in both bundles.
- Latent: eps=1.0, clean SNR 5.77 dB (measured), multi-bitrate Opus EoT {16,24,32,64,128} kbps in EnCodec latent space.
- Waveform: eps_0.075, clean SNR ~5.80 dB.
- Opus rows come from each attacker's `test_opus_robustness`; MP3/AAC rows come from a post-hoc inference-only eval (`scripts/eval_mp3_aac_bundle.py`), no retraining.


## Opus — target_substring_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| opus16k | 14.0 | 4.0 |
| opus24k | 56.0 | 20.0 |
| opus32k | 66.0 | 38.0 |
| opus64k | 90.0 | 44.0 |
| opus128k | 92.0 | 46.0 |
| opus192k | 92.0 | 46.0 |

### Opus — exact_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| opus16k | 10.0 | 2.0 |
| opus24k | 44.0 | 18.0 |
| opus32k | 50.0 | 16.0 |
| opus64k | 80.0 | 24.0 |
| opus128k | 88.0 | 26.0 |
| opus192k | 88.0 | 26.0 |

### Opus — avg WER

| Bitrate | Latent | Waveform |
|---|---|---|
| opus16k | 1.949 | 2.869 |
| opus24k | 1.245 | 2.020 |
| opus32k | 0.685 | 1.736 |
| opus64k | 0.220 | 1.345 |
| opus128k | 0.118 | 1.289 |
| opus192k | 0.108 | 1.500 |

### Opus — post-codec avg SNR (dB)

| Bitrate | Latent | Waveform |
|---|---|---|
| opus16k | 9.7 | 5.7 |
| opus24k | 12.7 | 9.0 |
| opus32k | 14.0 | 10.3 |
| opus64k | 20.5 | 16.6 |
| opus128k | 26.1 | 24.3 |
| opus192k | 32.4 | 32.4 |

## MP3 — target_substring_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| mp364k | 84.0 | 44.0 |
| mp396k | 88.0 | 46.0 |
| mp3128k | 94.0 | 42.0 |
| mp3192k | 94.0 | 44.0 |

### MP3 — exact_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| mp364k | 74.0 | 22.0 |
| mp396k | 84.0 | 22.0 |
| mp3128k | 88.0 | 24.0 |
| mp3192k | 90.0 | 22.0 |

### MP3 — avg WER

| Bitrate | Latent | Waveform |
|---|---|---|
| mp364k | 0.235 | 1.617 |
| mp396k | 0.144 | 1.304 |
| mp3128k | 0.106 | 1.465 |
| mp3192k | 0.085 | 1.455 |

### MP3 — post-codec avg SNR (dB)

| Bitrate | Latent | Waveform |
|---|---|---|
| mp364k | 23.0 | 18.7 |
| mp396k | 25.3 | 21.4 |
| mp3128k | 25.8 | 23.6 |
| mp3192k | 25.8 | 23.2 |

## AAC — target_substring_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| aac64k | 2.0 | 0.0 |
| aac96k | 2.0 | 0.0 |
| aac128k | 2.0 | 0.0 |
| aac192k | 2.0 | 0.0 |

### AAC — exact_match %

| Bitrate | Latent | Waveform |
|---|---|---|
| aac64k | 2.0 | 0.0 |
| aac96k | 2.0 | 0.0 |
| aac128k | 2.0 | 0.0 |
| aac192k | 2.0 | 0.0 |

### AAC — avg WER

| Bitrate | Latent | Waveform |
|---|---|---|
| aac64k | 4.025 | 4.513 |
| aac96k | 4.323 | 4.470 |
| aac128k | 4.185 | 4.392 |
| aac192k | 4.279 | 4.470 |

### AAC — post-codec avg SNR (dB)

| Bitrate | Latent | Waveform |
|---|---|---|
| aac64k | 22.2 | 17.1 |
| aac96k | 31.8 | 27.1 |
| aac128k | 34.6 | 33.5 |
| aac192k | 34.8 | 34.0 |

## Sources

- Latent bundle: `results_codec_robust/01_finance_voice_agent/qwen2_audio/eps_1.0/audio/*_summary.json` + `*_codecs.json`
- Waveform bundle: `results_codec_robust_waveform/01_finance_voice_agent/qwen2_audio/eps_0.075/audio/*_summary.json` + `*_codecs.json`
