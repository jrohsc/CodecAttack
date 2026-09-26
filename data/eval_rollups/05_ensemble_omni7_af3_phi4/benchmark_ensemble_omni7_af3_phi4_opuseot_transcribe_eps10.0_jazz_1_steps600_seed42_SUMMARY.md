# Alternate 3-model Opus-EoT ensemble (seed 42)

224 robot-action target strings; epsilon=10, 600 steps, zero initialization.
Mean waveform SNR: 3.25 dB.

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Member | Qwen2.5-Omni-7B | 99.1 | 99.1 | 99.1 | 98.7 | 99.1 | 92.9 | 94.6 |
| Member | Audio Flamingo 3 | 95.1 | 95.5 | 95.1 | 93.8 | 95.1 | 42.4 | 47.8 |
| Member | Phi-4-multimodal | 37.9 | 39.3 | 38.4 | 36.2 | 38.4 | 28.6 | 31.7 |
| Held out | Qwen2-Audio | 0.4 | 0.9 | 0.0 | 0.0 | 0.4 | 4.9 | 4.9 |
| Held out | Kimi-Audio-7B-Instruct | 3.1 | 4.0 | 3.6 | 4.0 | 3.1 | 4.0 | 4.0 |
| Held out | Qwen2.5-Omni-3B | 13.4 | 14.3 | 12.9 | 12.9 | 14.3 | 12.1 | 12.5 |
