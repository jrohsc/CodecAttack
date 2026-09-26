# Alternate 3-model Opus-EoT ensemble (seed 7)

224 robot-action target strings; epsilon=10, 600 steps, zero initialization.
Mean waveform SNR: 3.34 dB.

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Member | Qwen2.5-Omni-7B | 99.6 | 99.6 | 99.6 | 99.1 | 99.6 | 92.9 | 96.9 |
| Member | Audio Flamingo 3 | 96.4 | 95.5 | 95.5 | 93.3 | 96.4 | 44.2 | 45.1 |
| Member | Phi-4-multimodal | 43.3 | 39.7 | 41.5 | 37.1 | 41.5 | 25.9 | 37.1 |
| Held out | Qwen2-Audio | 1.3 | 0.9 | 0.9 | 0.4 | 1.3 | 4.5 | 6.7 |
| Held out | Kimi-Audio-7B-Instruct | 2.7 | 3.6 | 2.7 | 2.2 | 1.8 | 1.8 | 2.2 |
| Held out | Qwen2.5-Omni-3B | 16.1 | 13.8 | 14.3 | 14.7 | 13.8 | 9.4 | 14.3 |
