# Alternate 3-model Opus-EoT ensemble (seed 1)

224 robot-action target strings; epsilon=10, 600 steps, zero initialization.
Mean waveform SNR: 3.32 dB.

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Member | Qwen2.5-Omni-7B | 99.6 | 99.1 | 99.6 | 99.6 | 99.6 | 92.4 | 96.0 |
| Member | Audio Flamingo 3 | 94.6 | 93.8 | 95.1 | 89.3 | 94.6 | 46.4 | 47.8 |
| Member | Phi-4-multimodal | 37.5 | 37.9 | 40.6 | 33.5 | 38.4 | 29.0 | 33.0 |
| Held out | Qwen2-Audio | 0.9 | 0.9 | 0.4 | 0.9 | 0.4 | 7.1 | 6.7 |
| Held out | Kimi-Audio-7B-Instruct | 6.7 | 6.2 | 6.2 | 6.2 | 6.2 | 7.6 | 7.1 |
| Held out | Qwen2.5-Omni-3B | 18.8 | 16.1 | 18.3 | 17.9 | 19.2 | 18.8 | 18.8 |
