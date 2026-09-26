# Alternate three-model Opus-EoT ensemble: seeds 1, 7, 42

224 targets per seed. Exact target-substring match (%), mean ± sample SD.
Seed 1 did not explicitly seed per-clip Opus draws; seeds 7 and 42 did.

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Member | Qwen2.5-Omni-7B | 99.4 ± 0.3 | 99.3 ± 0.3 | 99.4 ± 0.3 | 99.1 ± 0.5 | 99.4 ± 0.3 | 92.7 ± 0.3 | 95.8 ± 1.2 |
| Member | Audio Flamingo 3 | 95.4 ± 0.9 | 94.9 ± 1.0 | 95.2 ± 0.2 | 92.1 ± 2.5 | 95.4 ± 0.9 | 44.3 ± 2.0 | 46.9 ± 1.6 |
| Member | Phi-4-multimodal | 39.6 ± 3.2 | 39.0 ± 0.9 | 40.2 ± 1.6 | 35.6 ± 1.9 | 39.4 ± 1.8 | 27.8 ± 1.7 | 33.9 ± 2.8 |
| Held out | Qwen2-Audio | 0.9 ± 0.5 | 0.9 ± 0.0 | 0.4 ± 0.5 | 0.4 ± 0.5 | 0.7 ± 0.5 | 5.5 ± 1.4 | 6.1 ± 1.0 |
| Held out | Kimi-Audio-7B-Instruct | 4.2 ± 2.2 | 4.6 ± 1.4 | 4.2 ± 1.8 | 4.1 ± 2.0 | 3.7 ± 2.3 | 4.5 ± 2.9 | 4.4 ± 2.5 |
| Held out | Qwen2.5-Omni-3B | 16.1 ± 2.7 | 14.7 ± 1.2 | 15.2 ± 2.8 | 15.2 ± 2.5 | 15.8 ± 3.0 | 13.4 ± 4.8 | 15.2 ± 3.2 |
