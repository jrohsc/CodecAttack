# 224-target Opus-EoT runs 1, 7, and 42

The published zero-initialized seed-1 bundle is combined with new zero-initialized runs 7 and 42. The published run did not explicitly seed Opus draws per clip; runs 7 and 42 do. Values are strict target-substring ASR (%), mean ± sample SD over these three runs, with 224 targets per run.
Mean waveform SNR across the sweep: 1.58 dB; per-seed means: {'1': 1.571, '7': 1.723, '42': 1.447}.

| Role | Victim | Clean | Opus 64 | Opus 128 | MP3 64 | MP3 128 | AAC-LC 64 | AAC-LC 128 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| member | Qwen2-Audio | 100.0 ± 0.0 | 99.9 ± 0.3 | 100.0 ± 0.0 | 99.9 ± 0.3 | 100.0 ± 0.0 | 1.8 ± 1.2 | 2.2 ± 0.8 |
| member | Qwen2.5-Omni-7B | 96.3 ± 1.7 | 95.1 ± 2.2 | 96.4 ± 1.9 | 94.8 ± 3.0 | 96.6 ± 1.8 | 82.3 ± 2.7 | 87.5 ± 1.9 |
| member | Audio Flamingo 3 | 95.7 ± 1.1 | 92.0 ± 0.9 | 94.9 ± 1.3 | 86.2 ± 0.9 | 95.4 ± 1.0 | 10.7 ± 1.2 | 11.5 ± 1.1 |
| held out | Qwen2.5-Omni-3B | 8.9 ± 1.2 | 9.4 ± 0.8 | 9.5 ± 0.7 | 8.0 ± 1.2 | 8.8 ± 0.7 | 7.9 ± 0.7 | 9.5 ± 1.1 |
| held out | Kimi-Audio-7B-Instruct | 0.4 ± 0.4 | 0.4 ± 0.4 | 0.1 ± 0.3 | 0.4 ± 0.4 | 0.4 ± 0.4 | 0.0 ± 0.0 | 0.1 ± 0.3 |
| held out | Phi-4-multimodal | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.1 ± 0.3 | 0.1 ± 0.3 |
