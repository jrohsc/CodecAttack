# Audio-Agent Safety Benchmark

**End-to-end safety benchmark for audio-native LLM voice agents under codec-robust adversarial audio.**

This folder consolidates adversarial-audio assets from across the project (`1_results_qa`, `1_results_transcribe`, `3_results_physical_attack`, `6_results_waveform`, `6_results_waveform_airgap`, `7-10_results_interview_*`, `11_results_bank_voiceagent_scenario`) into a single uniform harness with 4 attack scenarios + 1 smoke test.

> **Headline claim**: a single music-carrier latent-space attack flips voice-agent behavior across realistic delivery channels (Opus, G.711, G.722, simulated physical OTA), while a matched-budget waveform-PGD baseline does not survive past the first codec.

**See [`INVENTORY.md`](INVENTORY.md) for the per-scenario audio counts, channel grids, gaps, and how to run.**

---

## Four scenarios at a glance

| # | Case | Carrier | Channel grid | Latent / Waveform wavs |
|---|---|---|---|---:|
| 01 | Finance / banking voice agent | Speech (Mandarin) | VoIP — Opus / G.711 / G.722 / chain | 11 / 0 |
| 02 | Interview screening | Speech (Mandarin/English/murmur) | Clean voice — Opus 16/32/64 / G.722 | 27 / 0 |
| 03 | Music industry detection bypass | Music (5 carriers) | Streaming — Opus 24/64/128/192 | 1043 / 6 |
| 04 | Simulated OTA physical agent | Music | Physical sim — IR + timeshift + noise; chain | 62 / 104 |

Plus `diagnostics/` for harness validation and method comparison.

## Layout

```
0_all_combined/
├── README.md  INVENTORY.md  TODO.md
├── src/audio_agent_bench/    # python package: adapters, channels, metrics, runner, CLI
├── scenarios/
│   ├── 01_finance_voice_agent/
│   ├── 02_interview_screening/
│   ├── 03_music_industry_bypass/
│   │   └── subsets/sample30/        # 30-wav paper-ready subsample, full channel grid
│   ├── 04_simulated_ota_physical_agent/
│   └── diagnostics/
│       ├── smoke_latent_vs_waveform/
│       └── waveform_vs_latent_comparison/
├── data/                     # clean carriers (speech, music, music_trimmed)
├── configs/prompts/          # LLM-judge rubrics
├── scripts/                  # build_target_maps.py + run launchers
└── results_summary/          # aggregated CSV + markdown + heatmaps
```

Each `scenarios/NN_*/` contains:
- `scenario.json` — schema-validated record (system prompt, channels, metrics, target models, audio target map)
- `system_prompt.txt` — agent's policy
- `audio/` (or `audio/{latent,waveform,...}/`) — symlinks to the underlying experiment dirs (zero copy)
- `_target_map.json` — auto-generated per-file targets (do not edit by hand; rebuild with `python scripts/build_target_maps.py`)
- `README.md` — scenario rationale + reuse provenance
- `_target_map.misses.json` (only if any wavs failed target lookup)

## Quick start

```bash
conda activate codec-attack

# Rebuild per-file target maps if any audio symlinks change
python scripts/build_target_maps.py

# Smoke (one wav per attack type, 2 channels, 1 model)
bash scripts/run_scenario.sh scenarios/diagnostics/smoke_latent_vs_waveform

# Full case — music industry bypass through Opus codec sweep
bash scripts/run_scenario.sh scenarios/03_music_industry_bypass qwen2_audio opus_64k

# Everything
bash scripts/run_all.sh
```

## Headline experiment

Per-scenario plot: ASR (target-substring-match or WER ≤ 0.3) vs channel condition, two curves per scenario — `latent` (ours) vs `waveform` (Carlini-style baseline). Cases 03 (music industry, big N) and 04 (OTA, both methods present) are the cleanest comparisons. Cases 01 and 02 still need waveform baselines run.

## Headline metrics

| Metric | What | Where it matters |
|---|---|---|
| `target_substring_match` | output contains attacker's per-file target string | all |
| `wer` | edit distance to per-file target | all |
| `exact_match` | verbatim match | finance (auth), interview (verdict) |
| `snr_db` | perceptual quality of adversarial audio | all (reportable) |
| `tool_name_match` | best-effort substring proxy for tool calls | finance (cautioned — not function-calling-tuned) |
| LLM judge | policy violation rubric (TODO) | finance, interview |

## Relationship to ServiceNow EVA

Inspired by EVA's scenario/judge layout, but **different threat model and architecture**: EVA assumes cascade (STT → text-LLM with native function calling → TTS); adversarial audio there hits STT. This benchmark targets the audio LLM directly. EVA cited in Related Work; no code vendored.

## License

MIT.
