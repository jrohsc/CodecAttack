# sample30_opus_sweep — legacy codec-survival run on physical-agent carriers

Moved 2026-04-16 from `results/03_music_industry_bypass/qwen2_audio/sample30/` because
the carriers (generic music: `calm_1`, `christmas_jazz_1`, `classical_music_1`,
`empire_state_of_mind`, `jazz_1`) and per-file targets (SayCan action plans)
belong to S4's physical-agent threat model, not S3's music-industry classifier bypass.

Caveats:
- **Channels don't match the canonical S4 grid**. `scenario.json` for S4 specifies
  `physical_sim` + `chain[opus_64k → physical_sim]`. This run used the S3 streaming
  grid: clean / opus{16,24,32,64,128,192}k / g722_64k. Treat these numbers as
  "codec survival of physical-agent adv wavs," not the S4 physical-sim headline.
- **eps=None in results.jsonl**. Adv wavs are legacy symlinks from
  `1_results_qa/benchmark_qwen2_audio_eps1.0_*`, so the effective budget is eps=1.0.
- Real S4 physical_sim eval still needs to be run — see TODO.md § S4.

Headline from this run (TODO.md):
| channel | target_substring % |
|---|---|
| clean    | 45.4 |
| opus192k | 45.8 |
| opus128k | 40.3 |
| opus64k  | 14.8 |
| opus32k  |  0.0 |
| opus24k  |  0.0 |
| opus16k  |  0.0 |
| g722_64k |  1.9 |
