# 03 — Music-Industry Detection Bypass

**Threat model**: a benign user plays music from Spotify / YouTube / SoundCloud. The music has been adversarially perturbed by the attacker before upload. Streaming platforms re-encode through Opus / AAC / MP3 at various bitrates. Downstream audio LLM agents (smart speakers, voice agents listening to background) pick up the hidden command despite the lossy re-encoding chain.

This is the **headline scenario for the latent-vs-waveform method comparison** — with the largest amount of pre-existing data and the cleanest codec-survival story.

## Audio
- `audio/latent/` — **1043 wavs** symlinked from:
  - `1_results_qa/benchmark_qwen2_audio_eps1.0_*_saycan/` (SayCan command targets, multiple music carriers)
  - `1_results_transcribe/benchmark_qwen2_audio_eps1.0_*/` (nav / comm / sys / emerg / nego targets, multiple carriers)
- `audio/waveform/` — **6 wavs** symlinked from `6_results_waveform/sweep_waveform_eps{1.0,2.0,5.0}_empire_state_of_mind/` (saycan_000, saycan_001 each)
- Per-file targets auto-built from source `config.json:targets|commands` into `_target_map.json`

## Channel grid (streaming codecs)
- `clean`
- `opus 24 / 64 / 128 / 192 kbps` — covers Spotify (Vorbis≈Opus 96-320), YouTube (Opus 160), SoundCloud (Opus 128), Apple Music (AAC 256, TODO add)

## Headline experiment
Head-to-head plot: ASR vs Opus bitrate, two curves per (model, target). Expected: latent attack stays high across bitrates; waveform PGD falls off below 64 kbps.

## Sub-sampling for first run
1043 latent × 5 channels = ~5000 inference calls. Consider sub-sampling to N=20 wavs per carrier × 5 channels for the first pass.
