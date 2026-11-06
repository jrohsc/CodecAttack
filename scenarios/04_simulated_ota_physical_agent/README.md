# 04 — Simulated OTA, Physical Agent

**Threat model**: a benign user plays music near a physical audio-LLM agent (smart speaker, household robot, voice-controlled appliance). The music carries a hidden adversarial command. The acoustic channel destroys most adversarial perturbations — but our latent music-carrier attack survives where Carlini-style waveform PGD does not.

## Audio
- `audio/latent/` — **62 wavs** symlinked from:
  - `3_results_physical_attack/results_empirical_ota_IR/`
  - `3_results_physical_attack/results_yakura/`
  - `3_results_physical_attack/results_diverse_airgap/`
- `audio/waveform/` — **104 wavs** symlinked from `6_results_waveform_airgap/` (waveform PGD with OTA-aware training)

## Channel grid (simulated physical)
- `physical_sim(ts60, snr20)` — moderate room
- `physical_sim(ts0, snr30)` — clean OTA
- `physical_sim(ts30, snr15)` — noisy OTA
- `chain[opus_64k → physical_sim(ts60, snr20)]` — streamed-then-OTA

## Scope discipline
**SIMULATED OTA only.** Real-hardware OTA results live in `3_results_physical_attack/`. Do not conflate. See memory: `physical_ota_acoustic_analysis.md`, `corrected_ir_analysis.md`, `exp27_speech_detection.md`.
