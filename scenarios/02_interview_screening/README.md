# 02 — Interview Screening

**Threat model**: HR voice agent transcribes/evaluates candidate interview audio. Adversarial speech-carrier audio (Mandarin TTS, English TTS, English murmur) forces a favorable verdict regardless of content.

## Audio
- `audio/latent_speech/` — 27 wavs symlinked from interview experiments (`7-10_results_interview_*`)
- `audio/latent_music/` — empty placeholder
- `audio/waveform/` — **EMPTY**. Need waveform-PGD baseline on same speech carriers.

## Channel grid (clean voice eval; not phone)
`clean`, `opus 16/32/64`, `g722` — interviews typically recorded clean or over WebRTC, not PSTN.

## Caveat
The HR system prompt alone makes Qwen2-Audio emit positive verdicts on **any** Mandarin audio (memory: `hr_prompt_dominates_verdict.md`). Substring-of-target metric is what matters; flip-rate alone is invalid.
