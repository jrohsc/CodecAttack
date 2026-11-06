# 03c — Music-industry content-moderation bypass

**Threat model**: attacker uploads a track whose lyrics contain explicit content (profanity, slurs); the platform's Audio-LLM-backed content-moderation classifier is forced to emit a "clean / suitable for all audiences" verdict, bypassing age-gating and monetization restrictions. Streaming ingest re-encodes via Opus/G.722.

## Carrier — BLOCKER
- Target dir: `data/music/ai_generated_explicit/` — **EMPTY, NOT YET GENERATED**
- Requirement: 4–6 Suno/Udio AI-generated clips with explicit-lyrics flag on, ~10–30s each, ≥2 genres.
- Provenance constraint: AI-generated so distribution of explicit content stays license-clean (we own the output).
- Save filenames as `suno_explicit_{genre}_{n}.mp3`; document the prompt in a sidecar JSON.

## Targets
- Source: `target_drafts/03c_moderation_bypass.json`
- Primary: 5 moderation-bypass lines. **No secondary category** (attribution/ingestion don't compose with age-gate bypass).

## Pair count (planned)
- 4–6 carriers × 5 primary targets = **20–30 primary pairs**
- Per-model × 3 eps = up to 270 attack jobs once carriers exist.

## Channel grid
Opus sweep `{16, 24, 32, 64, 128, 192}` kbps + `g722` + `clean`.

## Status
- [ ] **Blocker**: generate `data/music/ai_generated_explicit/` carriers
- [ ] Attack runs (qwen2_audio / audio_flamingo / qwen25_omni) × 3 eps
- [ ] Eval — each model on its own adv wavs only
