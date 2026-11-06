# 03a — Music-industry AI-detection bypass

**Threat model**: attacker uploads an AI-generated track to a platform whose AI-detection classifier is backed by an Audio LLM. The carrier is adversarially perturbed so the classifier emits a "human-authored" verdict, letting synthetic content through. Streaming ingest re-encodes via Opus/G.722.

## Carrier
- Source: `data/music/ai_generated/` (8 Suno clips, instrumental + lyrical, multi-genre)
- Provenance constraint: carriers MUST be AI-generated so the threat surface (AI-detection classifier) applies.

## Targets
- Source: `target_drafts/03a_ai_detection_bypass.json`
- Primary: 5 AI-detection bypass lines (headline metric = substring match)
- Secondary (reported separately): 2 attribution + 2 ingestion lines

## Pair count
- 8 carriers × 5 primary targets = **40 primary pairs**
- Per-model × 3 eps (0.3 / 0.5 / 1.0) = 40 × 3 × 3 = **360 attack jobs**

## Channel grid
Opus sweep `{16, 24, 32, 64, 128, 192}` kbps + `g722` + `clean` (matches S3b/S3c for cross-scenario comparability).

## Status
- [ ] Attack runs (qwen2_audio / audio_flamingo / qwen25_omni) × 3 eps
- [ ] Eval — each model on its own adv wavs only (no cross-model transfer; see memory `cross-model transfer 0%`)
