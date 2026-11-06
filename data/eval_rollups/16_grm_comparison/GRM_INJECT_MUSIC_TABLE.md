# CodecAttack vs GRM — preliminary comparison (n=25 versus n=104)

Qwen2-Audio, n=25 of GRM's Explicit_Advbench test seeds for CodecAttack, eps=1.0, steps=500. CodecAttack carrier = benign music; GRM carrier = TTS audio speaking the request. Prompt = GRM PAPER_PROMPT (used for CodecAttack optimization and evaluation). CodecAttack target = behavior-specific affirmative compliance. Judge = HarmBench for both attacks.

GRM uses universal Mel PGD; CodecAttack uses behavior-specific EnCodec latent-space perturbation. The attack carriers also differ.

## JSR per delivery channel (%)

The baseline is GRM's original, unperturbed TTS request audio. It is shown
once as a reference for both attacks. CodecAttack's unperturbed carrier is
music, so a difference from this baseline would not be its attack effect.
These CodecAttack results use 25 behaviors; the GRM reference and attacked
results use 104, so this smaller table is preliminary.

| channel | baseline (GRM original audio) | ours attack | GRM attack |
|---|---:|---:|---:|
| direct | 18.27 | 80.00 | 72.12 |
| opus64 | 18.27 | 60.00 | 25.96 |
| opus32 | 22.12 | 0.00 | 23.08 |

*(GRM columns: JUDGE_COMPARISON.md, Qwen2-Audio, paper prompt, HarmBench. GRM's `direct` row is its Mel rung — the representation GRM optimizes in; ours is the attacked waveform fed directly. GRM's mel is not codec-deliverable.)*

## Median CodecAttack response words (length is a known JSR confound)

| channel | attack |
|---|---:|
| direct | 82 |
| opus64 | 80 |
| opus32 | 76 |
