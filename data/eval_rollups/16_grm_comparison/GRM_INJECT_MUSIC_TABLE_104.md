# CodecAttack vs GRM — GRM behaviors and evaluation prompt (n=104)

Qwen2-Audio on GRM's 104 `Explicit_Advbench` test behaviors, with GRM's
`PAPER_PROMPT` as the common evaluation prompt. CodecAttack also uses that
prompt during optimization; its behavior-specific targets are seeded from
GRM's test behaviors. GRM trains its universal tile under `main.py`'s own
matched prompt and evaluates under `generate.py`'s `PAPER_PROMPT`.

CodecAttack uses the benign `music:calm_1` waveform and injects
behavior-specific affirmative compliance. GRM starts from a TTS waveform that
audibly speaks each harmful request and applies universal PGD in log-Mel
feature space.

CodecAttack: EnCodec latent-space perturbation, eps=1.0, 500 steps. GRM: the
reproduced 100-epoch K=48 tile. JSR is evaluated on the full continuation.
Llama Guard 3 and 4 use GRM's `unsafe AND >=50 words` rule.

## JSR by classifier and delivery channel (%)

The baseline is GRM's JSR on its original, unperturbed TTS request audio,
under the same evaluation prompt, behaviors, model, and judge. It is the
single reference for both attack columns. CodecAttack uses music as its
carrier, so a difference from this reference would not measure its causal
attack effect.

| judge | channel | baseline (GRM original audio) | ours attack | GRM attack |
|---|---|---:|---:|---:|
| HarmBench | direct / Mel | 18.27 | 80.77 | 72.12 |
| HarmBench | Opus 64 | 18.27 | 67.31 | 25.96 |
| HarmBench | Opus 32 | 22.12 | 0.00 | 23.08 |
| Llama Guard 3 | direct / Mel | 19.23 | 92.31 | 92.31 |
| Llama Guard 3 | Opus 64 | 20.19 | 89.42 | 30.77 |
| Llama Guard 3 | Opus 32 | 24.04 | 70.19 | 25.96 |
| Llama Guard 4 | direct / Mel | 16.35 | 81.73 | 82.69 |
| Llama Guard 4 | Opus 64 | 18.27 | 78.85 | 26.92 |
| Llama Guard 4 | Opus 32 | 20.19 | 48.08 | 24.04 |
| GLiGuard | direct / Mel | 74.04 | 98.08 | 79.81 |
| GLiGuard | Opus 64 | 75.96 | 84.62 | 77.88 |
| GLiGuard | Opus 32 | 73.08 | 8.65 | 75.00 |

GLiGuard entries are `response_safety == unsafe` rates rather than verified
jailbreak success. Its baseline on native speech uses the raw TTS waveform
(`wav_raw`, 77/104 flagged). Its Opus baselines use unperturbed Mel features
inverted to a waveform with Griffin-Lim before compression (Opus 64: 79/104;
Opus 32: 76/104). GLiGuard frequently labels Qwen2-Audio refusals unsafe on
this corpus; see `GLIGUARD_RESULTS.md` for the false-positive analysis.

`direct / Mel` is not the same physical rung: ours is an attacked waveform fed
directly to Qwen2-Audio, whereas GRM's number injects the perturbation directly
into the model's Mel representation. GRM's Mel attack is not directly
deliverable as audio; its waveform effect before Opus is only +3.85 pp under
HarmBench, +3.85 pp under Guard 3, and +2.88 pp under Guard 4.

The old version of this table incorrectly combined HarmBench values for ours
with the paper's Llama-Guard-3 values for GRM. The rows above keep the judge
fixed within every comparison.

## Interpretation

HarmBench is the primary result. The attacked JSRs are descriptive
cross-method comparisons, not paired attack effects: CodecAttack embeds the
behavior in music, while GRM perturbs audio that speaks the behavior. The
Guard-3/4 values are especially sensitive to the affirmative-prefix prompt.

Sources: `verdicts.json`, `verdicts_guard3.json`,
`verdicts_guard4.json`, and `verdicts_gliguard.json` under
`codec_attack/results_qa/grm_inject_music_qwen2/`; GRM values from
`JUDGE_COMPARISON.md`'s paper-prompt Qwen2-Audio rows.
