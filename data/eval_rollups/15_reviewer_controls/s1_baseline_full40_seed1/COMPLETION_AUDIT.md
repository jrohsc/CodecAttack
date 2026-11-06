# S1 ≤4 kHz waveform control: completed paired comparison

## Frozen protocol

- Victim: Qwen2-Audio-7B-Instruct, snapshot `0a095220c30b7b31434169c3086508ef3ea5bf0a`; the bank system prompt is recorded verbatim in `reviewer_controls/config.json` and every evaluation row.
- The 50 exact carrier–target pairs and their split are in `../S1_SPLIT_MANIFEST.csv`: 10 development and 40 held-out. Each source recording occurs once; the cross-split source-recording intersection is empty. The four-arm estimate uses only the 40 held-out IDs. The separate ten-case ≤4 kHz development summary is `../s1_4khz_full50_seed1/eval/summary.json`.
- The learning-rate choice was frozen before held-out ASR in `../pilot_s1_seed1/frozen/learning_rate.json`. Four development cases supplied the grid. For both tuned waveform arms, alpha/epsilon 0.02 won by exact target-substring success across clean and Opus 16/24/32/64/128; D scored 13/24 and C 20/24. MP3/AAC and held-out cases were excluded. The chosen pilot WAVs were below the latent distortion budget on all four cases (minimum signed SNR difference +1.27 dB for D and +2.12 dB for C). Candidate learning rates were not all distortion-matched, so this is effectiveness subject to a distortion ceiling, not a matched-SNR learning-rate effect.
- Both C and D use Adam, zero initialization, seed 1, 1,000 steps, 30% clean warmup, then alternating direct and one uniformly sampled Opus-BPDA update over 16/24/32/64/128 kbps, with the last iterate saved. They differ only in perturbation parameterization and their distortion-calibrated epsilon. The historical B arm has fixed epsilon 0.06082, alpha/epsilon 0.2, 20% warmup and codec update on every later step; it also uses the legacy carrier normalization. B and C are distinct attacks. The published A arm can restore its last periodic successful checkpoint, whereas B/C/D save the last iterate. Those differences limit causal attribution to parameterization alone.

## Audio and calibration audit

All 160 final WAVs are 10-second, 24 kHz, mono PCM_16. They and the original carrier were saved and reloaded before measuring `10 log10(sum(carrier²)/sum((attacked-carrier)²))`. The common original is `load_music_by_name`'s 24 kHz mono carrier, peak-normalized to 0.95 before PCM serialization. The original carrier bytes agree across the latent and D references on all 40 IDs. Sample rate, duration, alignment, normalization, carrier reference and PCM serialization were checked; no measurement bug explained the five residuals. p001, p003, p005 and p022 were too weak (positive waveform-minus-latent SNR); p014 was too strong. All five were range-projection-bound, so the earlier log-linear epsilon updates were not reliable. The first two refinement rounds left 35/40 within ±1 dB, with a 6.40 dB worst residual.

The additional distortion-only calibrations optimized the same frozen attack for only the affected IDs: five in round 4, then p001, p005 and p014 once more each. The selected artifact for each failing ID minimized the absolute saved/reloaded common-carrier SNR residual among documented candidate runs. Selection read no ASR output, and the chosen WAV was fixed before its channel evaluation. No waveform was postscaled, no case was dropped, and the tolerance stayed ±1 dB. The selected residuals are p001 −0.201, p003 +0.010, p005 −0.400, p014 +0.496 and p022 −0.657 dB. The full 40-case gate is **40/40**, maximum absolute residual **0.967 dB**. All attempts, epsilon values and selected paths are in `../s1_4khz_full50_seed1/frozen/bandlimited_4khz.evaluation.r4_selection.json`. The eight added optimizations consumed 55 minutes 11 seconds of allocated GPU time across four jobs (64738260, 64738709, 64739200, 64739438), plus scoring and local audit time.

The ≤4 kHz restriction is applied to the additive 16 kHz perturbation by an exact rFFT-bin mask (`k × 16000 ≤ 4000 × n`) in `robust_waveform_attack.py`; there is no transition band in that projection. A single global nonnegative scale enforces both the L∞ budget and the valid-amplitude range. A pointwise clamp after projection would create high-frequency harmonics and is avoided. The final waveform is resampled to 24 kHz and serialized as PCM_16. On the selected files, the mean fraction of energy above 4 kHz in *attack minus attacker reference* is 1.56×10⁻⁶; the 16 kHz optimizer-domain fraction averages 2.18×10⁻¹⁴. Attack minus the full-band original carrier averages 0.00298 because the waveform path also resamples the carrier through 16 kHz. The two fractions answer different questions and are both recorded per case. The mean PCM full-scale clipping rate is 2.29×10⁻⁶ samples. The cutoff remained 4 kHz throughout.

## Evaluation and interpretation

The same selected WAV per method and ID went through clean delivery, Opus 16/32/64/128, MP3 64 and AAC-LC 64; Opus 24 is included as an additional recorded training-grid channel. The common evaluator used the same model snapshot, prompt, preprocessing, FFmpeg codec functions, length normalization and success rule for every arm. Success is the complete target as a contiguous substring after lowercasing and whitespace normalization, retaining punctuation. Existing A/B results were reused after checking all 40 IDs, path existence, unchanged WAV modification times, prompt, 24 kHz / 240,000-sample identity and all eight channel outputs. C/D were freshly scored. Every success flag was recomputed from the saved generation text during final validation.

The four-arm summary is in `baseline_report.md` and `baseline_report.json`; `per_example_records.csv` has 160 rows with carrier-referenced SNR, signed residual, L2/RMS/peak perturbation, clipping rate, both spectral fractions where relevant, waveform SHA-256 and channel flags. `evaluation_records.jsonl` retains all generation text. The common-clean-success intersection has five IDs, with per-channel counts in the report. No evaluation failure was excluded.

At matched distortion, D is 25/40 clean and 3/40 at Opus 16; A is 34/40 clean and 4/40 at Opus 16. D therefore does not close the CodecAttack gap in this S1/Qwen2-Audio/seed-1 regime. Conditional Opus-16 survival is 3/25 for D versus 4/34 for A, but those denominators differ because D loses more clean successes. C is 40/40 clean and 11/40 at Opus 16. These are paired descriptive estimates for this victim, subset, seed and distortion regime only.

### What arm C establishes about the threat model

“Tuned unrestricted” means **no 4 kHz perturbation mask**. It still optimizes an additive perturbation to the 16 kHz working copy of the carrier, with a per-example L∞ budget and valid-amplitude projection; it does not replace the recording or modify the victim. Its optimizer has no perceptual penalty (`perceptual_weight=0`). Thus C fits the paper's formal white-box, prerecorded-audio, digital-delivery attacker, but the results do not establish the paper's stronger “hidden command in benign-sounding carrier” condition. The 16 kHz working rate also means “unrestricted” does not mean arbitrary 24 kHz spectral content.

To probe that condition, `reviewer_controls.measure_s1_speech_quality` measured PESQ-WB and STOI on the same 40 saved/reloaded pairs against the same carrier (`speech_quality.csv`, `speech_quality_summary.json`). C averages PESQ-WB **1.082** and STOI **0.883**; A averages **1.665** and **0.895**. A has higher PESQ on **all 40** pairs. These metrics do not replace a listening study, but matched SNR clearly did not equalize objective speech quality. C should be described as a strong **waveform optimization control** whose stealth is unverified, not dismissed as outside the formal attacker capabilities. Its superior ASR also means the older claim that waveform methods remain below 26% ASR or that the latent domain alone explains the A/B gap is unsupported for this S1/Qwen2-Audio comparison.

## Reproduce and verify without rerunning attacks

From `Qualcomm/codec_attack`:

```bash
export PYTHONPATH="$PWD/0_all_combined"
CA=${WORK_ROOT}/miniconda3/envs/codec-attack/bin/python
$CA -m reviewer_controls.finish_s1_4khz
$CA -m reviewer_controls.select_s1_4khz_r4
$CA -m reviewer_controls.finalize_s1_baselines
$CA -m reviewer_controls.measure_s1_speech_quality
$CA -m pytest 0_all_combined/reviewer_controls/test_controls.py -q
```

The targeted optimization scripts and their frozen maps are `sbatch_s1_4khz_targeted_r4.sh`, `sbatch_s1_4khz_targeted_extra.sh` and `../s1_4khz_full50_seed1/frozen/epsilon_bandlimited_4khz.evaluation.*.json`. Running the attack jobs again is unnecessary for reproducing the completed tables; the selected PCM files and full per-example evaluator JSONL are preserved.
