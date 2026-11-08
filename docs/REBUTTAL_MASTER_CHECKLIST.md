# Reviewer-requested artifact update

This release snapshot addresses reviewer requests with reproducible records and explicit limits. Completed experiments and pending studies are listed separately. The original paper rollups remain under `data/eval_rollups/`; new evidence is under [`data/reviewer_update/`](../data/reviewer_update/).

| Request | Released evidence | Remaining work |
|---|---|---|
| Three-seed primary statistics and matched-control provenance (6hsC) | CPU audit, per-seed/aggregate CSVs, submitted numeric reference, 40-pair control audit | Historical AAC discrepancy remains unresolved |
| Competitively tuned unrestricted waveform control (6hsC) | Fresh 40-recording, three-seed evaluation and quality records; development/calibration drivers; failed quality gates | Equal-tuning success–quality sweep and other-model controls are incomplete |
| Clean accuracy and conditional decision flips (6hsC, 1b9A) | Decision rubric; explicit distinction between raw-carrier decisions and target-string hits | Human annotations and validated conditional flip rates pending |
| Independent content and source distribution (6hsC, 1b9A) | Original cohort metadata; separate 150-item content manifest and protocol | Human content/listening review and full attack evaluation pending |
| Transfer at the primary budget (izWN, 1b9A) | Completed three-member ε=1 paper-domain receiver evaluations, 345 seeded trials | Matched jazz ε=1/10 comparison over seeds 1, 7, 42 is queued/incomplete |
| Audio quality (izWN) | Saved objective quality measurements and failed matching gates | No human listening results; equal SNR does not establish perceptual equivalence |
| Mechanism and terminology (izWN, 1b9A) | Scope clarified below; diagnostic checkpoint support and three-member driver | Manuscript revision and stronger causal ablations remain separate work |
| Downstream/API relevance (1b9A) | Explicit evidence limits below | No mock-tool execution or commercial/API results |

## Reproduce the released statistics

From the repository root, these commands need Python's standard library only:

```bash
python scripts/build_main_results_tables.py
python scripts/rebuttal/audit_controls.py
python scripts/rebuttal/summarize_reviewer_update.py
```

The main builder writes `results/tables/tab_*_multiseed.tex` and `results/tables/audit/{per_seed.csv,aggregated.csv,audit.json}`. It checks all 168 mean/SD cells against [`the submitted numeric reference`](reference/tab_latent_vs_waveform.tex), verifies complete bundles and carrier-target identities, and fails on missing data or mismatches. SD is the population SD of the three seed-level ASRs, not an estimate of independent-carrier uncertainty. The historical auxiliary single-run builders report point estimates only.

The primary audit lowercases and collapses whitespace while retaining punctuation. It requires the complete nonempty target as a substring and does not use the legacy lenient `matches_target` flag. This reproduces the submitted primary table; a punctuation-stripping example previously in the results map gave different answers on 46 channel records.

The control audit writes `results/rebuttal/audit/controls_audit.json`. Its source is `data/eval_rollups/15_reviewer_controls/s1_baseline_full40_seed1/evaluation_records.jsonl`. On the same 40 carriers and targets, historical latent AAC64 success is 19/40 in the control bundle versus 2/40, 1/40, and 2/40 in the main seeds. Cohort size does not explain the difference. Original historical attack WAVs/configuration provenance are insufficient to establish the cause; the discrepancy is retained, not reconciled by changing outcomes. Historical SNR fields cannot be independently remeasured without those WAVs.

## Completed tuned-control comparison

The fresh S1 comparison uses 40 held-out recordings repeated across seeds 1, 7, and 42. Records include unrestricted waveform, latent, and bandlimited arms; no failed cases are excluded.

| Delivery | Tuned waveform hits | Latent hits |
|---|---:|---:|
| Uncompressed | 119/120 | 101/120 |
| Opus16 | 6/120 | 35/120 |
| Opus24 | 45/120 | 65/120 |
| Opus32 | 96/120 | 78/120 |
| Opus64 | 118/120 | 92/120 |
| Opus128 | 118/120 | 100/120 |
| MP3-64 | 69/120 | 61/120 |
| AAC-64 | 79/120 | 48/120 |

The protocol relaxed the original 1 dB SNR tolerance to 1.5 dB after development residuals were inspected. This is a revised protocol. Waveform calibration adds optimization cost. Objective quality matching failed on all ten development examples in the recorded gates. These outcomes support bitrate-dependent comparisons, not blanket parameterization superiority or perceptual equivalence. The separately released equal-tuning protocol gives both methods four budgets and four learning rates; it is still a development study with no completed held-out comparison.

## Transfer, content, and interpretation

The completed ε=1 paper-domain ensemble optimizes Qwen2.5-Omni, Audio Flamingo, and Phi-4. Qwen2-Audio and Kimi are held out; each has 0/345 target hits on every evaluated channel. The export includes member and receiver outputs and objective quality. This cohort covers S1/S2/S3a, not copyright S3b. The matched jazz extension uses one carrier and 56 target identities per seed; repeated seeds do not establish content diversity.

[`configs/paper_cohorts.json`](../configs/paper_cohorts.json) records 50 banking recordings from five scripts, 25 interview recordings from five scripts, eight AI-music recordings, and nine copyright recordings. There are 160 carrier-target pairs across 92 original recordings. Task text accompanies audio in the user message; legacy filenames do not establish system-role placement. Original generation/acquisition provenance is incomplete.

The new 150-carrier manifest is separate: 100 speech recordings and 50 music tracks, with ten development and forty evaluation items per domain. Automated integrity/transcription checks passed locally; human content validation is pending. Music generator and vocal/instrumental status are confounded. Audio is not included in this metadata update.

We describe the measured behavior as targeted adversarial audio attacks. EoT is an established framework; the evaluated adaptation combines codec-latent optimization and real-forward channel transformations with surrogate gradients. Target-string reproduction does not establish instruction-priority override, an initially correct decision flip, or a downstream tool action. Product names provide motivation only. No participant ratings, real transactions, hiring actions, or commercial/API efficacy results are claimed.

## Artifact boundaries

[`export_manifest.json`](../data/reviewer_update/export_manifest.json) records original and exported hashes. Machine-specific paths are replaced with placeholders; nonfinite quality values become JSON null. Recorded WAV paths are provenance references, not downloadable assets. Completed evidence snapshots are fixed; pending protocols are not results. Internal job IDs, local scheduling logs, obsolete plans, and draft reviewer responses are not part of this update.
