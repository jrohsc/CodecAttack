# Results map: paper table / figure -> shipped rollup

Every number in the paper comes from a JSON/JSONL rollup in `data/eval_rollups/`.
This file is the original paper index. For the three-seed primary table, completed reviewer controls, and current evidence limits, use [the reviewer artifact update](REBUTTAL_MASTER_CHECKLIST.md). Adversarial audio is not redistributed (see the README);
each rollup keeps the *metadata* sidecars that the attack and eval drivers wrote
next to the WAVs, which is what the table builders read.

## Scoring convention

For the primary three-seed table, success requires the full nonempty target string, lowercased with whitespace collapsed and punctuation retained, to appear in the decoded output. Other historical tables use their documented builder conventions. Two files per attacked
clip carry the evidence:

| file | holds |
|---|---|
| `audio/<clip>_summary.json` | `final_output` (clean delivery) and `opus_robustness.<kbps>.output` |
| `audio/<clip>_codecs.json`  | `mp3_robustness.<kbps>` and `aac_robustness.<kbps>` |
| `neural_delivery_eval_<model>.json` | the DAC / EnCodec neural re-synthesis columns |
| `results.jsonl` | one `attack` event per pair: carrier, target, eps, channel mode, EoT bitrate grid, wall-clock |

`matches_target` in those files is the *lenient* criterion (exact OR WER <= 0.5);
the paper reports the strict substring criterion, so recompute rather than
reading `matches_target` if you want the published numbers.

## Main body

| Paper item | Rollup |
|---|---|
| `tab:q5_eot_codec_matrix` (`tab_different_codec_trained`) | Opus row: `06_main_multiseed/01_finance_voice_agent/qwen25_omni/eps_1.0_seed{1,7,42}/`; other rows: `07_eot_train_codec/{mp3,aac,ensemble,encodec,dac}/01_finance_voice_agent/qwen25_omni/eps_1.0_seed{1,7,42}/`. `ensemble` is the **Opus+AAC-LC** proxy. |
| `tab_eps_sweep`, S1/S2/S3 main tables | `06_main_multiseed/{01_finance_voice_agent,02_interview_screening,03a_ai_detection_bypass,03b_copyright_bypass}/<model>/eps_{0.5,1.0,1.5}*/` |
| `tab_snr_pesq`, `tab_SNR_comparison` | `06_main_multiseed/_quality/`, `09_waveform_pgd/.../_quality*`, `14_neural_resynthesis/*/` |
| `tab_latent_vs_waveform`, `tab_latent_vs_waveform_16-32` | `09_waveform_pgd/` (+ `09_waveform_pgd/_n50_latent_vs_waveform/` for the n=50 matched-rate arm) |
| `tab_mimi_results`, `tab_mimi_audio_quality` | `14_neural_resynthesis/{MIMI,DAC}/`, plus the pre-existing `03b_copyright_bypass/{mimi_eps0.2,dac_eps0.6194,encodec_eps1.0}/` |
| `tab_no_robust_S3` + `figure_non_robust_ablation` | `08_non_robust/` |
| `tab_watermark` | `_results_watermark/` and `_results_watermark_channel_sweep/` |
| `tab_ensemble_attack` (Qwen2-Audio + Omni-7B + AF3; 4 held-out incl. DeSTA2.5) | `11_ensemble_transcribe/benchmark_ensemble3_opuseot_*` and `desta_ensemble_matched_full224*` |
| `tab_ensemble_attack_alternative` (Omni-7B + AF3 + Phi-4; 3 held-out) | `05_ensemble_omni7_af3_phi4/` |
| `tab_encoder_residual_aac`, `tab_bark_band_speech_vs_music`, `fig_per_bark_*`, `fig_decoder_per_dim_heatmap`, `fig_three_trace`, `fig_cosine_survival` | `13_mechanistic/{encoder_residual_aac,per_bark_band,per_bark_band_speech_vs_music,decoder_freq,cosine_through_codec}/` (`_paper_artifacts/` holds the generated `.tex`) |

## Appendix

| Paper item | Rollup |
|---|---|
| `tab:waveform_controls`, `tab:waveform_controls_quality`, `tab_bandlimited_control` | `15_reviewer_controls/` (band-limited LP400 / 4 kHz / SNR-matched arms, `snr/` holds the per-arm JSONL + summaries) |
| `tab:neural_resynthesis_scenarios`, `tab:neural_resynthesis_adaptation`, `tab_neural_reencode_defense` | `14_neural_resynthesis/{DAC,MIMI,SNAC}/` + the `dac` EoT row of `07_eot_train_codec/dac/` (the adaptive counter-attack) |
| `tab:grm_protocol`, `tab:grm_comparison`, `tab:grm_transfer`, `tab_vs_GRM` | `16_grm_comparison/` (`results/`, `results_pp/`, `audit/`, and the `*_RESULTS.md` / `JUDGE_COMPARISON.md` write-ups) |
| `fig:capacity_cliff`, `ablation_length` | `12_capacity_ablation/{expA_words,expB_duration}/` + `rollup_expA_from_summaries.json` |
| `tab_cross_model` (transfer null) | `06_main_multiseed/` cells scored by `scripts/campaigns/cross_model_transfer_eval.py` |
| `tab:victim_ensemble` S1 co-optimization | `10_ensemble_victim_s1/` |

## Reproducing the primary three-seed statistics

Run `python scripts/build_main_results_tables.py`. This recomputes per-seed ASR and mean/population SD for all primary latent/waveform cells, checks the submitted numeric reference, and writes tables plus audit CSVs under `results/tables/`. See [the reviewer update](REBUTTAL_MASTER_CHECKLIST.md) for the separate matched-control audit and known AAC provenance discrepancy.

For this primary table, normalization lowercases and collapses whitespace while retaining punctuation. Do not use a punctuation-stripping example or the lenient `matches_target` field to reproduce these statistics.

## What is deliberately not here

- **Adversarial audio** (~10k WAVs). Regenerate with the drivers in `scripts/`.
- **Side campaigns that back no published claim**: the encoder-transfer probe,
  the pool-scaling ladder, the eps-ladder and 64/128-EoT ensemble variants, and
  the transcribe-prompt matrix. They were run, they are negative or superseded,
  and shipping them next to the paper's rollups would invite misreading.
- **Third-party datasets** that the GRM comparison was evaluated against
  (AIR-Bench, AdvBench audio renders) and the raw judge-audit dumps over them.
  `16_grm_comparison/` keeps the scored outputs and the audit summaries.
- **Model weights and third-party model code** (see the README).

## Caveat on `scripts/campaigns/`

`scripts/` holds the curated, release-wired drivers: they resolve paths through
`scripts/config.py` and read `data/eval_rollups/`. `scripts/campaigns/` is the
raw campaign layer exactly as it ran on our cluster — SLURM `sbatch_*.sh`
launchers, one-off `chain_*`/`sweep_*` shell drivers, and the analysis scripts
that produced the mechanistic figures. Their absolute paths have been replaced
with `${REPO_ROOT}` / `${PROJECT_ROOT}` / `${MODEL_ROOT}` placeholders, so they
are documentation of what was run, not drop-in runnable. Point them at your own
clone before use.
