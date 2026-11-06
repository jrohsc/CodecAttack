# Audio-Quality Reporting — Paper TODO

Goal: address advisor's "is the audio quality bad?" concern by reporting
SNR / PESQ / STOI consistently across the paper. Anchor the codec-robust
trade-off with a no-codec-EOT ablation on S3a (Qwen2-Audio).

Paper root: `${REPO_ROOT}/paper-neurips`

## Source data

- Multi-bitrate EOT sweep (already done):
  `0_all_combined/results_codec_robust/_quality/aggregate_table.md`
  (1193 wavs, all scenarios × 3 models × ε ∈ {0.5, 1.0, 1.5})
- No-codec-EOT ablation (running on user's GPU):
  `0_all_combined/results_codec_non_robust/03a_ai_detection_bypass/qwen2_audio/eps_{0.5,1.0}/`
  → re-run `compute_audio_quality.py --root 0_all_combined/results_codec_non_robust`
  once it finishes.

---

## 1. Appendix → new `tab:snr_pesq` (load-bearing table)

- [x] File: create `paper-neurips/tables/tab_snr_pesq.tex`
- [x] Content: the multibitrate sweep, distilled to 12 rows
  (4 scenarios × 3 ε, 3-model averages). Columns: scenario, ε, n,
  SNR, SI-SDR, PESQ-WB, STOI, ESTOI, LSD.
- [x] Placed in `appendix.tex` as new `\section{Audio Quality}` right
  after the waveform-baseline appendix section, before Background.
- [x] Caption notes: STOI only meaningful on speech carriers (S1/S2);
  on music STOI/ESTOI are coarse proxies. Also flagged the bimodal
  S3b SNR / SI-SDR mismatch (DC/scale offset, not perceptual noise).
- [x] Cross-ref from `experiments.tex:21` already existed; resolves now.

## 2. Methodology / Experiments → uncomment quality-metrics paragraph

- [x] File: `paper-neurips/sections/experiments.tex` line 21
- [x] Removed the `%` prefix on the paragraph that starts
  "A note on quality metrics."
- [x] Added the STOI sentence: "...we additionally report STOI on the
  speech-carrier scenarios (S1, S2) to confirm carrier intelligibility
  is preserved at all reported $\epsilon$ (STOI ≥ 0.95 at ε=0.5)."
- [x] No structural change.

## 3. Experiments → Ablation Studies (§6.4) → new "Codec-EOT ablation"

- [x] File: `paper-neurips/sections/experiments.tex` (after Adversarial
  channel capacity, before §6.3 / §6.4 close)
- [x] Added paragraph titled "Price of codec robustness."
- [x] New table: `paper-neurips/tables/tab_codec_eot_ablation.tex`
  Two rows (no-EOT vs multi-bitrate EOT) × {SNR, PESQ, STOI,
  ASR direct, Opus 32k, Opus 64k, MP3 64k, MP3 128k} for
  S3a / Qwen2-Audio / ε ∈ {0.5, 1.0}.
- [!] Narrative beats — REVISED FROM TODO HYPOTHESIS:
  * **Original TODO claim** ("EOT trades ~6 dB SNR / ~0.5 PESQ for
    cross-codec generalization") is **NOT what the data shows.**
  * Actual finding: at matched ε, audio quality is essentially
    identical between no-EOT and EOT (SNR Δ ≤ 0.2 dB, PESQ Δ ≤ 0.02,
    STOI Δ ≤ 0.005). Both attacks share the same latent ε-ball; EOT
    redirects *where* energy lands in latent space, not *how much.*
  * The price is on the success axis, not the quality axis: no-EOT
    saturates direct ASR (100%) but collapses on Opus 32k (0%) and
    crawls on MP3 64k (35.0% at ε=0.5); EOT gives back direct ASR
    (35% at ε=0.5; 55% at ε=1.0) in exchange for 17.5% / 62.5%
    on Opus 32k and 57.5% / 90.0% on MP3 64k.
  * This is a *stronger* answer to "is the quality bad?" than the TODO
    anticipated: at our headline ε=0.5, the music-carrier operating
    point is SNR ≈ 5 dB / PESQ ≈ 2.0 *regardless* of codec robustness;
    the EoT schedule does not move it.
- [x] Block resolved: S3a no-EoT run done (Qwen2-Audio, ε=0.5 and ε=1.0,
  both 40/40 pairs).

## 4. Related Work → SNR/PESQ columns in `tab:previous_work_comparison`

- [x] File: `paper-neurips/tables/tab_previous_work_comparison.tex`
- [x] Added 2 columns: SNR (dB), PESQ.
- [x] Populated:
  * Carlini & Wagner 2018 — "−31 peak" / —
  * CommanderSong 2018 — 14.0–18.6 / —
  * Qin et al. 2019 — "imperceptible" / —
  * Ours — 8.0 (S1+S2 mean at ε=0.5) / 2.66 (S1+S2 mean at ε=0.5)
  * (Zheng 2021 dropped — citation key not in bib; not load-bearing)
- [x] Caption note added: "Most prior work documents either quality OR
  codec robustness, but not both — the contrast this column makes
  visible is the point of this paper."

---

## What NOT to do

- Do NOT add an "Audio Quality" subsection in §6.2 Main Results — that
  dilutes the defense-breaking framing (per
  `feedback_benchmark_as_secondary.md`).
- Do NOT restructure the section order. The advisor's concern is fully
  addressed by tables (1)+(3) plus uncommenting (2).
- Do NOT use ε=1.5 numbers as the headline; they live in the appendix.
  Headline is ε=0.5 (matches prior-work SNR/PESQ regime closest).

## Order of operations

- [x] Step 0: run multi-bitrate EOT quality sweep (DONE)
- [x] Step 1: do (1) and (2) — no runtime cost (DONE 2026-05-02)
- [x] Step 2: ran compute_audio_quality on results_codec_non_robust
  (40+40 pairs S3a Qwen2-Audio ε∈{0.5,1.0}); did (3) (DONE 2026-05-02)
- [x] Step 3: (4) SNR/PESQ columns in prior-work table (DONE 2026-05-02)
